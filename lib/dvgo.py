import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
import math


def gaussian_filter(input, kernel_size=3, sigma=2, border_type='replicate'):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat((kernel_size, kernel_size)).view(kernel_size, kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    return kornia.filters.filter3d(input, kernel=gaussian_kernel.unsqueeze(0), border_type=border_type)


## adding the filter before downsampling usage.
def gaussian_kernel_filter(kernel_size=3, sigma=2, channels=3, padding_mode='reflect'):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat((kernel_size, kernel_size)).view(kernel_size, kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 3-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 3d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
    gaussian_filter = nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=int((kernel_size-1)/2), padding_mode=padding_mode)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def mean_kernel_filter(kernel_size=3, channels=3, padding_mode="reflect"):
    mean_kernel = torch.ones(kernel_size, kernel_size, kernel_size)
    mean_kernel = mean_kernel/torch.sum(mean_kernel)
    mean_kernel = mean_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    mean_kernel = mean_kernel.repeat(channels, 1, 1, 1, 1)
    mean_filter = nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=int((kernel_size-1)/2), padding_mode=padding_mode)
    mean_filter.weight.data = mean_kernel
    mean_filter.weight.requires_grad = False
    return mean_filter


def identity_kernel_filter(kernel_size=1, channels=3, padding_mode="reflect"):
    identity_kernel = torch.ones(kernel_size, kernel_size, kernel_size)
    identity_kernel = identity_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    identity_kernel = identity_kernel.repeat(channels, 1, 1, 1, 1)
    identity_filter = nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=int((kernel_size-1)/2), padding_mode=padding_mode)
    identity_filter.weight.data = identity_kernel
    identity_filter.weight.requires_grad = False
    return identity_filter


'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False, pre_act_density=False, in_act_density=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=2, rgbnet_width=16,
                 posbase_pe=0, viewbase_pe=4,
                 feature_vector_dim=4, view_direction_dim=3,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        self.pre_act_density = pre_act_density
        self.in_act_density = in_act_density
        if self.pre_act_density:
            print('dvgo: using pre_act_density may results in worse quality !!')
        if self.in_act_density:
            print('dvgo: using in_act_density may results in worse quality !!')

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)
        self.relu = nn.LeakyReLU(0.2)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe,
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                # which define in config/default, change from 12 to 7. [:3] is diffusion color. [3:] feature vector.
                self.k0_dim = rgbnet_dim
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = 3+feature_vector_dim + (3*viewbase_pe*2+3) ## total is 34

            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0.shape)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            self.mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres,
                    usage='mask').to(self.xyz_min.device)
            self._set_nonempty_mask()
            self.surf_cache = MaskCache(
                path=mask_cache_path,
                mask_cache_thres=mask_cache_thres,
                usage='surf').to(self.xyz_min.device)
        else:
            self.mask_cache = None
            self.nonempty_mask = None
            self.surf_cache = None

        # Initial filter for mipmap
        self.kernel_size = 5 # was 3
        self.pd_mode = "circular"
        self.sigma = 0.577

        self.lod_coef = 0.5
        # self.lowpass_filter = gaussian_kernel_filter(channels=self.k0.shape[1], kernel_size=self.kernel_size, sigma=self.sigma, padding_mode=self.pd_mode)
        self.lowpass_filter = mean_kernel_filter(channels=self.k0.shape[1])

    def _set_grid_resolution(self, num_voxels):
        import math
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        new_size = 2**round(math.log2(math.pow(num_voxels, 1/3)))
        self.world_size = torch.tensor([new_size, new_size, new_size])
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        self.voxel_size_rec = self.voxel_size

        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'generate_weights_for_neighbors': self.fast_color_thres,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)['mask'][None, None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                # step = stepsize * self.voxel_size_rec * rng
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation(self):
        tv = total_variation(self.activate_density(self.density, 1), self.nonempty_mask)
        return tv

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    def downsample_grid(self, grid, size_tuple):
        lowpass_filter = mean_kernel_filter(channels=grid.shape[1])
        return  F.interpolate(lowpass_filter(grid), size=size_tuple, mode="trilinear", align_corners=True)

    def grid_sampler(self, xyz, *grids, lod=None, lowpass_filter=None, mode=None, align_corners=True):
        ## only operate the color feature grids

        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'

        if (lod is None) or len(lod) == 0 or lod.sum() <= 0.:
            ## following the original way
            shape = xyz.shape[:-1]
            xyz = xyz.reshape(1,1,1,-1,3)
            ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
            ret_lst = [
                F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
                for grid in grids
            ]
            if len(ret_lst) == 1:
                return ret_lst[0]
            return ret_lst
        else:
            shape = xyz.shape[:-1]
            xyz = xyz.reshape(1, 1, 1, -1, 3)
            ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
            lod[lod < 0] = 0

            ret_lst = torch.zeros(shape[0], grids[0].shape[1], device=xyz.device)
            lod_lower = torch.floor(lod).int()
            values_lower, number_lower = lod_lower.unique(return_counts=True)
            assert torch.max(lod) > 0.0
            grids_list = [grids[0]]
            current_res = grids[0].shape[-1]
            for lod_lvl in range(torch.max(values_lower).item()+2):
                if lod_lvl == 0:  ## level 0
                    continue
                tmp_res = current_res // 2
                # tmp = F.interpolate(filter(grids_list[-1]), size=(tmp_res, tmp_res, tmp_res), mode="trilinear", align_corners=True)
                tmp = self.downsample_grid(grid=grids_list[-1], size_tuple=(tmp_res, tmp_res, tmp_res))
                grids_list.append(tmp)

            mask = (lod == 0)
            count = torch.sum(mask)

            x = F.grid_sample(grids[0], ind_norm[:, :, :, mask], mode=mode, align_corners=align_corners).reshape(grids[0].shape[1], -1).T.reshape(-1, grids[0].shape[1]).squeeze()
            target_shape = ret_lst[mask].shape
            ret_lst[mask] = x.reshape(target_shape)
            for level in values_lower:
                lower_grid = grids_list[level]
                upper_grid = grids_list[level+1]
                mask = (lod != 0) * (lod_lower == level)

                upper_weights = lod[mask] - lod_lower[mask]
                lower_weights = 1 - upper_weights
                pts = ind_norm[:, :, :, mask]

                _lower = F.grid_sample(lower_grid, pts, mode=mode, align_corners=align_corners)
                _lower = _lower.reshape(lower_grid.shape[1], -1).T
                _lower = _lower.reshape(pts.shape[-2], lower_grid.shape[1])
                _lower = lower_weights * torch.transpose(_lower, 0, 1)

                _upper = F.grid_sample(upper_grid, pts, mode=mode, align_corners=align_corners)
                _upper = _upper.reshape(lower_grid.shape[1], -1).T
                _upper = _upper.reshape(pts.shape[-2], upper_grid.shape[1])
                _upper = upper_weights * torch.transpose(_upper, 0, 1)

                ret_lst[mask] = (_lower + _upper).T
                count += torch.sum(mask)

            assert count.item() == shape[0]
            return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size_rec * rng
        interpx = (t_min[..., None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    @torch.no_grad()
    def query_weights_for_neighbors(self, rays_pts, mask, interval):
        alpha = torch.zeros_like(rays_pts[..., 0])
        if self.surf_cache is not None:
            alpha[~mask] = self.surf_cache(rays_pts[~mask])['alpha']
        else:
            density = self.grid_sampler(rays_pts[~mask], self.density.detach())
            alpha[~mask] = self.activate_density(density, interval)
        weights, alphainv_cum = get_ray_marching_ray(alpha)
        return alphainv_cum, weights

    def compute_lod(self, rays_pts, rays_pts_x, rays_pts_y,
                             sf_mask, sf_mask_x, sf_mask_y):
        sf_mask = torch.repeat_interleave(sf_mask[..., None], rays_pts.shape[-2], dim=-1)
        sf_mask_x = torch.repeat_interleave(sf_mask_x[..., None], rays_pts_x.shape[-2], dim=-1)
        sf_mask_y = torch.repeat_interleave(sf_mask_y[..., None], rays_pts_y.shape[-2], dim=-1)
        dx_vtc = rays_pts - rays_pts_x
        dy_vtc = rays_pts - rays_pts_y
        lod= torch.zeros(sf_mask.shape)

        voxel_size = (self.xyz_max - self.xyz_min) / self.k0.shape[-1]

        ind0 = sf_mask * sf_mask_x * sf_mask_y
        lod[ind0] = self.lod_coef * torch.log2(torch.max(
            torch.sqrt(torch.sum(torch.square(2*dx_vtc[ind0] / voxel_size), -1)),
            torch.sqrt(torch.sum(torch.square(2*dy_vtc[ind0] / voxel_size), -1))
        ))

        ind1 = (sf_mask_x == 0) * (sf_mask_y > 0) * (sf_mask != 0)
        lod[ind1] = self.lod_coef * torch.log2(torch.sqrt(torch.sum(torch.square(2*dy_vtc[ind1] / voxel_size), -1)))
        ind2 = (sf_mask_y == 0) * (sf_mask_x > 0) * (sf_mask != 0)
        lod[ind2] = self.lod_coef * torch.log2(torch.sqrt(torch.sum(torch.square(2*dx_vtc[ind2] / voxel_size), -1)))

        lod[lod < 0.0] = 0.0  ## fix the nan issues
        assert torch.isnan(lod).sum() <= 0
        if (lod > 0).sum() == 0:
            return None, None
        else:
            return lod, self.lowpass_filter


    def forward(self, rays_o, rays_d, viewdirs,
                rays_o_dx=None, rays_o_dy=None, rays_d_dx=None, rays_d_dy=None,
                use_mip=False, global_step=None, **render_kwargs):
        ''' Volume rendering '''
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # update mask for query points in known free space
        if self.mask_cache is not None:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox])['mask'])

        # query for alpha
        alpha = torch.zeros_like(rays_pts[..., 0])
        density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
        alpha[~mask_outbbox] = self.activate_density(density, interval)
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        mask = (weights > self.fast_color_thres)
        k0 = torch.zeros(*weights.shape, self.k0_dim).to(weights)
        if torch.sum(mask) <= 0:
            lodpts, lpf, lodmap = None, None, None
            rgb_diff_pts = render_kwargs['bg'] * torch.ones_like(alpha).to(alpha.device)
            rgb_diff = render_kwargs['bg'] * torch.ones_like(rays_o).to(alpha.device),
            rgb_spec = torch.zeros_like(rays_o).to(alpha.device)
            while type(rgb_diff) is tuple:
                rgb_diff = rgb_diff[0]
            rgb_marched = rgb_diff + rgb_spec
        else:
            if rays_d_dy is None or not use_mip or self.mask_cache is None or self.world_size[0] < 256:
                lodpts, lpf = None, None
            else:
                rays_pts_x, _ = self.sample_ray(
                    rays_o=rays_o_dx, rays_d=rays_d_dx, is_train=global_step is not None, **render_kwargs)
                rays_pts_y, _ = self.sample_ray(
                    rays_o=rays_o_dy, rays_d=rays_d_dy, is_train=global_step is not None, **render_kwargs)

                alphainv_cum_c, weights_c = self.query_weights_for_neighbors(rays_pts, ~mask, interval)
                depth_c = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
                depth_c = (weights_c * depth_c).sum(-1) + alphainv_cum_c[..., -1] * render_kwargs['far']

                alphainv_cum_x, weights_x = self.query_weights_for_neighbors(rays_pts_x, ~mask, interval)
                depth_x = (rays_o_dx[..., None, :] - rays_pts_x).norm(dim=-1)
                depth_x = (weights_x * depth_x).sum(-1) + alphainv_cum_x[..., -1] * render_kwargs['far']

                alphainv_cum_y, weights_y = self.query_weights_for_neighbors(rays_pts_y, ~mask, interval)
                depth_y = (rays_o_dy[..., None, :] - rays_pts_y).norm(dim=-1)
                depth_y = (weights_y * depth_y).sum(-1) + alphainv_cum_y[..., -1] * render_kwargs['far']

                sf_mask_c = depth_c < render_kwargs['far']
                sf_mask_x = depth_x < render_kwargs['far']
                sf_mask_y = depth_y < render_kwargs['far']

                lodpts, lpf = self.compute_lod(rays_pts, rays_pts_x, rays_pts_y,
                                                                    sf_mask_c, sf_mask_x, sf_mask_y)

            if lpf is None:
                lodmap = None
                k0[mask] = self.grid_sampler(rays_pts[mask], self.k0, lod=None, lowpass_filter=None)
            else:
                lodpts = torch.clip(lodpts, min=0., max=4.)
                lodmap = (weights * lodpts.clone()).sum(-1)
                k0[mask] = self.grid_sampler(rays_pts[mask], self.k0, lod=lodpts[mask], lowpass_filter=lpf)

            if self.rgbnet is None:
                rgb_diff_pts = torch.sigmoid(k0)
                rgb_spec = 0.
            else:
                # view-dependent color emission
                ## 3-dim diffuse color
                k0_diffuse_specular = torch.sigmoid(k0) # 7-Dim
                rgb_diff_pts = k0_diffuse_specular[..., :3]

                ## weights: alpha_i * T_i * v_s
                accum_diffuse_specular = (weights[..., None] * k0_diffuse_specular).sum(-2)
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                ## torch.cat((accumulate the feature vector, viewdirs)) per pixel
                feat = torch.cat([
                    viewdirs_emb, # bake
                    accum_diffuse_specular, # 7
                ], -1)
                specular_color_logit = self.rgbnet(feat)
                rgb_spec = torch.sigmoid(specular_color_logit)

            rgb_diff = (weights[..., None] * rgb_diff_pts).sum(-2)
            rgb_marched = rgb_diff + rgb_spec + alphainv_cum[..., -1:] * render_kwargs['bg']
            # rgb_marched = rgb_marched.clamp(0, 1)

        depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
        disp = 1 / depth
        lodmap = torch.zeros_like(depth) if lodmap is None else lodmap
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb_diff_pts,
            'depth': depth,
            'disp': disp,
            'mask': mask,
            'lodmap': lodmap,
            'rgb_diff': rgb_diff,
            'rgb_spec': rgb_spec,
        })
        return ret_dict

    def bake(self, rays_pts, lod=0., **render_kwargs):
        ''' Used for baking'''
        ret_dict = {}

        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)
        if self.mask_cache is not None:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox])['mask'])

        # query for alpha
        alpha = torch.zeros_like(rays_pts[..., 0])
        sigma = torch.zeros_like(rays_pts[..., 0])
        # post-activation
        density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
        alpha[~mask_outbbox] = self.activate_density(density, interval)
        sigma[~mask_outbbox] = F.softplus(density + self.act_shift)

        lodpts = torch.ones_like(rays_pts[..., 0]) * lod
        k0 = self.grid_sampler(rays_pts, self.k0, lod=lodpts, lowpass_filter=self.lowpass_filter)
        rgb = torch.sigmoid(k0)
        ret_dict.update({
            'raw_sigma': sigma,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
        })
        return ret_dict

    def viewdir_fn(self, rgb_features, viewdirs):
        """Used for finetuning"""
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        feat = torch.cat([
            viewdirs_emb,  # 27
            rgb_features,  # 7
        ], -1)
        specular_color_logit = self.rgbnet(feat)
        residual = torch.sigmoid(specular_color_logit)
        return residual


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, usage, ks=3):
        super().__init__()
        st = torch.load(path, map_location=torch.device('cpu'))
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        if usage == 'mask':
            self.register_buffer('density', F.max_pool3d(
                st['model_state_dict']['density'], kernel_size=ks, padding=ks//2, stride=1))
        elif usage == 'surf':
            self.register_buffer('density', st['model_state_dict']['density'])
        else:
            raise NotImplementedError
        self.act_shift = st['MaskCache_kwargs']['act_shift']
        self.voxel_size_ratio = st['MaskCache_kwargs']['voxel_size_ratio']
        self.nearest = st['MaskCache_kwargs'].get('nearest', False)
        self.pre_act_density = st['MaskCache_kwargs'].get('pre_act_density', False)
        self.in_act_density = st['MaskCache_kwargs'].get('in_act_density', False)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode='nearest')
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        elif self.pre_act_density:
            density = 1 - torch.exp(-F.softplus(self.density + self.act_shift) * self.voxel_size_ratio)
            alpha = F.grid_sample(self.density, ind_norm, align_corners=True)
        elif self.in_act_density:
            density = F.grid_sample(F.softplus(self.density + self.act_shift), ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-density * self.voxel_size_ratio)
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return {'alpha': alpha, 'mask': (alpha >= self.mask_cache_thres)}


''' Misc
'''
def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[...,[0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1-alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


def total_variation(v, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    if mask is not None:
        tv2 = tv2[:,:,(mask[:,:,:-1] & mask[:,:,1:])[0,0,...]]
        tv3 = tv3[:,:,(mask[:,:,:,:-1] & mask[:,:,:,1:])[0,0,...]]
        tv4 = tv4[:,:,(mask[:,:,:,:,:-1] & mask[:,:,:,:,1:])[0,0,...]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, render=False, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    if render == True:
        rays_d_dx = torch.cat(((rays_d[:-1, :, :]+rays_d[1:, :, :])/2, rays_d[-1:, :, :]), axis=0)
        rays_d_dy = torch.cat(((rays_d[:, :-1, :]+rays_d[:, 1:, :])/2, rays_d[:, -1:, :]), axis=1)

        return rays_o, rays_d, viewdirs, rays_d_dx, rays_d_dy
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays_coarse(rgb_tr_ori, lossmult, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_coarse: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    rgb_tr = torch.tensor(np.array(rgb_tr_ori))
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    lossmult_tr = torch.zeros([len(rgb_tr), H, W], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        lossmult_tr[i].copy_(torch.tensor(np.broadcast_to(lossmult[i], (H, W))).to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, None, None, lossmult_tr


@torch.no_grad()
def get_training_rays(rgb_tr_ori, lossmult, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    #import pdb
    #pdb.set_trace()
    print('get_training_rays: start')
    #assert len(np.unique(HW, axis=0)) == 1
    #assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    #H, W = HW[0]
    #K = Ks[0]
    dev = torch.tensor(1)
    DEVICE = dev.device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3]).to(DEVICE)
    rays_o_tr = torch.zeros([N, 3]).to(DEVICE)
    rays_d_tr = torch.zeros([N, 3]).to(DEVICE)
    viewdirs_tr = torch.zeros([N, 3]).to(DEVICE)
    rays_d_dx_tr = torch.zeros([N, 3]).to(DEVICE)
    rays_d_dy_tr = torch.zeros([N, 3]).to(DEVICE)
    lossmult_tr = torch.zeros([N]).to(DEVICE)
    imsz = []
    top = 0

    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=HW[i][0], W=HW[i][1], K=Ks[i], c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = HW[i][0] * HW[i][1]
        rgb_tr[top:top+n].copy_(torch.tensor(rgb_tr_ori[i]).flatten(0,1).to(DEVICE))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        rays_d_dx_tr[top:top+n].copy_(torch.cat(((rays_d[:-1, :, :]+rays_d[1:, :, :])/2, rays_d[-1:, :, :]), axis=0).flatten(0,1).to(DEVICE))
        rays_d_dy_tr[top:top+n].copy_(torch.cat(((rays_d[:, :-1, :]+rays_d[:, 1:, :])/2, rays_d[:, -1:, :]), axis=1).flatten(0,1).to(DEVICE))
        lossmult_tr[top:top+n].copy_(torch.tensor(np.broadcast_to(lossmult[i],rgb_tr_ori[i].shape[:2])).flatten(0,1).to(DEVICE))
        del rays_o, rays_d, viewdirs
        imsz.append(n)
        top += n
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, rays_d_dx_tr,rays_d_dy_tr, lossmult_tr



@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori,lossmult, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    #import pdb
    #pdb.set_trace()
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    # dev = torch.tensor(1).cuda()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_dx_tr = torch.zeros_like(rgb_tr)
    rays_d_dy_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    lossmult_tr = torch.zeros([N], device=DEVICE)
    imsz = []
    top = 0
    for j, (c2w, img, (H, W), K) in enumerate(zip(train_poses, rgb_tr_ori, HW, Ks)):
        img = torch.tensor(img)
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.sample_ray(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox])['mask'])
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        ## (x+1,y)  (x,y+1) -> (0,0)->(1,0), (0,1).
        ## torch.cat([rays_d[1:,:,:],rays_d[None,-1,:,:]],axis=0)
        ## set the last elements as same values, in the edge of image.
        rays_d_dx_tr[top:top+n].copy_(torch.cat(((rays_d[:-1, :, :]+rays_d[1:, :, :])/2, rays_d[-1:, :, :]), axis=0)[mask].to(DEVICE))
        rays_d_dy_tr[top:top+n].copy_(torch.cat(((rays_d[:, :-1, :]+rays_d[:, 1:, :])/2, rays_d[:, -1:, :]), axis=1)[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        lossmult_tr[top:top + n].copy_(
            torch.tensor(np.broadcast_to(lossmult[j], img.shape[:2]))[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    rays_d_plus_x_tr = rays_d_dx_tr[:top]
    rays_d_plus_y_tr = rays_d_dy_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, rays_d_plus_x_tr, rays_d_plus_y_tr, lossmult_tr


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS
