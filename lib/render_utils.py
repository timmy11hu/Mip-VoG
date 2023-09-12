import time
import numpy as np
import torch
import torch.nn.functional as F

from . import dvgo
from . import utils



def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    factor = 1
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        #if H == 800 and W == 800:
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H=H, W=W, K=K, c2w=c2w,
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min*factor)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max*factor)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min*factor, xyz_max*factor


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


class NeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, coarse_weight_decay=0.1):
        super(NeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay

    def forward(self, input, target, mask):
        # mse = (mask.unsqueeze(-1) * ((input - target[..., :3]) ** 2)).sum() / mask.sum()
        mse = (mask.unsqueeze(-1) * F.mse_loss(input, target, reduction="none")).sum() / mask.sum()
        loss = mse
        with torch.no_grad():
            psnrs = mse_to_psnr(mse)
        return loss, torch.Tensor(psnrs)


def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)
