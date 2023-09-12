import numpy as np

from .load_llff import load_llff_data
from .load_blender import load_blender_data, load_blender_ms_data
from .load_tankstemple import load_tankstemple_data



def load_data(args):

    K, depths = None, None
    render_Ks, render_HWs = None, None

    if args.dataset_type == 'blender_ms':
        images, intrinsics, extrinsics, image_sizes, near, far, _, i_split,  render_poses, lossmult = load_blender_ms_data(args.datadir, args.half_res, args.testskip)
        #print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        # import pdb
        # pdb.set_trace()
        i_train, i_val, i_test, _ = i_split
        # train_images, train_poses, _, _, i_split = load_blender_data("./data/nerf_synthetic/mic", args.half_res, args.testskip)
        K = intrinsics
        poses = extrinsics
        HW = image_sizes
        hwf = None
        Ks = K

        # if images.shape[-1] == 4:
        #    if args.white_bkgd:
        #        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        #    else:
        #        images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'blender_ss':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]

        lossmult = np.asarray([1.0]*len(images))

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = np.array(H).astype(int), np.array(W).astype(int)
        hwf = [H, W, focal]
        HW = np.array([im.shape[:2] for im in images])
        irregular_shape = (images.dtype is np.dtype('object'))

        if K is None:
            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])

        if len(K.shape) == 2:
            Ks = K[None].repeat(len(poses), axis=0)
        else:
            Ks = K

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(args.datadir)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

        lossmult = np.asarray([1.0] * len(images))

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = np.array(H).astype(int), np.array(W).astype(int)
        hwf = [H, W, focal]
        HW = np.array([im.shape[:2] for im in images])
        irregular_shape = (images.dtype is np.dtype('object'))

        if K is None:
            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])

        if len(K.shape) == 2:
            Ks = K[None].repeat(len(poses), axis=0)
        else:
            Ks = K

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    render_poses = render_poses[..., :4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses, render_Ks=render_Ks, render_HWs=render_HWs,
        images=images, depths=depths,
        irregular_shape=None, lossmult=lossmult,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

