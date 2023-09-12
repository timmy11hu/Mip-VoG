import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).float()
        @ c2w
    )
    return c2w.cpu()

def pose_spherical_torch(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def append_poses(c2w, t, scales, H, W, focal, render_poses, Ks, HWs):
    h, w, f = H // (2 ** t), W // (2 ** t), focal / (2 ** t)
    pose = np.copy(c2w)
    pose[:3, 3] *= scales[t]
    Ks.append([[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]])
    HWs.append([h, w])
    render_poses.append(pose)

def render_pose_special(num_pose=360, H=800, W=800, focal=800):
    poses_ = torch.stack(
        [pose_spherical(angle, -30.0, 4.0)  # @ cam_trans
         for angle in np.linspace(-180, 180, num_pose + 1)[:-1]], 0,)
    # 0.6-1.5
    # cam_scale_factor = 1.0
    # poses_[:, :3, 3] *= cam_scale_factor
    render_poses, Ks, HWs = [], [], []
    flag_ogres = True
    cutpoints = [90, 180, 270]
    scales = [0.7, 0.9, 1.1, 1.3]
    for i, c2w in enumerate(poses_.numpy()):
        t = 0 if flag_ogres else 3
        append_poses(c2w=np.copy(c2w), t=t, scales=scales, H=H, W=W, focal=focal,
                     render_poses=render_poses, Ks=Ks, HWs=HWs)
        if i in cutpoints:
            trans = (1, 2, 3) if flag_ogres else (2, 1, 0)
            for t in trans:
                append_poses(c2w=np.copy(c2w), t=t, scales=scales, H=H, W=W, focal=focal,
                             render_poses=render_poses, Ks=Ks, HWs=HWs)

    return np.stack(render_poses), np.array(Ks), np.array(HWs)


def flatten(x):
    # Always flatten out the height x width dimensions
    x = [y.reshape([-1, y.shape[-1]]) for y in x]
    # concatenate all data into one list
    x = np.concatenate(x, axis=0)
    return x


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 60+1)[:-1]], 0)
    # 0.6-1.5
    cam_scale_factor = 1.0
    render_poses[:, :3, 3] *= cam_scale_factor

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_ms_data(basedir, half_res=False, test_skip=1, white_bkgd=True):
    cam_scale_factor = 1.0
    cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    splits = ["train", "val", "test"]

    metadatapath = os.path.join(basedir, "metadata.json")
    with open(metadatapath) as fp:
        metadata = json.load(fp)

    images = []
    extrinsics = []
    counts = [0]
    focals = []
    multlosses = []

    for s in splits:
        meta = metadata[s]
        imgs = []
        poses = []
        fs = []
        multloss = []

        if s == "train":
            skip = 1
        elif s == "val":
            skip = 1
        elif s == "test":
            skip = test_skip

        for (filepath, pose, focal, mult) in zip(
            meta["file_path"][::skip],
            meta["cam2world"][::skip],
            meta["focal"][::skip],
            meta["lossmult"][::skip],
        ):
            fname = os.path.join(basedir, filepath)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(pose))
            fs.append(focal)
            multloss.append(mult)

        imgs = [(img / 255.0).astype(np.float32) for img in imgs]
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + len(imgs))
        images += imgs
        focals += fs
        extrinsics.append(poses)
        multlosses.append(np.array(multloss))

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    extrinsics = np.concatenate(extrinsics, 0)

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics #@ cam_trans

    image_sizes = np.array([img.shape[:2] for img in images])
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for (focal, (h, w)) in zip(focals, image_sizes)
        ]
    )

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0) #@ cam_trans
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
    render_poses[:, :3, 3] *= cam_scale_factor
    render_Ks, render_HWs = None, None

    render_poses, render_Ks, render_HWs = render_pose_special(num_pose=360, H=image_sizes[0][0], W=image_sizes[0][1], focal=focals[0])

    near = 2.0
    far = 6.0

    if white_bkgd:
        images = [
            image[..., :3] * image[..., -1:] + (1.0 - image[..., -1:])
            for image in images
        ]
    else:
        images = [image[..., :3] for image in images]

    multlosses = np.concatenate(multlosses)

    return (
        images,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        (-1, -1),
        i_split,
        render_poses,
        multlosses,  # Train only
    )
