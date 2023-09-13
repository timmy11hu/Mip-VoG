import gc
import os
from os import path
import random
from absl import app
from absl import flags
import torch
import numpy as np
from scipy import ndimage

from lib import dvgo
from lib import utils as dvgo_utils
from lib.load_data import load_data

from snerg import utils

from snerg import baking
from snerg import culling
from snerg import export
from snerg import params


FLAGS = flags.FLAGS
utils.define_flags()


def extract_mlp_params(model):
    model_state_dict = model.state_dict()
    mlp_0_weight = np.transpose(model_state_dict['rgbnet.0.weight'].cpu().numpy())
    mlp_0_bias = np.transpose(model_state_dict['rgbnet.0.bias'].cpu().numpy())
    mlp_1_weight = np.transpose(model_state_dict['rgbnet.2.0.weight'].cpu().numpy())
    mlp_1_bias = np.transpose(model_state_dict['rgbnet.2.0.bias'].cpu().numpy())
    mlp_2_weight = np.transpose(model_state_dict['rgbnet.3.weight'].cpu().numpy())
    mlp_2_bias = np.transpose(model_state_dict['rgbnet.3.bias'].cpu().numpy())
    assert mlp_0_weight.shape == (34, 16) and mlp_0_bias.shape == (16,)
    assert mlp_1_weight.shape == (16, 16) and mlp_1_bias.shape == (16,)
    assert mlp_2_weight.shape == (16, 3) and mlp_2_bias.shape == (3,)
    mlp_params = {'Dense_0': {'kernel': mlp_0_weight, 'bias': mlp_0_bias},
                  'Dense_1': {'kernel': mlp_1_weight, 'bias': mlp_1_bias},
                  'Dense_3': {'kernel': mlp_2_weight, 'bias': mlp_2_bias}, }
    return mlp_params


def process_atlas(atlas):
    atlas_t_list = []
    for i in range(atlas.shape[2]):
        atlas_t_list.append(torch.from_numpy(atlas[:, :, i, :]))
    atlas_t = torch.stack(atlas_t_list, 2)
    uint_multiplier = 2.0 ** 8 - 1.0
    atlas_t *= uint_multiplier
    gc.collect()
    atlas_t = torch.floor(atlas_t)
    gc.collect()
    atlas_t = torch.clip(atlas_t, min=0.0, max=uint_multiplier)
    gc.collect()
    atlas_t /= uint_multiplier
    gc.collect()
    return atlas_t


def main(unused_argv):
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)

    mip_size = 3
    scale = FLAGS.voxel_resolution
    path = FLAGS.log_path

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device: ", device)

    import mmcv
    cfg = mmcv.Config.fromfile(os.path.join(path, 'config.py'))
    data_dict = load_data(cfg.data)
    model = dvgo_utils.load_model(dvgo.DirectVoxGO, os.path.join(path, 'fine_last.tar')).to(device)

    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
      'model': model,
      'ndc': cfg.data.ndc,
      'render_kwargs': {
          'near': data_dict['near'], 'far': data_dict['far'], 'bg': 1 if cfg.data.white_bkgd else 0,
          'stepsize': stepsize, 'inverse_y': cfg.data.inverse_y, 'flip_x': cfg.data.flip_x, 'flip_y': cfg.data.flip_y,
      },
    }
    render_kwargs = render_viewpoints_kwargs['render_kwargs']

    viewdir_mlp_params = {'params': extract_mlp_params(model)}

    out_dir = os.path.join(path, 'baked')
    os.makedirs(out_dir, exist_ok=True)

    gc.collect()

    (render_params_init, culling_params_init, atlas_params_init, scene_params_init) = params.initialize_params(FLAGS)

    # Render out the low-res grid used for culling.
    culling_grid_coordinates = baking.build_3d_grid(
        scene_params_init["min_xyz"], culling_params_init["_voxel_size"],
        culling_params_init["_grid_size"],
        scene_params_init["worldspace_T_opengl"],
        np.dtype(scene_params_init["dtype"]))
    _, culling_grid_alpha = baking.render_voxel_block(
        model, 0, device, render_kwargs, culling_grid_coordinates,
        culling_params_init["_voxel_size"], scene_params_init)

    print(culling_grid_alpha.shape, culling_grid_alpha.min(), culling_grid_alpha.max())
    # Using this grid, maximize resolution with a tight crop on the scene.
    (render_params, culling_params, atlas_params,
     scene_params) = culling.crop_alpha_grid(render_params_init,
                                             culling_params_init,
                                             atlas_params_init,
                                             scene_params_init,
                                             culling_grid_alpha)
    print("grid size: ", atlas_params["_atlas_grid_size"])

    # Recompute the low-res grid using the cropped scene bounds.
    culling_grid_coordinates = baking.build_3d_grid(
        scene_params["min_xyz"], culling_params["_voxel_size"],
        culling_params["_grid_size"], scene_params["worldspace_T_opengl"],
        np.dtype(scene_params["dtype"]))
    _, culling_grid_alpha = baking.render_voxel_block(
        model, 0, device, render_kwargs, culling_grid_coordinates,
        culling_params["_voxel_size"], scene_params)

    num_training_cameras = len(data_dict['i_train'])
    culling_grid_visibility = np.zeros_like(culling_grid_alpha)
    h, w, focal = data_dict['hwf']
    c2ws = [data_dict['poses'][i] for i in data_dict['i_train']]
    assert len(c2ws) == num_training_cameras

    for camera_index in range(0, num_training_cameras,
                              culling_params["visibility_subsample_factor"]):
        culling.integrate_visibility_from_image(
          h * culling_params["visibility_image_factor"],
          w * culling_params["visibility_image_factor"],
          focal * culling_params["visibility_image_factor"],
          c2ws[camera_index], culling_grid_alpha,
          culling_grid_visibility, scene_params, culling_params)

    # Finally, using this updated low-res grid, compute the maximum alpha within each macroblock.
    atlas_grid_alpha = culling.max_downsample_grid(culling_params, atlas_params,
                                                   culling_grid_alpha)
    atlas_grid_visibility = culling.max_downsample_grid(
        culling_params, atlas_params, culling_grid_visibility)

    # Make the visibility grid more conservative by dilating it. We need to
    # temporarly cast to float32 here, as ndimage.maximum_filter doesn't work
    # with float16.
    atlas_grid_visibility = ndimage.maximum_filter(
        atlas_grid_visibility.astype(np.float32),
        culling_params["visibility_grid_dilation"]).astype(
            atlas_grid_visibility.dtype)
    print("Finish max downsample. Start to bake...")
    # Now we're ready to extract the scene and pack it into a 3D texture atlas.
    atlas, atlas_block_indices, atlas_mipmap = baking.extract_3d_atlas(
        model, mip_size, device, render_kwargs, scene_params, render_params, atlas_params,
        culling_params, atlas_grid_alpha, atlas_grid_visibility) # return[2048, 2048, 32, 8], [23,34,24]

    # Free up CPU memory wherever we can to avoid OOM in the larger scenes.
    del atlas_grid_alpha
    del atlas_grid_visibility
    del culling_grid_alpha
    del culling_grid_visibility
    gc.collect()

    # Convert the atlas to a tensor, so we can use can use tensorflow's massive
    # CPU parallelism for ray marching.
    atlas_block_indices_t = torch.from_numpy(atlas_block_indices)
    del atlas_block_indices
    gc.collect()

    atlas_t = process_atlas(atlas)

    atlas_mipmap_t_list = None
    if atlas_mipmap is not None:
        atlas_mipmap_t_list = []
        for map in atlas_mipmap:
            map_t = process_atlas(map)
            atlas_mipmap_t_list.append(map_t.numpy())

    # Export the baked scene so we can view it in the web-viewer.
    refined_viewdir_mlp_params = viewdir_mlp_params
    export.export_snerg_scene(scale, out_dir, atlas_t.numpy(),
                              atlas_block_indices_t.numpy(), atlas_mipmap_t_list,
                              refined_viewdir_mlp_params, render_params,
                              atlas_params, scene_params, h, w, focal)
    gc.collect()


if __name__ == "__main__":
  app.run(main)
