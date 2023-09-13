# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions that export a baked SNeRG model for viewing in the web-viewer."""
import os
import json
import math
import multiprocessing
import numpy as np
from PIL import Image
import shutil

from . import utils


def save_8bit_png(img_and_path):
  """Save an 8bit numpy array as a PNG on disk.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit,
      [height, width, channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path
  with utils.open_file(pth, 'wb') as imgout:
    Image.fromarray(img).save(imgout, 'PNG')

def export_snerg_scene(scale, output_directory, atlas, atlas_block_indices,
                       atlas_mipmap_t_list,
                       viewdir_mlp_params, render_params, atlas_params,
                       scene_params, input_height, input_width, input_focal):
  """Exports a scene to web-viewer format: a collection of PNGs and a JSON file.

  The scene gets exported to output_directory/png. Any previous results will
  be overwritten.

  Args:
    output_directory: The root directory where the scene gets written.
    atlas: The SNeRG scene packed as a texture atlas in a [S, S, N, C] numpy
      array, where the channels C contain both RGB and features.
    atlas_block_indices: The indirection grid of the SNeRG scene, represented as
      a numpy int32 array of size (bW, bH, bD, 3).
    viewdir_mlp_params: A dict containing the MLP parameters for the per-sample
      view-dependence MLP.
    render_params: A dict with parameters for high-res rendering.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    input_height: Height (pixels) of the NDC camera (i.e. the training cameras).
    input_width: Width (pixels) of the NDC camera (i.e. the training cameras).
    input_focal: Focal length (pixels) of the NDC camera (i.e. the
      training cameras).
  """
  # Slice the atlas into images.
  rgbs, alphas, levels, numbers = [], [], [], []
  count = 0
  for i in range(0, atlas.shape[2], 4):
    rgb_stack = []
    alpha_stack = []
    for j in range(4):
      plane_index = i + j
      rgb_stack.append(atlas[:, :, plane_index, :][Ellipsis,
                                                   0:3].transpose([1, 0, 2]))
      alpha_stack.append(
          atlas[:, :,
                plane_index, :][Ellipsis,
                                scene_params['_channels']].transpose([1, 0]))
    rgbs.append(np.concatenate(rgb_stack, axis=0))
    alphas.append(np.concatenate(alpha_stack, axis=0))
    levels.append(0)
    numbers.append(count)
    count += 1
  num_slices = count

  if len(atlas_mipmap_t_list) > 0:
    for l, map in enumerate(atlas_mipmap_t_list):
      count = 0
      for i in range(0, map.shape[2], 4):
        rgb_stack = []
        alpha_stack = []
        for j in range(4):
          plane_index = i + j
          rgb_stack.append(map[:, :, plane_index, :][Ellipsis, 0:3].transpose([1, 0, 2]))
          alpha_stack.append(
              map[:, :, plane_index, :][Ellipsis, scene_params['_channels']].transpose([1, 0])
          )
        rgbs.append(np.concatenate(rgb_stack, axis=0))
        alphas.append(np.concatenate(alpha_stack, axis=0))
        levels.append(l+1)
        numbers.append(count)
        count += 1

  atlas_index_image = np.transpose(atlas_block_indices, [2, 1, 0, 3]).reshape(
      (-1, atlas_block_indices.shape[0], 3)).astype(np.uint8)

  # Build a dictionary of the scene parameters, so we can export it as a json.
  export_scene_params = {}
  export_scene_params['voxel_size'] = float(render_params['_voxel_size'])
  export_scene_params['block_size'] = atlas_params['_data_block_size']
  export_scene_params['grid_width'] = int(render_params['_grid_size'][0])
  export_scene_params['grid_height'] = int(render_params['_grid_size'][1])
  export_scene_params['grid_depth'] = int(render_params['_grid_size'][2])
  export_scene_params['atlas_width'] = atlas.shape[0]
  export_scene_params['atlas_height'] = atlas.shape[1]
  export_scene_params['atlas_depth'] = atlas.shape[2]
  export_scene_params['num_slices'] = num_slices

  export_scene_params['min_x'] = float(scene_params['min_xyz'][0])
  export_scene_params['min_y'] = float(scene_params['min_xyz'][1])
  export_scene_params['min_z'] = float(scene_params['min_xyz'][2])

  export_scene_params['atlas_blocks_x'] = int(atlas.shape[0] /
                                              atlas_params['atlas_block_size'])
  export_scene_params['atlas_blocks_y'] = int(atlas.shape[1] /
                                              atlas_params['atlas_block_size'])
  export_scene_params['atlas_blocks_z'] = int(atlas.shape[2] /
                                              atlas_params['atlas_block_size'])

  export_scene_params['input_height'] = float(input_height)
  export_scene_params['input_width'] = float(input_width)
  export_scene_params['input_focal'] = float(input_focal)

  export_scene_params['worldspace_T_opengl'] = scene_params[
      'worldspace_T_opengl'].tolist()
  export_scene_params['ndc'] = scene_params['ndc']

  if export_scene_params.get('interval') is not None:
    export_scene_params['act_shift'] = float(render_params['act_shift'])
    export_scene_params['interval'] = float(render_params['interval'])
    export_scene_params['density_location'] = float(atlas_params['density_location'])
    export_scene_params['density_scale'] = float(atlas_params['density_scale'])
    export_scene_params['color_location'] = float(atlas_params['color_location'])
    export_scene_params['color_scale'] = float(atlas_params['color_scale'])

  # Also include the network weights in this dictionary.
  export_scene_params['0_weights'] = viewdir_mlp_params['params']['Dense_0'][
      'kernel'].tolist()
  export_scene_params['1_weights'] = viewdir_mlp_params['params']['Dense_1'][
      'kernel'].tolist()
  export_scene_params['2_weights'] = viewdir_mlp_params['params']['Dense_3'][
      'kernel'].tolist()
  export_scene_params['0_bias'] = viewdir_mlp_params['params']['Dense_0'][
      'bias'].tolist()
  export_scene_params['1_bias'] = viewdir_mlp_params['params']['Dense_1'][
      'bias'].tolist()
  export_scene_params['2_bias'] = viewdir_mlp_params['params']['Dense_3'][
      'bias'].tolist()

  # To avoid partial overwrites, first dump the scene to a temporary directory.
  output_tmp_directory = output_directory + '/temp'

  if utils.isdir(output_tmp_directory):
    shutil.rmtree(output_tmp_directory)
  utils.makedirs(output_tmp_directory)

  # Now store the indirection grid.
  atlas_indices_path = '%s/atlas_indices.png' % output_tmp_directory

  save_8bit_png((atlas_index_image, atlas_indices_path))

  # Make sure that all JAX hosts have reached this point in the code before we
  # proceed. Things will get tricky if output_tmp_directory doesn't yet exist.
  # synchronize_jax_hosts()

  # Save the alpha values and RGB colors as one set of PNG images.
  output_images = []
  output_paths = []
  for i, rgb_and_alpha in enumerate(zip(rgbs, alphas)):
    rgb, alpha = rgb_and_alpha
    rgba = np.concatenate([rgb, np.expand_dims(alpha, -1)], axis=-1)
    uint_multiplier = 2.0**8 - 1.0
    rgba = np.minimum(uint_multiplier,
                      np.maximum(0.0, np.floor(uint_multiplier * rgba))).astype(
                          np.uint8)
    output_images.append(rgba)
    atlas_rgba_path = '%s/rgba_%01d%02d.png' % (output_tmp_directory, levels[i], numbers[i])
    output_paths.append(atlas_rgba_path)

  # Save the computed features a separate collection of PNGs.
  count = 0
  uint_multiplier = 2.0**8 - 1.0
  for i in range(0, atlas.shape[2], 4):
    feature_stack = []
    for j in range(4):
      plane_index = i + j
      feature_slice = atlas[:, :, plane_index, :][Ellipsis,
                                                  3:-1].transpose([1, 0, 2])
      feature_slice = np.minimum(
          uint_multiplier,
          np.maximum(0.0, np.floor(uint_multiplier * feature_slice))).astype(
              np.uint8)
      feature_stack.append(feature_slice)

    output_images.append(np.concatenate(feature_stack, axis=0))
    output_paths.append('%s/feature_%01d%02d.png' % (output_tmp_directory, 0, count))
    count += 1

  if len(atlas_mipmap_t_list) > 0:
    for l, map in enumerate(atlas_mipmap_t_list):
      count = 0
      for i in range(0, map.shape[2], 4):
        feature_stack = []
        for j in range(4):
          plane_index = i + j
          feature_slice = map[:, :, plane_index, :][Ellipsis,
                          3:-1].transpose([1, 0, 2])
          feature_slice = np.minimum(
            uint_multiplier,
            np.maximum(0.0, np.floor(uint_multiplier * feature_slice))).astype(
            np.uint8)
          feature_stack.append(feature_slice)

        output_images.append(np.concatenate(feature_stack, axis=0))
        output_paths.append('%s/feature_%01d%02d.png' % (output_tmp_directory, l+1, count))
        count += 1

  # for i in range(len(rgbs)):
  #   output_paths.append('%s/feature_%03d.png' % (output_tmp_directory, i))

  print("Start store png:", len(list(zip(output_images, output_paths))))
  for img_and_path in list(zip(output_images, output_paths)):
    save_8bit_png(img_and_path)
  # parallel_write_images(save_8bit_png, list(zip(output_images, output_paths)))
  print("Finish store png")

  # Now export the scene parameters and the network weights as a JSON.
  export_scene_params['format'] = 'png'
  scene_params_path = '%s/scene_params.json' % output_tmp_directory

  with utils.open_file(scene_params_path, 'wb') as f:
    f.write(json.dumps(export_scene_params).encode('utf-8'))

  # Again, make sure that the JAX hosts are in sync. Don't delete
  # output_tmp_directory before all files have been written.
  # synchronize_jax_hosts()

  # Finally move the scene to the appropriate output path.
  output_png_directory = output_directory + '/png' + str(scale)

  # Delete the folder if it already exists.
  if utils.isdir(output_png_directory):
    shutil.rmtree(output_png_directory)
  os.rename(output_tmp_directory, output_png_directory)


def compute_scene_size(output_directory, atlas_block_indices, atlas_params,
                       scene_params):
  """Computes the size of an exported SNeRG scene.

  Args:
    output_directory: The root directory where the SNeRG scene was written.
    atlas_block_indices: The indirection grid of the SNeRG scene.
    atlas_params: A dict with params for building the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    png_size_gb: The scene size (in GB) when stored as compressed 8-bit PNGs.
    byte_size_gb: The scene size (in GB), stored as uncompressed 8-bit integers.
    float_size_gb: The scene size (in GB), stored as uncompressed 32-bit floats.
  """

  output_png_directory = output_directory + '/png'
  png_files = [
      output_png_directory + '/' + f
      for f in sorted(utils.listdir(output_png_directory))
      if f.endswith('png')
  ]
  png_size_gb = sum(
      [tf.io.gfile.stat(f).length / (1000 * 1000 * 1000) for f in png_files])

  block_index_size_gb = np.array(
      atlas_block_indices.shape).prod() / (1000 * 1000 * 1000)

  active_atlas_blocks = (atlas_block_indices[Ellipsis, 0] >= 0).sum()
  active_atlas_voxels = (
      active_atlas_blocks * atlas_params['atlas_block_size']**3)
  active_atlas_channels = active_atlas_voxels * scene_params['_channels']

  byte_size_gb = active_atlas_channels / (1000 * 1000 *
                                          1000) + block_index_size_gb
  float_size_gb = active_atlas_channels * 4 / (1000 * 1000 *
                                               1000) + block_index_size_gb

  return png_size_gb, byte_size_gb, float_size_gb
