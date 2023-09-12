# mip-vog

To train a single scene with mip-vog, run ```python run.py --config configs/nerf_ms/"$scene".py --resolution 512 --mip_train```
Test the model with mip-vog, run ```python run.py --config configs/nerf_ms/"$scene".py --resolution 512 --mip_train --render_only --render_test --mip_test```
Test the model without mip-vog, run ```python run.py --config configs/nerf_ms/"$scene".py --resolution 512 --mip_train --render_only --render_test```
