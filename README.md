## Multiscale Representation for Real-Time Anti-Aliasing Neural Rendering

This repository contains the implementation of Mip-VoG described in
[D. Hu, Z. Zhang, T. Hou, T. Liu, H. Fu^1^, M. Gong^2^: Multiscale Representation for Real-Time Anti-Aliasing Neural Rendering. International Conference on Computer Vision (ICCV) 2023. ]([https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620229.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Multiscale_Representation_for_Real-Time_Anti-Aliasing_Neural_Rendering_ICCV_2023_paper.pdf))


### Training and evaluation with different configuration

To train a single scene with mip-vog, run 
```python run.py --config configs/nerf_ms/"$scene".py --resolution 512 --mip_train```

Test the model with mip-vog, run 
```python run.py --config configs/nerf_ms/"$scene".py --resolution 512 --mip_train --render_only --render_test --mip_test```

Test the model with voxel grids but without mipmapping, run 
```python run.py --config configs/nerf_ms/"$scene".py --resolution 512 --mip_train --render_only --render_test```


### Citation
If you find it useful, please consider citing:
```
@InProceedings{Hu_2023_ICCV,
    author    = {Hu, Dongting and Zhang, Zhenkai and Hou, Tingbo and Liu, Tongliang and Fu, Huan and Gong, Mingming},
    title     = {Multiscale Representation for Real-Time Anti-Aliasing Neural Rendering},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {17772-17783}
}
```

### Acknowledgement
The training code base is origined from [DVGO](https://github.com/sunset1995/DirectVoxGO), and web viewer base is origined from [SNeRG](https://github.com/google-research/google-research/tree/master/snerg).
