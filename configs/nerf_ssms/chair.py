_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_synthetic/chair'

data = dict(
    datadir='../data/nerf_multiscale/chair',
    dataset_type='blender_ms',
    white_bkgd=True,
)