_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_multiscale/chair'

data = dict(
    datadir='../data/nerf_multiscale/chair',
    dataset_type='blender_ms',
    white_bkgd=True,
)