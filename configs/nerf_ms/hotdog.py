_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_multiscale/hotdog'

data = dict(
    datadir='../data/nerf_multiscale/hotdog',
    dataset_type='blender_ms',
    white_bkgd=True,
)

