_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_multiscale/ship'

data = dict(
    datadir='../data/nerf_multiscale/ship',
    dataset_type='blender_ms',
    white_bkgd=True,
)

