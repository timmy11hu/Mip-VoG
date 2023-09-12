_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_multiscale/lego'

data = dict(
    datadir='../data/nerf_multiscale/lego',
    dataset_type='blender_ms',
    white_bkgd=True,
)

