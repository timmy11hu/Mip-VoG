_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_multiscale/materials'

data = dict(
    datadir='../data/nerf_multiscale/materials',
    dataset_type='blender_ms',
    white_bkgd=True,
)

