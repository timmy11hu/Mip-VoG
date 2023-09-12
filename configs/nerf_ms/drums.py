_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_multiscale/drums'

data = dict(
    datadir='../data/nerf_multiscale/drums',
    dataset_type='blender_ms',
    white_bkgd=True,
)

