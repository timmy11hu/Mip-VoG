_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_synthetic/ficus'

data = dict(
    datadir='../data/nerf_multiscale/ficus',
    dataset_type='blender_ms',
    white_bkgd=True,
)

