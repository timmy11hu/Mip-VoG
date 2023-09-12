_base_ = '../default.py'

expname = 'dvgo'
basedir = './logs/nerf_synthetic/mic'

data = dict(
    datadir='../data/nerf_multiscale/mic',
    dataset_type='blender_ms',
    white_bkgd=True,
)

