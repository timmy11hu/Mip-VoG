_base_ = '../default.py'

scene = 'ficus'
expname = 'dvgo'
basedir = './logs/nerf_synthetic/'+scene

data = dict(
    datadir='../data/nerf_synthetic/'+scene,
    dataset_type='blender_ss',
    white_bkgd=True,
)

