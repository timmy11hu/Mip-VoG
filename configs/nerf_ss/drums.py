_base_ = '../default.py'

scene = 'drums'
expname = 'dvgo'
basedir = './logs/nerf_synthetic/'+scene

data = dict(
    datadir='../data/nerf_synthetic/'+scene,
    dataset_type='blender_ss',
    white_bkgd=True,
)


