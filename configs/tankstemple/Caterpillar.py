_base_ = '../default2.py'

scene = 'Caterpillar'
expname = 'dvgo'
basedir = './logs/tanks_and_temple/'+scene

data = dict(
    datadir='../data/TanksAndTemple/'+scene,
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
)

coarse_train = dict(
    pervoxel_lr_downrate=2,
)

