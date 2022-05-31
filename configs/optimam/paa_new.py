# --deterministic
# import torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# The new config inherits a base config to highlight the necessary modification
_base_ = '../paa/paa_r101_fpn_mstrain_3x_coco.py'
model = dict(
    bbox_head=dict(
        type='PAAHead',
        num_classes=1,
        ))
# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('mass',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_train.json'),
    val=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'),
    test=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'))

load_from = 'checkpoints/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth'
# URL: https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html

#Train 
# PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/paa.py --work-dir /home/lidia/source/mmdetection/experiments/high_density/cyclegan/high_density_h800/no_data_aug/paa --seed 999 --deterministic
                                                        