# The new config inherits a base config to highlight the necessary modification
# Auto-Assign
#from configs.optimam.def_detr import RESIZE_RATIO

_base_ = '../paa/paa_r101_fpn_mstrain_3x_coco.py'
model = dict(
    bbox_head=dict(
        type='PAAHead',
        num_classes=1,
        ))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 2
LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(
        type='Resize',
        img_scale=[(LONGER_EDGE, 640*RESIZE_PARAM), (LONGER_EDGE, SHORTER_EDGE)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(LONGER_EDGE, SHORTER_EDGE),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('mass',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        img_prefix='',
        classes=classes,
        ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_train.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_val.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_test.json'))

load_from = 'checkpoints/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth'
# URL http://download.openmmlab.com/mmdetection/v2.0/

# optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)


#Train 
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/paa.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/paa/hstd --seed 999
                                                        