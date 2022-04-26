
_base_ = '../queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
#model = dict(bbox_head=dict(num_classes=1))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 2
LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in min_values],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

load_from = 'checkpoints/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'
#Train
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/queryinst.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/queryinst/no_hstd --seed 999
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/queryinst.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/queryinst/hstd --seed 999
                                                      