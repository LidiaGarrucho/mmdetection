# The new config inherits a base config to highlight the necessary modification
_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py'
model = dict(
    type='PointRend',
    roi_head=dict(
        mask_head=dict(
            _delete_=True,
            type='CoarseMaskHead',
            num_classes=1),
        point_head=dict(
            type='MaskPointHead',
            num_classes=1)))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 2
LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
RESIZE_PARAM = 2 # 1
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(
        type='Resize',
        img_scale=[(LONGER_EDGE, 640*RESIZE_PARAM), (LONGER_EDGE, 672*RESIZE_PARAM), (LONGER_EDGE, 704*RESIZE_PARAM), (LONGER_EDGE, 736*RESIZE_PARAM),
                   (LONGER_EDGE, 768*RESIZE_PARAM), (LONGER_EDGE, SHORTER_EDGE)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
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
ddataset_type = 'COCODataset'
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

load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
# URL http://download.openmmlab.com/mmdetection/v2.0/

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)

#Train 
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/pointred.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/pointred --seed 999
                                                        