# The new config inherits a base config to highlight the necessary modification
# DETR
_base_ = '../detr/detr_r50_8x2_150e_coco.py'

LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
model = dict(
    bbox_head=dict(num_classes=1))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 2
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480*RESIZE_PARAM, LONGER_EDGE), (512*RESIZE_PARAM, LONGER_EDGE), (544*RESIZE_PARAM, LONGER_EDGE),
                               (576*RESIZE_PARAM, LONGER_EDGE), (608*RESIZE_PARAM, LONGER_EDGE), (640*RESIZE_PARAM, LONGER_EDGE),
                               (672*RESIZE_PARAM, LONGER_EDGE), (704*RESIZE_PARAM, LONGER_EDGE), (736*RESIZE_PARAM, LONGER_EDGE),
                               (768*RESIZE_PARAM, LONGER_EDGE), (SHORTER_EDGE, LONGER_EDGE)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400*RESIZE_PARAM, 4200), (500*RESIZE_PARAM, 4200), (600*RESIZE_PARAM, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384*RESIZE_PARAM, 600*RESIZE_PARAM),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480*RESIZE_PARAM, LONGER_EDGE), (512*RESIZE_PARAM, LONGER_EDGE), (544*RESIZE_PARAM, LONGER_EDGE),
                               (576*RESIZE_PARAM, LONGER_EDGE), (608*RESIZE_PARAM, LONGER_EDGE), (640*RESIZE_PARAM, LONGER_EDGE),
                               (672*RESIZE_PARAM, LONGER_EDGE), (704*RESIZE_PARAM, LONGER_EDGE), (736*RESIZE_PARAM, LONGER_EDGE),
                               (768*RESIZE_PARAM, LONGER_EDGE), (SHORTER_EDGE, LONGER_EDGE)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
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
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
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

# # evaluate the model every 5 epoch.
# evaluation = dict(interval=5)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# DETR
load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-05, #lr=2e-4, 2 images x 8 GPU: 2 images x 1 GPU
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

# URL http://download.openmmlab.com/mmdetection/v2.0/

#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/detr.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/detr --seed 999
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_transforms.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/def_detr/refine_nyul --seed 999