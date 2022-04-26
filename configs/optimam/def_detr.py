#import torch
#torch.backends.cudnn.benchmark = True # Improves overall speed
#torch.backends.cudnn.enabled = False
# The new config inherits a base config to highlight the necessary modification

# Deformable Transformers
#_base_ = '../deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
_base_ = '../deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py'
# _base_ = '../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'
# DETR
#_base_ = '../detr/detr_r50_8x2_150e_coco.py'
model = dict(
    bbox_head=dict(num_classes=1))

LANDMARKS = '/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/optimam_hologic_data_aug_mass_train_set.pth'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS, to_rgb=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
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
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
    workers_per_gpu=1,
    train=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_train.json',
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_train.json',
        filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json',
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_val.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json',
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_val.json',
        pipeline=test_pipeline))

# # evaluate the model every 5 epoch.
# evaluation = dict(interval=5)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# Deformable Transformers
# load_from = 'checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
load_from = 'checkpoints/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth'
# load_from = 'checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
# DETR
# load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-5, #2e-4
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=40)

# python mmdet/utils/collect_env.py
# Check install: python -c 'import mmcv; import mmcv.ops'

# URL http://download.openmmlab.com/mmdetection/v2.0/
# CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr.py --work-dir /home/lidia/source/mmdetection/experiments/high_density/cyclegan/high_density_h800/no_data_aug --seed 999 --deterministic
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr.py --work-dir /home/lidia/source/mmdetection/experiments/high_density/cyclegan/high_density_h800/no_data_aug_seed42 --seed 42 --deterministic
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr.py --work-dir /home/lidia/source/mmdetection/experiments/high_density/cyclegan/high_density_h800/no_data_aug_seed999 --seed 999 --deterministic
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr.py --work-dir /home/lidia/source/mmdetection/experiments/high_density/cyclegan/high_density_h800/data_aug_acr12_seed999 --seed 999 --deterministic
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/train.py ./configs/optimam/def_detr.py --work-dir /test/experiments/high_density/cyclegan/high_density_h800/HOLII --seed 999 --deterministic
# PYTHONPATH=${PYTHONPATH}:./ python tools/train.py ./configs/optimam/def_detr.py --work-dir /test/experiments/test/gpu0 --seed 999 --deterministic