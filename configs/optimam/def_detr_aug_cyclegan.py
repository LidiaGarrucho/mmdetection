# The new config inherits a base config to highlight the necessary modification
# Deformable Transformers
#_base_ = '../deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
_base_ = '../deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py'
# _base_ = '../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'
#Albumentations code example
#configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py
#https://albumentations.ai/docs/api_reference/augmentations/transforms/
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Cutout',
                num_holes=25000,
                max_h_size=2,
                max_w_size=2,
                always_apply=True,
                p=1),
            dict(
                type='Cutout',
                num_holes=100000,
                max_h_size=1,
                max_w_size=1,
                always_apply=True,
                p=1)
        ], p=0.5) #test2 10% of total pixels
]
model = dict(
    backbone=dict(
        type='ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        #init_cfg=None,
        #neck=dict(freeze=True),
        bbox_head=dict(num_classes=1, with_box_refine=True, freeze=None))

LONGER_EDGE = 1333
SHORTER_EDGE = 800
RESIZE_PARAM = 1
#LANDMARKS = '/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/optimam_hologic_data_aug_mass_train_set.pth'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'), #color_type='grayscale'), #'color', 'grayscale', 'unchanged'
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(LONGER_EDGE, SHORTER_EDGE), keep_ratio=True),
    # dict(type='RandConvAug', kernel_size=(1,3,5), mixing=True, identity_prob=0.5, mixing_alpha=0.5,
    #         img_scale=(LONGER_EDGE, SHORTER_EDGE), img_std=[58.395, 57.12, 57.375], img_mean=[123.675, 116.28, 103.53],
    #         in_channels=1, to_rgb=False), #to_rgb=True means no Hstd
    #dict(type='Low2HighBreastDensityAug', checkpoint_name='high_density_h800', img_scale=(LONGER_EDGE, SHORTER_EDGE)),
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
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
                    keep_ratio=True,
                    override=False) # if resized before this call set to True
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True,
                    override=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
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
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_masks': 'masks',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
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
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
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
    samples_per_gpu=2, #2
    workers_per_gpu=2, #2
    train=dict(
        pipeline=train_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_csaw_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_csaw_bcdr_optimam_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_optimam_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_optimam_bcdr_train.json'),
        
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_bcdr_optimam_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_optimam_train.json'),
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_optimam_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_optimam_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_high_to_low/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_high_to_low_train.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_csaw_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_csaw_bcdr_optimam_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_optimam_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_optimam_bcdr_val.json'),   
        
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_bcdr_optimam_val.json'), 
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_optimam_val.json'),
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_optimam_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_optimam_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_high_to_low/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_high_to_low_val.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_csaw_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_csaw_bcdr_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_acr12_optimam_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_acr12_optimam_bcdr_val.json'))

        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_bcdr_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_optimam_val.json'))
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_high_to_low/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_high_to_low_val.json'))



# # evaluate the model every 5 epoch.
# evaluation = dict(interval=5)

# Deformable Transformers
load_from = 'checkpoints/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth' #Trained with this

# To resume from certain model: resume_from
#resume_from = '/home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_csaw_optimam_bcdr_seed_33/latest.pth'

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
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=40)

# URL http://download.openmmlab.com/mmdetection/v2.0/

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/mass_data_aug_acr12_bcdr --seed 33 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/data_aug_acr12_csaw --seed 33 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/data_aug_acr12_csaw_bcdr_optimam_seed_33 --seed 33 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/data_aug_acr12_bcdr_optimam_seed_33 --seed 33 --deterministic

# Test:
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr12_csaw_bcdr_optimam_seed_33 --seed 33 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr12_csaw_seed_33 --seed 33 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_csaw_optimam_bcdr_seed_33 --seed 33 --deterministic
# Training ...
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/no_data_aug_seed_42 --seed 42 --deterministic

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_csaw_optimam_seed_33 --seed 33 --deterministic

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_optimam_seed_999 --seed 999 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_csaw_seed_999 --seed 999 --deterministic

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_bcdr_seed_999 --seed 999 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_acr_all_bcdr_optimam_seed_999 --seed 999 --deterministic

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/train_acr123/no_hstd/data_aug_all_high_to_low_seed_999 --seed 999 --deterministic
