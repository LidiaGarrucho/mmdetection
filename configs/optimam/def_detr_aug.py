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
        type='ResNetMixStyle',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3), 
        frozen_stages=1, #4
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        #init_cfg=None,
        mixstyle=True,
        mixstyle_layers=[0,1,2]),
        neck=dict(freeze=True),
        bbox_head=dict(num_classes=1, with_box_refine=True, freeze=None))#, 'DeformableDetrTransformerDecoder']))
        #bbox_head=dict(num_classes=1, with_box_refine=True, freeze=['DetrTransformerEncoder', 'DeformableDetrTransformerDecoder']))

LONGER_EDGE = 1333
SHORTER_EDGE = 800
RESIZE_PARAM = 1
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_mass_data_aug_high_density.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/data_aug/optimam_bcdr_data_aug_mass_train_set.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/data_aug/optimam_bcdr_no_data_aug_mass_train_set.pth'
LANDMARKS = '/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/optimam_hologic_data_aug_mass_train_set.pth'

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
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
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
    samples_per_gpu=2, #2
    workers_per_gpu=2, #2
    train=dict(
        pipeline=train_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr1_study_lvl_train.json'),
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr123_study_lvl_train.json'),

        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_train.json'),
        #ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_only_data_aug_high_density_train.json'),
        #ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/data_aug/OPTIMAM_HOLOGIC_mass_breast_density_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/data_aug/OPTIMAM_HOLOGIC_mass_breast_density_train_no_aug.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_HOLOGIC_finetune_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/BCDR/cropped/detection/masses/BCDR_mass_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/BCDR/cropped/detection/masses/BCDR_mass_train_4_clients.json'),
        #ann_file='/home/lidia-garrucho/datasets/BCDR/cropped/detection/masses/BCDR_mass_train_8_clients.json'),
        #ann_file='/home/lidia-garrucho/datasets/INBREAST/AllPNG_cropped/detection/masses/INBreast_mass_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_Philips_Digital_Mammography_Sweden_AB_finetune_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_GE_MEDICAL_SYSTEMS_finetune_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_SIEMENS_finetune_train.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_combined_finetune_train.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr1_study_lvl_val.json'),
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr123_study_lvl_val.json'),

        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/data_aug/OPTIMAM_HOLOGIC_mass_breast_density_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_HOLOGIC_finetune_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/BCDR/cropped/detection/masses/BCDR_mass_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/INBREAST/AllPNG_cropped/detection/masses/INBreast_mass_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_Philips_Digital_Mammography_Sweden_AB_finetune_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_GE_MEDICAL_SYSTEMS_finetune_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_SIEMENS_finetune_val.json'),
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_combined_finetune_val.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr1_study_lvl_val.json'))
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr12_study_lvl_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_data_aug_acr123_study_lvl_val.json'))

        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_test.json'))
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/data_aug/OPTIMAM_HOLOGIC_mass_breast_density_val.json'))
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_HOLOGIC_finetune_test.json'))
        #ann_file='/home/lidia-garrucho/datasets/BCDR/cropped/detection/masses/BCDR_mass_test.json'))
        #ann_file='/home/lidia-garrucho/datasets/INBREAST/AllPNG_cropped/detection/masses/INBreast_mass_test.json'))
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_Philips_Digital_Mammography_Sweden_AB_finetune_test.json'))
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_GE_MEDICAL_SYSTEMS_finetune_test.json'))
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_SIEMENS_finetune_test.json'))

# # evaluate the model every 5 epoch.
# evaluation = dict(interval=5)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# Deformable Transformers
# load_from = 'checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
load_from = 'checkpoints/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth' #Trained with this
# resume_from = '/home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/randconv_k135_mix05_alpha05_hstd/epoch_12.pth'
# load_from = 'checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
# DETR
# load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
# To resume from certain model: resume_from

#load_from = 'experiments/optimam/hologic/mass/def_detr/mixstyle_res13_hstd/epoch_13.pth' # test finetune Hstd
#load_from = 'experiments/optimam/hologic/mass/def_detr/test_results/epoch_10.pth'
#load_from =  'experiments/optimam/hologic/mass/def_detr/k135_mix05_cutout_mixstyle_hstd_resume/epoch_23.pth'
#load_from = 'experiments/optimam/hologic/mass/def_detr/test_results/epoch_10.pth' # Baseline
#load_from = 'experiments/optimam/hologic/mass/def_detr/hstd/epoch_10.pth' # Baseline + Hstd
#load_from = 'experiments/optimam/hologic/mass/def_detr/mixstyle_r123_cutout_t2_hstd/epoch_23.pth'
#load_from = 'experiments/optimam/hologic/mass/def_detr/mixstyle_r123_cutout_t2_randconv_k123_hstd/epoch_12.pth'

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-05, #lr=2e-4, 2 images x 8 GPU: 2 images x 1 GPU
    #lr=2.5e-05,
    #lr=1e-04,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=50)

# URL http://download.openmmlab.com/mmdetection/v2.0/


#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/prior_high_bcdr --seed 999
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/new_high_bcdr --seed 999
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/no_aug_high_bcdr --seed 999
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/new_high_bcdr_seed_42 --seed 42
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/no_aug_high_bcdr_seed_33 --seed 33
#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/def_detr/no_aug_high_bcdr_seed_42 --seed 42

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

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_aug.py --work-dir /home/lidia/source/mmdetection/experiments/high_density/cyclegan/high_density_h800/no_data_aug_seed999_new --seed 999 --deterministic