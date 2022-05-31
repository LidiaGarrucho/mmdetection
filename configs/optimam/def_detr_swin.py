# base = [
# '../base/datasets/coco_detection.py', '../base/default_runtime.py'
# ]
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
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=200,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=8,
        #         max_h_size=8,
        #         max_w_size=8,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=2,
        #         max_h_size=20,
        #         max_w_size=20,
        #         always_apply=True,
        #         p=1),
        # ], p=0.7) #test0
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=50,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=500,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1)
        # ], p=0.7) #test1
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=500,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=1000,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1)
        # ], p=0.7) #test2
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
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=13000,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=50000,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1)
        # ], p=0.5) #test2 5% of total pixels
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=50,
        #         max_h_size=5,
        #         max_w_size=5,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=500,
        #         max_h_size=4,
        #         max_w_size=4,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=2000,
        #         max_h_size=3,
        #         max_w_size=3,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=5000,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=10000,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1)
        # ], p=0.7) #test3
        # type='OneOf',
        # transforms=[
            # dict(
            #     type='Cutout',
            #     num_holes=50,
            #     max_h_size=5,
            #     max_w_size=5,
            #     always_apply=True,
            #     p=1),
            # dict(
            #     type='Cutout',
            #     num_holes=500,
            #     max_h_size=4,
            #     max_w_size=4,
            #     always_apply=True,
            #     p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=4000,
        #         max_h_size=3,
        #         max_w_size=3,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=10000,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=20000,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1)
        # ], p=0.7) #test4
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='MultiplicativeNoise',
        #         multiplier=(0.9, 1.1),
        #         per_channel=False,
        #         elementwise=False,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='MultiplicativeNoise',
        #         multiplier=(0.9, 1.1),
        #         per_channel=False,
        #         elementwise=True,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='RandomRain',
        #         always_apply=True,
        #         p=1),
        #     #RandomRain (slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5)
        #     dict(
        #         type='ISONoise',
        #         color_shift=(0.01, 0.05),
        #         intensity=(0.1, 0.5),
        #         always_apply=True,
        #         p=1)
        # ], p=0.5) #test5
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=50,
        #         max_h_size=5,
        #         max_w_size=5,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=500,
        #         max_h_size=4,
        #         max_w_size=4,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=2000,
        #         max_h_size=3,
        #         max_w_size=3,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=5000,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=10000,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='MultiplicativeNoise',
        #         multiplier=(0.9, 1.1),
        #         per_channel=False,
        #         elementwise=False,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='MultiplicativeNoise',
        #         multiplier=(0.9, 1.1),
        #         per_channel=False,
        #         elementwise=True,
        #         always_apply=True,
        #         p=1)
        # ], p=0.7) #test6
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='Cutout',
        #         num_holes=10000,
        #         max_h_size=3,
        #         max_w_size=3,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=10000,
        #         max_h_size=2,
        #         max_w_size=2,
        #         always_apply=True,
        #         p=1),
        #     dict(
        #         type='Cutout',
        #         num_holes=10000,
        #         max_h_size=1,
        #         max_w_size=1,
        #         always_apply=True,
        #         p=1)
        # ], p=0.7) #test7
        # type='OneOf',
        # transforms=[
        #     dict(
        #         type='MultiplicativeNoise',
        #         multiplier=(0.9, 1.1),
        #         per_channel=False,
        #         elementwise=False,
        #         always_apply=True,
        #         p=1)
        #     # dict(
        #     #     type='Cutout',
        #     #     num_holes=10000,
        #     #     max_h_size=2,
        #     #     max_w_size=2,
        #     #     always_apply=True,
        #     #     p=1),
        #     # dict(
        #     #     type='Cutout',
        #     #     num_holes=5000,
        #     #     max_h_size=1,
        #     #     max_w_size=1,
        #     #     always_apply=True,
        #     #     p=1)
        # ], p=0.5) #test9
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RandomBrightnessContrast',
    #             brightness_limit=[0.1, 0.3],
    #             contrast_limit=[0.1, 0.3],
    #             p=1.0),
    #         dict(
    #             type='RandomContrast',
    #             limit=0.2,
    #             p=1.0),
    #         dict(
    #             type='RandomGamma',
    #             gamma_limit=[80, 120],
    #             p=1.0),
    #         dict(
    #             type='RandomBrightness',
    #             limit=0.2,
    #             p=1.0),
    #         dict(
    #             type='CLAHE',
    #             clip_limit=4.0,
    #             tile_grid_size=[8,8],
    #             p=1.0),
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.2,
    #             contrast=0.2,
    #             saturation=0.2,
    #             hue=0.2,
    #             p=1.0),
    #         dict(
    #             type='Equalize',
    #             mode='cv',
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0),
    #         dict(type='InvertImg', p=1.0)
    #     ],
    #     p=0.8),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0),
    #         dict(type='GaussNoise', var_limit=[10.0, 50.0], mean=0, p=1.0),
    #         dict(type='ISONoise', color_shift=[0.01, 0.05], intensity=[0.1, 0.5], p=1.0),
    #         dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.1)
    #     ],
    #     p=0.1),
    #dict(type='InvertImg', p=0.2)
]
model = dict(
    # backbone=dict(
    #     type='ResNetMixStyle',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(1, 2, 3), 
    #     frozen_stages=1,
    #     #frozen_stages=4,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    #     #init_cfg=None,
    #     mixstyle=True,
    #     mixstyle_layers=[0,1,2]),
        backbone=dict(
        type='SwinTransformer',
        frozen_stages=1, # LIDIA: same as resnet frozen_stages (int): Stages to be frozen (stop grad and set eval mode). Default: -1 (-1 means not freezing any parameters).
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0,1,2,3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/swin_tiny_patch4_window7_224.pth')
        ),
        neck=dict(freeze=True),
        bbox_head=dict(num_classes=1, with_box_refine=True, freeze=None))#, 'DeformableDetrTransformerDecoder']))
        #bbox_head=dict(num_classes=1, with_box_refine=True, freeze=['DetrTransformerEncoder', 'DeformableDetrTransformerDecoder']))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 1
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
LANDMARKS = '/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'), #color_type='grayscale'), #'color', 'grayscale', 'unchanged'
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(LONGER_EDGE, SHORTER_EDGE), keep_ratio=True),
    # dict(type='RandConvAug', kernel_size=(1,3,5), mixing=True, identity_prob=0.5, mixing_alpha=0.5,
    #         img_scale=(LONGER_EDGE, SHORTER_EDGE), img_std=[58.395, 57.12, 57.375], img_mean=[123.675, 116.28, 103.53],
    #         in_channels=1, to_rgb=False), #to_rgb=True means no Hstd
    # dict(type='Low2HighBreastDensityAug', checkpoint_name='high_density_h800', img_scale=(LONGER_EDGE, SHORTER_EDGE)),
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
                    override=True) # if resized before this call set to True
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
    samples_per_gpu=1, #2
    workers_per_gpu=1, #2
    train=dict(
        pipeline=train_pipeline,
        img_prefix='',
        classes=classes,
        #ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_train.json'),
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_train.json'),
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
        # ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_val.json'),
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_val.json'),
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
        # ann_file='/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_test.json'))
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_test.json'))
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

# Don't load COCO pretrained
# load_from = None
# resume_from = None

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-05, #lr=2e-4, 2 images x 8 GPU: 2 images x 1 GPU (Internet: lr=2e-4/16 if 2 batch size and 2 workers per GPU)
    #lr=2.5e-05,
    #lr=1e-04,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=0.2, norm_type=2))
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=60)
# TODO add this IOU thresholds in all the models
# checkpoint_config = dict(interval=5)
# log_config = dict(
# interval=267,
# hooks=[
# dict(type='TextLoggerHook')
# ])
# evaluation = dict(
#  interval=1, # Evaluation interval
#  save_best='bbox_mAP',
#  iou_thrs=[0.1, 0.15, 0.20, 0.25, 0.50]
# )
#Check Issue using OPTIMAM: https://github.com/open-mmlab/mmdetection/issues/7885


#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_swin.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/swin_backbone/hstd_only_seed_999 --seed 999 --deterministic
