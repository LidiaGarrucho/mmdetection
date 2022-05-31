
_base_ = '../vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
model = dict(bbox_head=dict(num_classes=1))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 2
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    #dict(type='HistogramStretchingRGB', landmarks_path=LANDMARKS),
    #dict(type='CombinedRescaleNyulStretchRGB', landmarks_path=LANDMARKS),
    dict(
        type='Resize',
        img_scale=[(LONGER_EDGE, 480*RESIZE_PARAM), (LONGER_EDGE, 960*RESIZE_PARAM)],
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
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    #dict(type='HistogramStretchingRGB', landmarks_path=LANDMARKS),
    #dict(type='CombinedRescaleNyulStretchRGB', landmarks_path=LANDMARKS),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(LONGER_EDGE, SHORTER_EDGE),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
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
        
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_bcdr_optimam_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_optimam_train.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_optimam_train.json'),
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
        
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_bcdr_optimam_val.json'), 
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_optimam_val.json'),
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_optimam_val.json'),
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

        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_bcdr_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_csaw/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_csaw_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_bcdr_optimam/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_bcdr_optimam_val.json'))
        #ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/mass_acr123_data_aug_all_high_to_low/OPTIMAM_HOLOGIC_mass_acr123_data_aug_all_high_to_low_val.json'))


# optimizer

# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    lr=0.00125, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.1,
#     step=[8, 11])

lr_config = dict(step=[50, 75])
runner = dict(type='EpochBasedRunner', max_epochs=100)

load_from = 'checkpoints/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'

# # learning policy
# lr_config = dict(step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=24)

#Train
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/vfnet.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/vfnet/hstd --seed 999
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/vfnet.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/vfnet --seed 999
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/vfnet.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/vfnet/stretching --seed 999
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/vfnet.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/vfnet/combined --seed 999

#python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/vfnet.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/vfnet/2100_1700 --seed 999
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/vfnet_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/vfnet/train_acr123/no_hstd/data_aug_all_csaw_bcdr_optimam_seed_999 --seed 999 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/vfnet_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/vfnet/train_acr123/no_hstd/no_data_aug_seed_999 --seed 999 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/vfnet_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/vfnet/train_acr123/no_hstd/data_aug_all_optimam_seed_999 --seed 999 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/vfnet_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/vfnet/train_acr123/no_hstd/data_aug_all_csaw_seed_999 --seed 999 --deterministic
#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/vfnet_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/vfnet/train_acr123/no_hstd/data_aug_all_bcdr_seed_999 --seed 999 --deterministic

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/vfnet_aug_cyclegan.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/vfnet/train_acr123/no_hstd/data_aug_all_100_epochs_seed_5 --seed 5 --deterministic



                                                        