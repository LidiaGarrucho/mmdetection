# The new config inherits a base config to highlight the necessary modification
# Auto-Assign
_base_ = '../autoassign/autoassign_r50_fpn_8x2_1x_coco.py'
model = dict(bbox_head=dict(num_classes=1))

LONGER_EDGE = 1333 #2100
SHORTER_EDGE = 800 #1700
RESIZE_PARAM = 1 # 2
LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks_all_no_asymmetries.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_landmarks_mass_calc.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_mass_data_aug_high_density.pth'
#LANDMARKS = '/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/optimam_train_hologic_mass_data_aug_high_density.pth'


img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(type='Resize', img_scale=(LONGER_EDGE, SHORTER_EDGE), keep_ratio=True),
    #dict(type='Low2HighBreastDensityAug', checkpoint_name='high_density_h800', img_scale=(LONGER_EDGE, SHORTER_EDGE)),
    #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            #dict(type='ImageStandardisationRGB', landmarks_path=LANDMARKS),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# Modify dataset related settings
dataset_type = 'COCODataset'
save_name = 'mass_data_aug_high_density'
class_name = 'mass' #'distortion' #'calc' #'mass'
classes = (f'{class_name}',)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        pipeline=train_pipeline,
        img_prefix='',
        classes=classes,
        ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_{save_name}_train.json'),
        #ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_only_data_aug_high_density_train.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_{save_name}_val.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_{save_name}_test.json'))

# {'mass': 0, 'calcification': 1, 'distortion': 2}
# dataset_type = 'COCODataset'
# classes = ('mass', 'calcification', 'distortion')
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         pipeline=train_pipeline,
#         img_prefix='',
#         classes=classes,
#         ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_all_pathologies_train.json'),
#     val=dict(
#         pipeline=test_pipeline,
#         img_prefix='',
#         classes=classes,
#         ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_all_pathologies_val.json'),
#     test=dict(
#         pipeline=test_pipeline,
#         img_prefix='',
#         classes=classes,
#         ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_all_pathologies_test.json'))

#{'mass': 0, 'calcification': 1}
# dataset_type = 'COCODataset'
# classes = ('mass', 'calcification')
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         pipeline=train_pipeline,
#         img_prefix='',
#         classes=classes,
#         ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_calc_train.json'),
#     val=dict(
#         pipeline=test_pipeline,
#         img_prefix='',
#         classes=classes,
#         ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_calc_val.json'),
#     test=dict(
#         pipeline=test_pipeline,
#         img_prefix='',
#         classes=classes,
#         ann_file=f'/home/lidia-garrucho/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_calc_test.json'))
# # evaluate the model every 5 epoch.
# evaluation = dict(interval=5)

load_from = 'checkpoints/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth'

# optimizer
#optimizer = dict(lr=0.01, paramwise_cfg=dict(norm_decay_mult=0.))
optimizer = dict(lr=0.00125, paramwise_cfg=dict(norm_decay_mult=0.)) #batch2 lr=0.00125 1 worker lr=0.000625
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11, 14])
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# URL http://download.openmmlab.com/mmdetection/v2.0/

# --gpu-ids 3
# Mass 
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/auto_assign_std --seed 999
# Calc
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/calc/auto_assign_std --seed 999
# Distortion
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/distortion/auto_assign_test --seed 999
# All
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass_calc/auto_assign --seed 999
# CycleGAN Data Aug (and HStd)
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/auto_assign_cyclegan/no_hstd_20ep --seed 999
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/auto_assign_cyclegan/hstd_20ep --seed 999
# python tools/train.py /home/lidia-garrucho/source/mmdetection/configs/optimam/auto_assign.py --work-dir /home/lidia-garrucho/source/mmdetection/experiments/optimam/hologic/mass/auto_assign_cyclegan/only_aug_hstd_20ep --seed 999