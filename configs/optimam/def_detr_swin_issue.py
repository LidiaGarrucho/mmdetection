base = [
'../base/datasets/coco_detection.py', '../base/default_runtime.py'
]

model = dict(
type='DeformableDETR',
backbone=dict(
type='SwinTransformer',
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
#init_cfg=None
),
neck=dict(
    type='ChannelMapper',
    in_channels=[96, 192, 384, 768],
    kernel_size=1,
    out_channels=256,
    act_cfg=None,
    norm_cfg=dict(type='GN', num_groups=32),
    num_outs=4),
bbox_head=dict(
    type='DeformableDETRHead',
    num_query=300,
    num_classes=1,
    in_channels=2048,
    sync_cls_avg_factor=True,
    as_two_stage=False,
    transformer=dict(
        type='DeformableDetrTransformer',
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention', embed_dims=256),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder=dict(
            type='DeformableDetrTransformerDecoder',
            num_layers=6,
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256)
                ],
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')))),
    positional_encoding=dict(
        type='SinePositionalEncoding',
        num_feats=128,
        normalize=True,
        offset=-0.5),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
# training and testing settings
train_cfg=dict(
    assigner=dict(
        type='HungarianAssigner',
        cls_cost=dict(type='FocalLossCost', weight=2.0),
        reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
test_cfg=dict(max_per_img=100)
)

load_from ='checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
# work_dir = '/content/drive/MyDrive/OPTIMAM_7626/to_save_files_and_logs_deformable_detr_swin'
optimizer = dict(
type='AdamW',
lr=2e-4/16,
weight_decay=0.0001,
paramwise_cfg=dict(
custom_keys={
'backbone': dict(lr_mult=0.1),
'sampling_offsets': dict(lr_mult=0.1),
'reference_points': dict(lr_mult=0.1)
}))

optimizer_config = dict(grad_clip=dict(max_norm=0.2, norm_type=2))

lr_config = dict(
policy='step',
gamma=0.1,
warmup='linear',
warmup_iters=8010,
#warmup_ratio=1.0 / 3,
step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=267,
    hooks=[dict(type='TextLoggerHook')]
)
evaluation = dict(
interval=1, # Evaluation interval
save_best='bbox_mAP',
iou_thrs=[0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
)
auto_scale_lr = dict(base_batch_size=32)

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
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_train.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_val.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/detection/mask_rcnn/OPTIMAM_mass_test.json'))

#OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/def_detr_swin_issue.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/hologic/mass/def_detr/swin_backbone/hstd_only_seed_999 --seed 999 --deterministic
