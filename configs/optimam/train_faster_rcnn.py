# The new config inherits a base config to highlight the necessary modification
# Faster-RCNN
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
    # test_cfg=dict(
    #     rcnn=dict(
    #         score_thr=0.05,
    #         nms=dict(type='nms', iou_threshold=0.1),
    #         max_per_img=100))
    )

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('mass',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_train.json'),
    val=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'),
    test=dict(
        img_prefix='',
        classes=classes,
        ann_file='/datasets/OPTIMAM/png_screening_cropped_fixed/data_aug/high_density/detection/cyclegan/high_density_h800/OPTIMAM_HOLOGIC_hologic_mass_no_data_aug_val.json'))

# evaluate the model every 5 epoch.
# evaluation = dict(interval=5)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# Faster-RCNN
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# URL http://download.openmmlab.com/mmdetection/v2.0/
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=20)

#docker build -t mmdetection docker/
#docker run --gpus all --shm-size=8g -itd -v /{DATA_DIR}:/mmdetection/data mmdetection
# set PATH for cuda 11.1 installation
# if [ -d "/usr/local/cuda-11.1/bin/" ]; then
#     export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
#     export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# fi
# conda create -n mmdet python=3.8
#conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.7.1/index.html
#pip install -r requirements/build.txt
#pip install -v -e . 
#MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .
#pip install mxnet_cu102==1.7.0

#Train 
# PYTHONPATH=${PYTHONPATH}:./ python tools/train.py /home/lidia/source/mmdetection/configs/optimam/train_faster_rcnn.py --work-dir /home/lidia/source/mmdetection/experiments/optimam/faster_rcnn/mass_20epochs --seed 999