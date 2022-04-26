#Docker build from mmdetection/ folder
#Install container toolkit so that the GPUs are available
#https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
#sudo docker build -t mmdetection docker/

ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0 8.5+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install MMDetection
RUN conda clean --all
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN mkdir /mmdetection
ADD . /mmdetection 
WORKDIR /mmdetection
#WORKDIR .
ENV FORCE_CUDA="1"

# RUN this after copying the files to the container
RUN pip install --no-cache-dir -r /mmdetection/requirements/build.txt
RUN pip install --no-cache-dir -e /mmdetection/.
RUN pip install scikit-image

ENV GPU_SELECTED="0"
ENV EXEC_FILE="/mmdetection/tools/train.py"
ENV CONFIG_FILE="/mmdetection/configs/optimam/def_detr.py"
ENV SAVE_PATH="/mmdetection/experiments/high_density/cyclegan/high_density_h800/HOLII"
ENV CUDA_SEED="999"

# -e GPU_SELECTED -e SAVE_PATH -e CONFIG_FILE -e EXEC_FILE -e CUDA_SEED
#docker run --shm-size=8g --gpus all -v /datasets:/datasets mmdet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_SELECTED python $EXEC_FILE $CONFIG_FILE --work-dir $SAVE_PATH --seed $CUDA_SEED --deterministic

#VOLUME [ "/datasets" ]
#RUN CUDA_VISIBLE_DEVICES=${GPU_SELECTED} CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ${EXEC_FILE} ${CONFIG_FILE} --work-dir ${SAVE_PATH} --seed ${CUDA_SEED} --deterministic

# sudo docker run -it --rm --gpus all ubuntu nvidia-smi
# sudo docker run --shm-size=8g --gpus all --rm -it -v /datasets:/datasets -v /home/lidia/source/mmdetection:/test mmdetection_t0
# git config --global --add safe.directory /test

# Acceso a las imagenes 
# docker run --gpus all -v /datasets:/datasets mmdet
# docker exec -it mmdet /bin/bash
# git config --global --add safe.directory /test

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/train.py ./configs/optimam/def_detr.py --work-dir /mmdetection/experiments/high_density/cyclegan/high_density_h800/HOLII --seed 999 --deterministic
# Change owner the work_dir or create the folder first


