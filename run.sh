#!/bin/bash
IS_TRAIN=1
BASE_DIR="new"

CKPT_DIR="/ceph/raunaks/GeoNet-PyTorch/models/${BASE_DIR}"
GRAPHS_DIR="/ceph/raunaks/GeoNet-PyTorch/graphs/${BASE_DIR}"
OUTPUTS_DIR="/ceph/raunaks/GeoNet-PyTorch/outputs"
#WEIGHT_DECAY=0.000001
WEIGHT_DECAY=0
LEARNING_RATE=0.0002
SEED=1000
IMG_HEIGHT=128
IMG_WIDTH=416

python3 reconstruction/core/GeoNet_main.py --img_height=${IMG_HEIGHT} --img_width=${IMG_WIDTH} --is_train=${IS_TRAIN} --ckpt_dir=${CKPT_DIR} --graphs_dir=${GRAPHS_DIR} --weight_decay=${WEIGHT_DECAY} --learning_rate=${LEARNING_RATE} --seed=${SEED}
