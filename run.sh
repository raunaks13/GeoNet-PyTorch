#!/bin/bash
IS_TRAIN=1
BASE_DIR="bs-1"

CKPT_DIR="/ceph/raunaks/GeoNet-PyTorch/reconstruction/models/${BASE_DIR}"
GRAPHS_DIR="/ceph/raunaks/GeoNet-PyTorch/reconstruction/graphs/${BASE_DIR}"
OUTPUTS_DIR="/ceph/raunaks/GeoNet-PyTorch/reconstruction/outputs"
WEIGHT_DECAY=0.000001
LEARNING_RATE=0.0001
SEED=1000

python3 reconstruction/core/GeoNet_main.py --is_train=${IS_TRAIN} --ckpt_dir=${CKPT_DIR} --graphs_dir=${GRAPHS_DIR} --weight_decay=${WEIGHT_DECAY} --learning_rate=${LEARNING_RATE} --seed=${SEED}
