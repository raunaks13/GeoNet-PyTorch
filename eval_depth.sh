#!/bin/bash

IS_TRAIN=0
BASE_DIR="new-0"

for i in {10000..100000..5000}; do
    CKPT_INDEX=$i
    CKPT_DIR="/ceph/raunaks/GeoNet-PyTorch/reconstruction/models/${BASE_DIR}"
    PRED_FILE="/ceph/raunaks/GeoNet-PyTorch/reconstruction/outputs/${BASE_DIR}/rigid__${CKPT_INDEX}.npy"
    python3 reconstruction/core/GeoNet_main.py --is_train=${IS_TRAIN} --ckpt_index=${CKPT_INDEX} --ckpt_dir=${CKPT_DIR}
    python3 reconstruction/kitti_eval/eval_depth.py --pred_file=${PRED_FILE}
done
