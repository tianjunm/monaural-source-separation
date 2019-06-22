#!/bin/bash

DATASET_CONFIG="5-10"
DATASET_PATH="${DATASET_CONFIG}-2000-500"
python ../train.py \
    --num_sources 5 \
    --spect_dim 129 690 \
    --batch_size 4 \
    --model google \
    --criterion min \
    --gpu_id 2 \
    --job_id 0 \
    --train_dir "/usr0/home/tianjunm/datasets/processed/mixer/${DATASET_PATH}/train" \
    --test_dir "/usr0/home/tianjunm/datasets/processed/mixer/${DATA_PATH}/test" \
    --model_dir "/usr0/home/tianjunm/multimodal-listener/pretrained" \
    --output_dir "/usr0/home/tianjunm/multimodal-listener/runs" \
    --dataset $DATASET_CONFIG
