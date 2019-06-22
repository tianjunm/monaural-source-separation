#!/bin/bash

DATASET_PATH="3-10-2000-400"
python ../train.py \
    --num_sources 3 \
    --spect_dim 129 690 \
    --batch_size 4 \
    --model "B1" \
    --job_id 5 \
    --gpu_id 1 \
    --train_dir "/usr0/home/tianjunm/datasets/processed/mixer/${DATASET_PATH}/train" \
    --test_dir "/usr0/home/tianjunm/datasets/processed/mixer/${DATA_PATH}/test" \
    --model_dir "/usr0/home/tianjunm/multimodal-listener/pretrained" \
    --output_dir "/usr0/home/tianjunm/multimodal-listener/runs"
