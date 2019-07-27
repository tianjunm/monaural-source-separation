#!/bin/bash

python train.py \
    --spect_dim 129 690 \
    --batch_size 4 \
    --model "B1" \
    --job_id 4 \
    --gpu_id 1 \
    --train_dir "/usr0/home/tianjunm/datasets/processed/mixer/a1/train" \
    --test_dir "/usr0/home/tianjunm/datasets/processed/mixer/a1/test" \
    --model_dir "/usr0/home/tianjunm/multimodal-listener/pretrained" \
    --output_dir "/usr0/home/tianjunm/multimodal-listener/runs"