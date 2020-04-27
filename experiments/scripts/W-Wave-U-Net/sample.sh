#!/bin/sh

WORK_ROOT="/work/sbali/monaural-source-separation"
RESULTS_ROOT="/work/sbali/results/"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

python3 $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-2s-10c/4s-waveunet.json" \
    --model_spec "$MODEL_SPEC/W-Wave-U-Net/t0-2s-10c/000.json" \
    --checkpoint_freq 50