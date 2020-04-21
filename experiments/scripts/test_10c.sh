#!/bin/bash
#SBATCH -o test_10c.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-2s-10c/4s-aaai.json" \
    --model_spec "$MODEL_SPEC/STT/sample.json" \
    --checkpoint_freq 50 \
    --early_stopping_limit 20