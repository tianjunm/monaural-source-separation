#!/bin/bash
#SBATCH -o stt_2s_long.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t1-2s-10c/4s-aaai.json" \
    --model_spec "$MODEL_SPEC/STT/sample2.json" \
    --checkpoint_freq 50 \
    --early_stopping_limit 20

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t2-2s-10c/4s-aaai.json" \
    --model_spec "$MODEL_SPEC/STT/sample2.json" \
    --checkpoint_freq 50 \
    --early_stopping_limit 20
