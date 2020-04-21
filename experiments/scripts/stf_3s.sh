#!/bin/bash
#SBATCH -o stf_3s.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-3s-10c/4s-aaai.json" \
    --model_spec "$MODEL_SPEC/STF/sample.json" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t1-3s-10c/4s-aaai.json" \
    --model_spec "$MODEL_SPEC/STF/sample.json" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t2-3s-10c/4s-aaai.json" \
    --model_spec "$MODEL_SPEC/STF/sample.json" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15