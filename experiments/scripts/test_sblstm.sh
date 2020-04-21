#!/bin/bash
#SBATCH -o test_sblstm.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

MODEL="SingleBLSTM"
PARAM="sample.json"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-2s-10c/4s.json" \
    --model_spec "$MODEL_SPEC/$MODEL/$PARAM" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15 