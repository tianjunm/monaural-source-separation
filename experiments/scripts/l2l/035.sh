#!/bin/bash
#SBATCH -o 035.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-3s-5c/4s.json" \
    --model_spec "$MODEL_SPEC/l2l/baseline.json" \
    --checkpoint_freq 50