#!/bin/bash
#SBATCH -o test_l2l_on_fast_10c.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

MODEL="L2L"
PARAM="baseline.json"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t1-2s-10c/4s.json" \
    --model_spec "$MODEL_SPEC/$MODEL/$PARAM" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15 
python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t2-2s-10c/4s.json" \
    --model_spec "$MODEL_SPEC/$MODEL/$PARAM" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15 
python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-3s-10c/4s.json" \
    --model_spec "$MODEL_SPEC/$MODEL/$PARAM" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15 
python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t1-3s-10c/4s.json" \
    --model_spec "$MODEL_SPEC/$MODEL/$PARAM" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15 
python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t2-3s-10c/4s.json" \
    --model_spec "$MODEL_SPEC/$MODEL/$PARAM" \
    --checkpoint_freq 50 \
    --early_stopping_limit 15 