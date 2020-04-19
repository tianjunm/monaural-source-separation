WORK_ROOT="/Users/shreya/Desktop/monaural-source-separation/"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

DATASET_SPEC="$WORK_ROOT/experiments/hyperparameter/dataset"
MODEL_SPEC="$WORK_ROOT/experiments/hyperparameter/model"

python $WORK_ROOT/train.py \
    --dataset_spec "$DATASET_SPEC/t0-2s-5c/4s-waveunet.json" \
    --model_spec "$MODEL_SPEC/Wave-U-Net/000.json" \
    --checkpoint_freq 50