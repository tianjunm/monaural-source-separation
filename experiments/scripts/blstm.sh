#!/bin/bash
#SBATCH -o ../sbatch_logs/blstm.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

CONFIG="$WORK_ROOT/experiments/hyperparameter/blstm"

python $WORK_ROOT/train.py --config_path "$CONFIG/t0-2s-5c/000.json" --checkpoint_freq 50
# python $WORK_ROOT/train.py --config_path "$CONFIG/t0-3s-5c/000.json" --checkpoint_freq 50
# python $WORK_ROOT/train.py --config_path "$CONFIG/t0-5s-5c/000.json" --checkpoint_freq 50