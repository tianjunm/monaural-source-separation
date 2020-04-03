#!/bin/bash
#SBATCH -o ../sbatch_logs/blstm.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

CONFIG="$WORK_ROOT/experiments/hyperparameter/blstm"

python $WORK_ROOT/train.py --config_path "$CONFIG/000.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/002.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/003.json" --checkpoint_freq 50