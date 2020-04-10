#!/bin/bash
#SBATCH -o 0.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

CONFIG="$WORK_ROOT/experiments/hyperparameter/cSA-LSTM"

python $WORK_ROOT/train.py --config_path "$CONFIG/t0-2s-10c/paper.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/t0-2s-30c/paper.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/t0-3s-5c/paper.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/t0-5s-5c/paper.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/t1-2s-5c/paper.json" --checkpoint_freq 50
python $WORK_ROOT/train.py --config_path "$CONFIG/t2-2s-5c/paper.json" --checkpoint_freq 50