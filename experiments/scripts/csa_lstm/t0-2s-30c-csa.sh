#!/bin/bash
#SBATCH -o t0-2s-30c.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

CONFIG="$WORK_ROOT/experiments/hyperparameter/cSA-LSTM"

python $WORK_ROOT/train.py --config_path "$CONFIG/t0-2s-30c/paper.json" --checkpoint_freq 50