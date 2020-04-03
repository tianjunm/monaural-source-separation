#!/bin/bash
#SBATCH -o ../sbatch_logs/output.%a.out # STDOUT


WORK_ROOT="/work/tianjunm/monaural-source-separation"
RESULTS_ROOT="/results/tianjunm/monaural-source-separation"

CONFIG="$WORK_ROOT/experiments/hyperparameter/cSA-LSTM/003.json"

python $WORK_ROOT/train.py --config_path $CONFIG --checkpoint_freq 50