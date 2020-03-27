#!/bin/bash
#SBATCH -o ../sbatch_logs/output.%a.out # STDOUT


ROOT="/work/tianjunm/monaural-source-separation"
CONFIG="$ROOT/experiments/hyperparameter/stt/000.json"


echo $CONFIG
python $ROOT/train.py -h