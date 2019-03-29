#!/bin/bash

COUNTER=1
NUM_CLIPS=1000
PERCENTAGE=0.0

echo 'Start generating wav files'
while [ $COUNTER -lt $NUM_CLIPS ]; do
    let COUNTER++
    let "PERCENTAGE=$COUNTER / 10"
    echo -ne "Progress: ${PERCENTAGE}%\r"
    # sleep 0.01
    python mix.py \
    --id ${COUNTER} \
    --num_sources 2 \
    --duration 4 \
    --selected_classes 4 9 \
    --wav_ids 0 0 \
    --source_duration 1 1 \
    --out_path "dataset_a1"
done
echo ''