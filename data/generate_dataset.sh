#!/bin/bash

COUNTER=0
NUM_CLIPS=1000
PERCENTAGE=0.0
DATASET_PATH='/home/tianjunm/Documents/Projects/dataset/dataset_b1'
echo 'Start generating wav files'
while [ $COUNTER -lt $NUM_CLIPS ]; do
    let "PERCENTAGE=$COUNTER / 10"
    echo -ne "Progress: ${PERCENTAGE}%\r"
    # echo $((COUNTER + 1))
    python mix.py \
        --id $(($COUNTER + 1)) \
        --num_sources 2 \
        --duration 2 \
        --selected_classes 0 1 \
        --source_durations 0.87 0.87 \
        --out_path $DATASET_PATH 
    let COUNTER++
done
# echo ''
let "PERCENTAGE=$COUNTER / 10"
echo "Progress: ${PERCENTAGE}%"
