#!/bin/bash

NUM_SOURCES=2
AGG_DUR=2
GT_DUR=1
RANGE=1
RAW_DIR="./"

python mix.py \
    --raw_data_dir $RAW_DIR \
    --num_sources $NUM_SOURCES \
    --aggregate_duration $AGG_DUR \
    --num_examples 3 \
    --ground_truth_durations $GT_DUR $GT_DUR \
    --selected_classes "Acoustic_guitar" "Applause" \
    --dataset_dir "./tmp" \
    --selection_range $RANGE
 
# COUNTER=0
# NUM_CLIPS=1
# PERCENTAGE=0.0
# # DATASET_PATH='/home/tianjunm/Documents/Projects/dataset/dataset_b1'
# DATASET_PATH='tmp/wav'
# echo 'Start generating wav files'
# while [ $COUNTER -lt $NUM_CLIPS ]; do
#     let "PERCENTAGE=$COUNTER / 10"
#     echo -ne "Progress: ${PERCENTAGE}%\r"
#     # echo $((COUNTER + 1))
#     python mix.py \
#         --id $(($COUNTER + 1)) \
#         --num_sources 2 \
#         --duration 2 \
#         --selected_classes 0 1 \
#         --source_durations 0.87 0.87 \
#         --out_path $DATASET_PATH 
#     let COUNTER++
# done
# # echo ''
# let "PERCENTAGE=$COUNTER / 10"
# echo "Progress: ${PERCENTAGE}%"
