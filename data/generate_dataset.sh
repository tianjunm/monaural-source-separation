#!/bin/bash

NUM_SOURCES=7
SELECTED_CLASSES=("Acoustic_guitar" "Applause" "Saxophone" "Bark" "Cough" "Electric_piano" "Telephone")
AGG_DUR=4
#GT_DUR=1.5
GT_DURS=(2.0 2.0 2.0 2.0 2.0 2.0 2.0)
RANGE=10
RAW_DIR="/home/ubuntu/dataset/FSDKaggle2018.audio_train"
DATASET_DIR="/home/ubuntu/dataset/dataset_e1/train_e1"

python mix.py \
    --raw_data_dir $RAW_DIR \
    --num_sources $NUM_SOURCES \
    --aggregate_duration $AGG_DUR \
    --num_examples 6000 \
    --ground_truth_durations "${GT_DURS[@]}" \
    --selected_classes "${SELECTED_CLASSES[@]}" \
    --dataset_dir $DATASET_DIR \
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
