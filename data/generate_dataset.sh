#!/bin/bash

CLASSES_2=("Acoustic_guitar" "Applause")
CLASSES_3=("Acoustic_guitar" "Applause" "Bark")
CLASSES_5=("Acoustic_guitar" "Applause" "Saxophone" "Bark" "Cough")
CLASSES_7=("Acoustic_guitar" "Applause" "Saxophone" "Bark" "Cough" "Electric_piano" "Telephone")

NUM_SOURCES=5
# NUM_SOURCES=2
# SELECTED_CLASSES=${A1_CLASSES[@]}
AGG_DUR=2
# GT_DURS=(2.0 2.0 2.0 2.0 2.0 2.0 2.0)
GT_DURS=(0.5 0.5 0.5 0.5 0.5) 
# GT_DURS=(1.5 1.5)
RANGE=10
NUM_TRAIN=2000
NUM_TEST=4

DATASET_ID="mixer/${NUM_SOURCES}-${RANGE}-${NUM_TRAIN}-${NUM_TEST}"
RAW_DIR="/usr0/home/tianjunm/datasets/original/FSD_Kaggle/train"
#RAW_DIR="/media/bighdd7/tianjunm/datasets/original/FSDKaggle2018.audio_train"
DATASET_DIR="/usr0/home/tianjunm/datasets/processed/$DATASET_ID/test"
#DATASET_DIR="/media/bighdd7/tianjunm/datasets/processed/$DATSET_ID/train"

python mix.py \
    --raw_data_dir $RAW_DIR \
    --num_sources $NUM_SOURCES \
    --aggregate_duration $AGG_DUR \
    --num_examples $NUM_TEST\
    --ground_truth_durations "${GT_DURS[@]}" \
    --selected_classes "${CLASSES_5[@]}" \
    --dataset_dir $DATASET_DIR \
    --selection_range $RANGE
