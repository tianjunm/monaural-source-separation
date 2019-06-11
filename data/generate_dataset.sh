#!/bin/bash

DATASET_ID="mixer/a1"
A1_CLASSES=("Acoustic_guitar" "Applause")
E1_CLASSES=("Acoustic_guitar" "Applause" "Saxophone" "Bark" "Cough" "Electric_piano" "Telephone")

NUM_SOURCES=2
# SELECTED_CLASSES=${A1_CLASSES[@]}
AGG_DUR=2
#GT_DUR=1.5
# GT_DURS=(2.0 2.0 2.0 2.0 2.0 2.0 2.0)
GT_DURS=(1.0 1.0)
RANGE=10
RAW_DIR="/usr0/home/tianjunm/datasets/original/FSD_Kaggle/train"
DATASET_DIR="/usr0/home/tianjunm/datasets/processed/$DATASET_ID/test"

python mix.py \
    --raw_data_dir $RAW_DIR \
    --num_sources $NUM_SOURCES \
    --aggregate_duration $AGG_DUR \
    --num_examples 200 \
    --ground_truth_durations "${GT_DURS[@]}" \
    --selected_classes "${A1_CLASSES[@]}" \
    --dataset_dir $DATASET_DIR \
    --selection_range $RANGE
