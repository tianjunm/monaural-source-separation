#!/bin/bash

# CLASSES_2=("Acoustic_guitar" "Applause")
# CLASSES_3=("Acoustic_guitar" "Applause" "Bark")
# CLASSES_5=("Acoustic_guitar" "Applause" "Saxophone" "Bark" "Cough")
# CLASSES_7=("Acoustic_guitar" "Applause" "Saxophone" "Bark" "Cough" "Electric_piano" "Telephone")

NUM_SOURCES=5
AGG_DUR=2
# GT_DURS=(2.0 2.0 2.0 2.0 2.0 2.0 2.0)
# GT_DURS=(0.5 0.5 0.5 0.5 0.5) 
# GT_DURS=(1.5 1.5)
# RANGE=10
NUM_TRAIN=$(( ${NUM_SOURCES} * 10000))
# NUM_TRAIN=4
NUM_TEST=$(( ${NUM_SOURCES} * 1000))

DATASET_ID="mixer/${NUM_SOURCES}-${NUM_TRAIN}-${AGG_DUR}"
# RAW_DIR="/usr0/home/tianjunm/datasets/original/FSD_Kaggle/train"
RAW_DIR="/home/ubuntu/datasets/original/FSDKaggle/train"
METADATA_PATH="/home/ubuntu/multimodal-listener/data/metadata.csv"
# DATASET_DIR="/usr0/home/tianjunm/datasets/processed/$DATASET_ID/test"
DATASET_DIR="/home/ubuntu/datasets/processed/${DATASET_ID}/train"

# echo $NUM_TRAIN
python mix.py \
    --raw_data_dir $RAW_DIR \
    --num_sources $NUM_SOURCES \
    --metadata_path $METADATA_PATH \
    --aggregate_duration $AGG_DUR \
    --num_examples $NUM_TRAIN \
    --dataset_dir $DATASET_DIR
