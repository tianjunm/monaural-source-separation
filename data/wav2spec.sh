#!/bin/bash

COUNTER=1
DATASET="/home/tianjunm/Documents/Projects/dataset"
WAVS_PATH="$DATASET/dataset_b2_test"
OUT_PATH="$DATASET/b2_spectrograms_test"
percentage=0.0

echo 'Start generating spectrograms for wav files'
for f in ${WAVS_PATH}/*.wav; do
    # echo $f
    let "PERCENTAGE=$COUNTER / 30"
    echo -ne "Progress: ${PERCENTAGE}%\r"
    python wav2spec.py \
        --filename $f \
        --output_dir $OUT_PATH 
    let COUNTER++
    # echo $COUNTER
    # if [[ "$COUNTER" -gt 10 ]]; then
    #     break
    # fi
    # break
done
echo ''
