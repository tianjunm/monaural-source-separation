#!/bin/bash

COUNTER=1
DATASET="/Users/tianjunma/Projects/dataset"
WAVS_PATH="$DATASET/dataset_a1"
OUT_PATH="$DATASET/a1_spectrograms"
PERCENTAGE=0.0

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
done
echo ''
