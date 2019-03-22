#!/bin/bash

COUNTER=1
WAVS_PATH="dataset_a1/"
PERCENTAGE=0.0

echo 'Start generating spectrograms for wav files'
for f in ${WAVS_PATH}/*.wav; do
    # echo $f
    let "PERCENTAGE=$COUNTER / 30"
    echo -ne "Progress: ${PERCENTAGE}%\r"
    python wav2spec.py \
        --filename $f \
        --output_dir a1_spectrograms 
    let COUNTER++
    # echo $COUNTER
    # if [[ "$COUNTER" -gt 10 ]]; then
    #     break
    # fi
done
echo ''
