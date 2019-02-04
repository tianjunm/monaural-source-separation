#/!bin/bash

v_d="raw_clips/active/gun/train"
n_d="raw_clips/ambient/crowd/train"
out_d="./combined"

COUNTER=0
for v in $(find ${v_d} -type f | shuf -n 1); do
    for n in $(find ${n_d} -type f | shuf -n 1); do
        echo $v; echo $n
        python3 combination.py --s_level 1 --v_file $v --n_file $n \
        --id $COUNTER --out_dir $out_d
    done
done
