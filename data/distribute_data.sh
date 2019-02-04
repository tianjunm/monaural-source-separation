#/!bin/bash

TEST_SIZE=15
ACTIVE="raw_clips/active"
AMBIENT="raw_clips/ambient"

currd="${AMBIENT}/idling"
testd="${currd}/test"
traind="${currd}/train"
testf="${testd}.txt"
mkdir ${traind} ${testd}
touch ${testf}
find "${currd}" -type f -name "*.wav" | shuf -n 15 > ${testf}
cat ${testf} | while read line
do
    mv ${line} ${testd}
done
mv ${currd}/*wav $traind