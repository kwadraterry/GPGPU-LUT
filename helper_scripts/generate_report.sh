#!/bin/bash
# ex. run: ./helper_scripts/generate_report.sh "./ColorHistograms-cuda/build/ColorHistogram" "data/" 10
declare -a sizes=(1 5 10 15 25 50 75 100 150 250 500 1000 2500 5000 7500)

for size in "${sizes[@]}";
do
        name=$(printf '%s/plasma%04u.png' "$2" "$size");
        time_cpu=`"$1" "$name" -c -q -t "$3" | grep -oh -E "[[:digit:]]+.[[:digit:]]+ sec" | cut -f 1 -d " "`
        time_gpu=`"$1" "$name" -g -q -t "$3" | grep -oh -E "[[:digit:]]+.[[:digit:]]+ sec" | cut -f 1 -d " "`
        time_mgpu=`"$1" "$name" -m -q -t "$3" | grep -oh -E "[[:digit:]]+.[[:digit:]]+ sec" | cut -f 1 -d " "`
        printf "%4d, %s, %s, %s\n" "$size" "$time_cpu" "$time_gpu" "$time_mgpu"
done;
 

