#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

source ./config.sh  # Source the common configuration script

k=0

for nq in {28..34}; do
    for cx in "${nCX[@]}"; do
        expN=${N}${nq}q${cx}cx
        k=$((k + 1))
        echo "$k  expN: $expN"
        ./gen_gateList.py -k $cx -i $nCirc --expName $expN -q $nq --basePath ${basePath}
    done
done

echo "done $k jobs"
date
