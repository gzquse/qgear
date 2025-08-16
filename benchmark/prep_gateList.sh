#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

source ./config.sh  # Source the common configuration script

k=0

# Inception!
basePath=/dataVault2024/dataCudaQ_Aug16

# Ensure the basePath exists
if [ ! -d "$basePath" ]; then
    echo "create $basePath"
    mkdir -p "$basePath"
    cd "$basePath"
    mkdir circ meas post 
    cd -
fi

for nq in {28..29}; do
    for cx in "${nCX[@]}"; do
        expN=${N}${nq}q${cx}cx
        k=$((k + 1))
        echo "$k  expN: $expN"
        ./gen_gateList.py -k $cx -i $nCirc --expName $expN -q $nq --basePath ${basePath}
    done
done

echo "done $k jobs"
date
