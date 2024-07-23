#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

source ./config.sh  # Source the common configuration script

k=0
c=64  # cores for CPU
n=4   # ntasks per node
#targets=("par-cpu" "par-gpu" "adj-gpu")
targets=("par-gpu")
# Function to submit a job
submit_job() {
    local expN=$1
    local trg=$2
    local SCMD="./batchPodman.slr $expN $trg"

    if [ "$trg" == "par-cpu" ]; then
        sbatch -C cpu --exclusive --cpus-per-task=$c --ntasks-per-node=$n -N1 -A $ACCT $SCMD
    else
        sbatch -C gpu --gpus-per-task=4 --ntasks=1 -N1 -A $ACCT $SCMD # currently only one node
    fi
}

for nq in {30..31}; do
    for cx in "${nCX[@]}"; do
        expN=${N}${nq}q${cx}cx
        for trg in "${targets[@]}"; do
            k=$((k + 1))
            echo "$k  expN:$expN   trg:$trg"
            submit_job "$expN" "$trg"
        done
    done
done

echo "submitted: $k jobs"
date
