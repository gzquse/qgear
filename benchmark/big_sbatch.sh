#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

source ./config.sh  # Source the common configuration script

k=0
c=64  # cores for CPU
n=4   # ntasks per node

#targets=("par-cpu" "par-gpu" "adj-gpu")
targets=("adj-gpu")

# Function to submit a job
submit_job() {
    local expN=$1
    local trg=$2
    local s=$3
    local qft=$4
    local SCMD="./batchPodman.slr $expN $trg $s $qft"

    if [ "$trg" == "par-cpu" ]; then
        sbatch -C cpu --exclusive --cpus-per-task=$c --ntasks-per-node=$n -N1 -A $ACCT $SCMD
    else
        sbatch -C gpu --gpus-per-task=4 --ntasks=1 -N1 -A $ACCT $SCMD # currently only one node
    fi
}

for nq in 28; do
    for cx in "${nCX[@]}"; do
        for s in "${shots[@]}"; do
            # Determine expN name based on the value of qft
            if [ "$qft" -eq 1 ]; then
                expN="${N}${nq}q_qft${qft}"
            else
                expN="${N}${nq}q${cx}cx"
            fi

            for trg in "${targets[@]}"; do
                k=$((k + 1))
                echo "$k  expN:$expN trg:$trg shots: $s qft: $qft"
                submit_job "$expN" "$trg" "$s" "$qft"
            done
        done
    done
done

echo "submitted: $k jobs"
date
