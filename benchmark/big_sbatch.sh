#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

source ./config.sh  # Source the common configuration script

k=0
c=64  # cores for CPU
n=4   # ntasks per node

#targets=("par-cpu" "par-gpu" "adj-gpu")
targets=("gpu")

# Function to submit a job
submit_job() {
    local expN=$1
    local trg=$2
    local s=$3
    local qft=$4
    local option=$5
    local SCMD="./batchPodman.slr $expN $trg $s $qft $option"

    if [ "$trg" == "par-cpu" ]; then
        sbatch -C cpu --exclusive --cpus-per-task=$c --ntasks-per-node=$n -N1 -A $ACCT $SCMD
    else
        sbatch -C "gpu&hbm80g" -N1 --gpus-per-task=4 --ntasks-per-node=1 --gpu-bind=none -A $ACCT $SCMD # currently only one node
    fi
}

for nq in {33..34}; do
    if [ "$qft" -eq 1 ]; then
        expN="${N}${nq}q_qft${qft}"
        for s in "${shots[@]}"; do
            for trg in "${targets[@]}"; do
                k=$((k + 1))
                echo "$k  expN:$expN trg:$trg shots: $s qft: $qft option: $option"
                submit_job "$expN" "$trg" "$s" "$qft" "$option"
            done
        done
    else
        for cx in "${nCX[@]}"; do
            expN="${N}${nq}q${cx}cx"
            for s in "${shots[@]}"; do
                for trg in "${targets[@]}"; do
                    k=$((k + 1))
                    echo "$k  expN:$expN trg:$trg shots: $s qft: $qft option: $option"
                    submit_job "$expN" "$trg" "$s" "$qft" "$option"
                done
            done
        done
    fi
done

echo "submitted: $k jobs"
date
