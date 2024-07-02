#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
 
# runs in PM login node, bere-OS
k=0
#for nq in {18..19}; do
for nq in {20..33}; do
	for trg in   gpu cpu ; do
	    k=$[ $k +1 ]
	    expN=cg${nq}q
 	    echo $k  expN:$expN   trg:$trg 
	    SCMD="  ./batchPodman.slr  $expN $trg  "

	    if [ "$trg" == "cpu" ]; then
		sbatch -C cpu --exclusive --cpus-per-task=32 --ntasks-per-node=4 -N1   $SCMD
	    elif [ "$trg" == "gpu" ]; then
		sbatch -C gpu --gpus-per-task=4 --ntasks=1 -N1  $SCMD
	    fi	    
	done
done

echo submitted:  $k jobs
date

# testing :   sbatch   -C gpu  ./batchPodman.slr
#       sbatch   -C cpu  ./batchPodman.slr
#
# re-submit one job by hand on FULL node

# re-submit one job by hand on shared 1/4 of a node
#      sbatch  -q shared -C gpu --gpus-per-task 1 --cpus-per-task=32  --ntasks=1 ./batchPodman.slr cb22q gpu 
#     sbatch  -q shared -C cpu  --cpus-per-task=64  --ntasks=1  ./batchPodman.slr cb22q cpu 
