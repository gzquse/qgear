#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value
 
# runs in PM login node, bere-OS
k=0
for shots in 10000 1001000 ; do
    for nq in {17..18}; do
#    for nq in {22..32}; do
#     for nq in {22..24}; do	    
	for trg in  gpu cpu  ; do
	    k=$[ $k +1 ]
	    expN=ck${nq}q
 	    echo $k  expN:$expN   trg:$trg  shots:$shots
	    SCMD="  ./batchPodman.slr  $expN $trg $shots "
	    
	    # Slurm incantation depends on GPU vs. CPU
	    if [ "$trg" == "cpu" ]; then
		sbatch -C cpu --exclusive  --ntasks-per-node=1  $SCMD
	    elif [ "$trg" == "gpu" ]; then
		sbatch -C gpu --gpus-per-task=1 --ntasks 4 --gpu-bind=none --module=cuda-mpich  $SCMD
	    fi	    
	done
    done
done

echo submitted:  $k jobs
date

# testing :   sbatch   -C gpu  ./batchPodman.slr
#       sbatch   -C cpu  ./batchPodman.slr
#
# re-submit one job by hand:
#   	sbatch -C gpu --gpus-per-task=1 --ntasks 4 --gpu-bind=none --module=cuda-mpich  ./batchPodman.slr ck29q gpu 1001000
#       sbatch -C cpu --exclusive  --ntasks-per-node=1   ./batchPodman.slr ck24q cpu 1001000
