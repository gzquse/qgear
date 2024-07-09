#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value

ACCT=nstaff
# runs in PM login node, bere-OS
k=0
#for nq in 16  ; do
for nq in {20..28}; do	    
	for trg in par-cpu par-gpu  adj-gpu ; do
	    k=$[ $k +1 ]
	    expN=cg${nq}q
 	    echo $k  expN:$expN   trg:$trg 
	    SCMD="  ./batchPodman-martin.slr  $expN $trg  "
	    
	    # Slurm incantation depends on GPU vs. CPU
	    if [ "$trg" == "par-cpu" ]; then
	       sbatch  -C cpu --exclusive --cpus-per-task=32 --ntasks-per-node=4 -N1 -A $ACCT $SCMD
	    elif [ "$trg" == "par-gpu" ]; then
	        sbatch  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -A $ACCT $SCMD
	    elif [ "$trg" == "adj-gpu" ]; then
	        sbatch  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -A $ACCT $SCMD			
	    fi	    
	done
done

echo submitted:  $k jobs
date

# testing :   sbatch   -C gpu  ./batchPodman.slr
#       sbatch   -C cpu  ./batchPodman.slr
#
# re-submit one job by hand on FULL node
#   	sbatch -C gpu --gpus-per-task=1 --ntasks 4 --gpu-bind=none --module=cuda-mpich  ./batchPodman.slr ck29q gpu 1001000
#       sbatch -C cpu --exclusive  --ntasks-per-node=1   ./batchPodman.slr c20q cpu 

# re-submit one job by hand on shared 1/4 of a node
#      sbatch  -q shared -C gpu --gpus-per-task 1 --cpus-per-task=32  --ntasks=1 ./batchPodman.slr cb22q gpu 
#     sbatch  -q shared -C cpu  --cpus-per-task=64  --ntasks=1  ./batchPodman.slr cb22q cpu 
