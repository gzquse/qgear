#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value

ACCT=nintern
# runs in PM login node, bere-OS
k=0
#for nq in 16  ; do
for nq in {20..28}; do
    echo nq=nq
    for trg in par-cpu par-gpu  adj-gpu ; do
	k=$[ $k +1 ]
	expN=cg${nq}q
 	echo $k  expN:$expN   trg:$trg 
	SCMD="  ./batchPodman.slr  $expN $trg  "
	    
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

exit

---- TESTING ----
*** CPU  interactive
salloc  -C cpu --exclusive --cpus-per-task=32 --ntasks-per-node=4 -N1  -t 4:00:00 -q interactive

  ./batchPodman.slr  cg18q par-cpu


*** GPU  interactive
salloc  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -t 4:00:00 -q interactive -A nstaff

GPUs  parallel , tag par-gpu
./batchPodman.slr  cg18q  par-gpu


***GPU  interactive
salloc  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -t 4:00:00 -q interactive -A nstaff
GPUs  adjoined, tag adj-gpu

  ./batchPodman.slr  cg18q  adj-gpu
