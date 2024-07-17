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


*** GPU   parallel interactive
salloc  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -t 4:00:00 -q interactive -A nstaff

./batchPodman.slr  cg18q  par-gpu


***GPU adjoined,  interactive
salloc  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -t 4:00:00 -q interactive -A nstaff

  ./batchPodman.slr  cg18q  adj-gpu

TRY - no impact
 salloc  -C gpu  --gpu-bind=none --gpus-per-task=4 --ntasks=1 -N1  -t 4:00:00 -q interactive -A nstaff 
  
= = = =
  cd prjs/2024_martin_gradient-dev/benchmark
salloc  -C gpu   --gpus-per-task=4 --ntasks=1 -N1  -t 4:00:00 -q interactive -A nstaff
xterm&  

export PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/nintern/gzquse/podman_common/
IMG=gzquse/cudaquanmpi-qiskit:p6
podman-hpc run   --privileged -it --gpu --volume `pwd`:/wrk  --workdir /wrk    -e OMPI_ALLOW_RUN_AS_ROOT=1    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1   -e UCX_WARN_UNUSED_ENV_VARS=n  $IMG bash

source ./distributed/activate_custom_mpi.sh
# use 4 GPU
mpirun -np 4 python3 -u ./simple_ghz50_cudaq.py 
