#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
#CMD=" cat etc/os-release "
CMD=" pwd "
CMD=" hostname "
CMD=" run_cudaq_gateList.py --expN fsdgfsdgf --myRank $SLURM_PROCID "
#outPath=$3
#QCrankLib=$4

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi
#hostname
echo W:args '1='$1 #',2='$2 
podman-hpc run -it \
     $IMG $CMD 

#     --volume $outPath:/wrk \
#     --volume $QCrankLib:/daan_qcrank \
#    --workdir /wrk \
