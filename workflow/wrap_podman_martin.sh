#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
CMD=$2
outPath=$3
QCrankLib=$4

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

echo W:args '1='$1 ',2='$2 '3='$3 ',4=',$4
podman-hpc run -it \
     --volume $outPath:/wrk \
     --volume $QCrankLib:/daan_qcrank \
     --workdir /wrk \
     $IMG $CMD 
