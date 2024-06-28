#!/bin/bash
export IMG=gzquse/cudaquanmpi-qiskit1:p3
export CMD="python3 cuda_quantum_test/examples/python/cuquantum_backends.py -target nvidia-mqpu "
#CMD=" ls -l "
#CMD=" nvidia-smi "
export CFSH=/pscratch/sd/g/gzquse

export BASE_DIR=/GRADIENT_IMAGE  # here git has home
export WORK_DIR=$BASE_DIR
#WORK_DIR=/


echo W:myRank is $SLURM_PROCID

if [ $SLURM_PROCID -eq 0 ]; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

#hostname
#echo W:args '1='$1 '


podman-hpc run --gpu --rm\
    -e DISPLAY  -v $HOME:$HOME -e HOME  \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --volume $CFSH/$BASE_DIR:/$BASE_DIR \
    --workdir $WORK_DIR  \
    $IMG $CMD
