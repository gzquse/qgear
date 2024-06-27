#!/bin/bash

export IMG=gzquse/cudaquanmpi-qiskit1:p2.1
export CMD="python3 cuda_quantum_test/examples/python/cuquantum_backends.py --target nvidia-mgpu"
export CFSH=/pscratch/sd/g/gzquse
export JNB_PORT=' '
export BASE_DIR=GRADIENT_IMAGE  # here git has home
export WORK_DIR=$BASE_DIR
export DATA_VAULT=${CFSH}/quantDataVault2024
export DATA_DIR=/dataEhand_tmp 

# Check if the IMG environment variable is set
if [ -z "$IMG" ]; then
  echo "Error: IMG environment variable is not set."
  echo "Usage: export IMG=<image_name> and then run the script."
  exit 1
fi


echo W:myRank is $SLURM_PROCID

if [ $SLURM_PROCID -eq 0 ]; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

#hostname
#echo W:args '1='$1 '

# Run the container with a unique name
CONTAINER_NAME="cudaq_container_$SLURM_PROCID"

podman-hpc run --gpu --rm  --volume $CFSH/$BASE_DIR:/$BASE_DIR \
    --volume ${CFSH}/daan_qcrank:/daan_qcrank  \
    --volume ${DATA_VAULT}:/dataVault \
    --volume ${DATA_VAULT}/$DATA_DIR:/data_tmp  \
    -e EHands_dataVault=/dataEhand_tmp  \
    -e DISPLAY  -v $HOME:$HOME -e HOME  \
    -e HDF5_USE_FILE_LOCKING='FALSE' \
    --workdir /$BASE_DIR    $IMG $CMD


