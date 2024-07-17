#!/bin/bash
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
export IMG=gzquse/cudaquanmpi-qiskit1:p3
export CMD="python3 cuda_quantum_test/examples/python/cuquantum_backends.py"
export CFSH=/pscratch/sd/g/gzquse
export JNB_PORT=' '
export BASE_DIR=GRADIENT_IMAGE  # here git has home
export WORK_DIR=$BASE_DIR

# Check if the IMG environment variable is set
# if [ -z "$IMG" ]; then
#   echo "Error: IMG environment variable is not set."
#   echo "Usage: export IMG=<image_name> and then run the script."
#   exit 1
# fi


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
    --workdir /$BASE_DIR    $IMG $CMD
