#!/bin/bash

set -u  # exit if you try to use an uninitialized variable
# set -e  # bash exits if any statement returns a non-true return value

export CUDAQ_MPI_COMM_LIB=pscratch/sd/g/gzquse/GRADIENT_IMAGE/benchmark/cudaq/libcudaq_distributed_interface_mpi.so
export LD_LIBRARY_PATH=/pscratch/sd/g/gzquse/GRADIENT_IMAGE/benchmark/cudaq:$LD_LIBRARY_PATH

python $1


# srun -N 2 -n 8 shifter bash -l run_single.sh run_cudaq_qft.py