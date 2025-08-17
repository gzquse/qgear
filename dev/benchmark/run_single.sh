#!/bin/bash

# set -u  # exit if you try to use an uninitialized variable
# set -e  # bash exits if any statement returns a non-true return value

# export CUDAQ_MPI_COMM_LIB=pscratch/sd/g/gzquse/GRADIENT_IMAGE/benchmark/cudaq/libcudaq_distributed_interface_mpi.so
# export LD_LIBRARY_PATH=/pscratch/sd/g/gzquse/GRADIENT_IMAGE/benchmark/cudaq:$LD_LIBRARY_PATH


module load conda
conda activate $SCRATCH/cudaq
python -u ./run_gateList.py --expName mar28q100cx --backend nvidia --numShots 1000 --basePath /pscratch/sd/g/gzquse/quantDataVault2025/dataCudaQ_Aug16 --qft 0 --target-option fp32

conda deactivate
# srun -N 2 -n 8 shifter bash -l run_single.sh run_cudaq_qft.py