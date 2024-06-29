#!/bin/bash

# Ensure MPI is set up correctly

# Run the MPI program with the desired number of processes
mpirun -np 2 --bind-to none python3 cuquantum_backends.py -target nvidia-mgpu

