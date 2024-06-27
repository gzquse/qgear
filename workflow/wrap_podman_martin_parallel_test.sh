#!/bin/bash

IMG=gzquse/cudaquanmpi-qiskit1:p2.1

# Check if the IMG environment variable is set
if [ -z "$IMG" ]; then
  echo "Error: IMG environment variable is not set."
  echo "Usage: export IMG=<image_name> and then run the script."
  exit 1
fi

CMD=python3 cuda_quantum_test/examples/python/cuquantum_backends.py --myRank $SLURM_PROCID
echo W:myRank is $SLURM_PROCID

if [ $SLURM_PROCID -eq 0 ]; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

#hostname
#echo W:args '1='$1 '
# Run the container with a unique name
CONTAINER_NAME="cudaq_container_$SLURM_PROCID"

podman-hpc run --name $CONTAINER_NAME -d $IMG $CMD

# Wait for the container to finish
podman-hpc wait $CONTAINER_NAME

# Remove the container
podman-hpc rm $CONTAINER_NAME
