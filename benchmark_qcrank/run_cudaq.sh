#!/bin/bash
#SBATCH -N 4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest
#SBATCH --module=cuda-mpich
#SBATCH --output=out/%j.out
#SBATCH --licenses=scratch

export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so

# Read tags from config file, skipping comments and empty lines
config_file="run_config.txt"
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found!"
    exit 1
fi

# Process each tag from config
while read -r tag || [ -n "$tag" ]; do
    # Skip comments and empty lines
    [[ $tag =~ ^#.*$ || -z $tag ]] && continue
    
    # Construct the circuit name based on the tag
    case $tag in
        "a1") circName="canImg_${tag}_32_1" ;;
        "b1") circName="canImg_${tag}_64_16" ;;
        "b2") circName="canImg_${tag}_32_32" ;;
        "c1") circName="canImg_${tag}_64_80" ;;
        "d1") circName="canImg_${tag}_192_128" ;;
        "d2") circName="canImg_${tag}_192_128" ;;
        "d3") circName="canImg_${tag}_128_192" ;;
        "d4") circName="canImg_${tag}_192_128" ;;
        "d5") circName="canImg_${tag}_192_128" ;;
        "e1") circName="canImg_${tag}_384_256" ;;
        "e2") circName="canImg_${tag}_256_384" ;;
        "e3") circName="canImg_${tag}_384_256" ;;
        *)  echo "Warning: Unknown tag $tag"
            continue ;;
    esac
    
    echo "Processing $circName"
    CMD="./run_cudaq_job.py --circName canImg_e1_384_256 -n 300"
    srun -N 4 -n 16 shifter bash -l launch.sh ./run_cudaq_job.py -c canImg_b2_32_32 -n 300
done < "$config_file"