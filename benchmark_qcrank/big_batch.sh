#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

trg=gpu
c=64  # cores for CPU
n=4   # ntasks per node
ACCT=nintern
shots=400

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
    SCMD="./run_job.slr $circName $trg $shots"

    if [ "$trg" == "cpu" ]; then
        sbatch -N1 -A $ACCT $SCMD
    else
        sbatch -C "gpu&hbm80g" -N1 -A $ACCT $SCMD # currently only one node
    fi
    sleep 1
    
done < "$config_file"
