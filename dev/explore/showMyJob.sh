#!/bin/bash

# Set the priority threshold
priority_threshold=67799  # Change this value to your desired threshold

# List jobs with detailed information and filter by priority
squeue --qos=gpu_shared  -o "%.7i %.10P %.12j %.8u %.2t %.10M %.6D %.10Q %.20N" | nl

#awk -v threshold=$priority_threshold '$8 > threshold {print $0}'
