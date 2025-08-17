#!/bin/bash
module load cudatoolkit
module load conda
conda activate /pscratch/sd/g/gzquse/cudaq
exec "$@"