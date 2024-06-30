#!/bin/bash
echo W:myRank is $SLURM_PROCID
echo W:args '1= '$1 ',2= '$2  ',3= '$3 ',4= '$4

IMG=$1
CMD=$2
wrkPath=$3
absDataPath=$4
myRank=${SLURM_PROCID:-0}  # protection if srun is not used

if [ $myRank -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
fi

# it is harmless to use '--gpu' on CPU node, so same script on CPU or GPU node
podman-hpc run -i --gpu \
   --volume $absDataPath:/myData \
   --volume $wrkPath:/wrk \
   --workdir  /wrk \
   -e HDF5_USE_FILE_LOCKING='FALSE' \
   $IMG <<EOF
   echo I:started
   #nvidia-smi
   pwd
   ls -l
   ls -l /myData
   $CMD
   echo I:ended
EOF
echo W:done
#  -e HDF5_USE... fixes error message:  'Unknown error 524'
