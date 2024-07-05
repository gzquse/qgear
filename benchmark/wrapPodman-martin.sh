#!/bin/bash
echo W:myRank is $SLURM_PROCID `hostname`  localRank=$SLURM_LOCALID
echo W:args '1= '$1 ', 2= '$2  ', 3= '$3 ', 4= '$4

IMG=$1
CMD=$2
wrkPath=$3
absDataPath=$4
myRank=${SLURM_PROCID:-0}  # protection if srun is not used
 
if [ $myRank -eq 0 ] ; then
    echo W: worldSize=${SLURM_NTASKS:-1}
    echo W:IMG=$IMG 
    echo W:CMD=$CMD
    echo W:`env|grep PODMANHPC`
fi

tSleep=$(( SLURM_LOCALID * 2))
echo W: rank=$myRank tSlep=$tSleep  
sleep $tSleep
# enable mpi plugin in cuda-quantum
ENABLE_MPI="source ./distributed/activate_custom_mpi.sh && exec bash"

# it is harmless to use '--gpu' on CPU node, so same script on CPU or GPU node
echo podman-hpc run --privileged -it --gpu \
   --volume $absDataPath:/myData \
   --volume $wrkPath:/wrk \
   -e HDF5_USE_FILE_LOCKING='FALSE' \
   -e SLURM_NTASKS=${SLURM_NTASKS:-0} -e SLURM_PROCID=${SLURM_PROCID-1} \
   -e OMPI_ALLOW_RUN_AS_ROOT=1 \
   -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
   -e UCX_WARN_UNUSED_ENV_VARS=n \
   --workdir /wrk \
   $IMG bash -c "\"$ENABLE_MPI\""

podman-hpc run --privileged -it --gpu \
   --volume $absDataPath:/myData \
   --volume $wrkPath:/wrk \
   -e HDF5_USE_FILE_LOCKING='FALSE' \
   -e SLURM_NTASKS=${SLURM_NTASKS:-0} -e SLURM_PROCID=${SLURM_PROCID-1} \
   -e OMPI_ALLOW_RUN_AS_ROOT=1 \
   -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
   -e UCX_WARN_UNUSED_ENV_VARS=n \
   --workdir /wrk \
   $IMG bash -c "$ENABLE_MPI" <<EOF
   echo I:started `date`
   $CMD
EOF


echo W:done
#  -e HDF5_USE... fixes error message:  'Unknown error 524'
