#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value

# runs inside IMAGE

basePath=/quantDataVault2024/dataCudaQ_test3

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir -p $basePath
    cd $basePath
    mkdir circ  meas post 
    cd -
fi

# testing
# nq=18 ; ./gen_gateList.py -k 10000 -i 4  --expName  mar${nq}q  -q $nq  --basePath ${basePath} ; exit
#  ./run_gateList.py  --expName cb18q  -b qiskit-cpu --basePath ${basePath} ; exit

nCX=101000  # num cx-gates
nCirc=4

# ......  run jobs .......
k=0

for nq in {18..21}; do
#for nq in {16..18}; do  # for testing
    expN=mar${nq}q
    k=$[ $k +1 ]
    echo $k  expN: $expN
    ./gen_gateList.py -k $nCX -i $nCirc  --expName  $expN  -q $nq  --basePath ${basePath}

done
echo done $k jobs
date

