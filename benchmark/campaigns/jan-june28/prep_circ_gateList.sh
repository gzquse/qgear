#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value

# runs inside IMAGE

basePath=/dataVault2024/dataCudaQ_june28

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir -p $basePath
    cd $basePath
    mkdir circ  meas post 
    cd -
fi

nCX=101000  # num cx-gates
nCirc=2

# ......  run jobs .......
k=0

for nq in {22..33}; do
#for nq in {16..18}; do  # for testing
    expN=ck${nq}q
    k=$[ $k +1 ]
    echo $k  expN: $expN
    ./gen_gateList.py -k $nCX -i $nCirc  --expName  $expN  -q $nq  --basePath ${basePath}

done
echo done $k jobs
date
