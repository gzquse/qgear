#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -x ;  # print out the statements as they are being executed
set -e ;  #  bash exits if any statement returns a non-true return value

# runs inside IMAGE

basePath=/dataVault2024/dataCudaQ_QEra_July12

if [ ! -d "$basePath" ]; then
    echo create $basePath
    mkdir -p $basePath
    cd $basePath
    mkdir circ meas post 
    cd -
fi

nCX=(100 10000 20000)  # list of cx-gates
nCirc=8 # num of circuits

# prefix
N="mar"
# ......  run jobs .......
k=0

for nq in {28..34}; do
    for cx in ${nCX[@]}; do
        expN=$N${nq}q${cx}cx
        k=$[ $k +1 ]
        echo $k  expN: $expN
        ./gen_gateList.py -k $cx -i $nCirc  --expName  $expN  -q $nq  --basePath ${basePath}
    done
done
echo done $k jobs
date

# CMD
# nq=18 ; ./gen_gateList.py -k 10000 -i 4  --expName  mar${nq}q  -q $nq  --basePath ${basePath} ; exit
#  ./run_gateList.py  --expName cb18q  -b qiskit-cpu --basePath ${basePath} ; exit