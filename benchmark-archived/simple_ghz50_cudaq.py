#!/usr/bin/env python3

# This example is meant to demonstrate the cuQuantum
# GPU-accelerated backends and their ability to easily handle
# a larger number of qubits compared the CPU-only backend.
#
# This will take a noticeably longer time to execute on
# CPU-only backends.
# mpirun -np 4 --allow-run-as-root python3 cuquantum_backends.py
import cudaq
from time import time

# We can set a larger `qubit_count` if running on a GPU backend.
qubit_count = 35
target="nvidia-mgpu"
cudaq.mpi.initialize()

#cudaq.set_target("nvidia-mqpu")  # parallel-GPU
cudaq.set_target(target)  # adj-GPU
ranks = cudaq.mpi.num_ranks()
myrank = cudaq.mpi.rank()

if myrank == 0:
    print('\n start nq=%d, target=%s  ranks=%d... '%(qubit_count,target,ranks))
    
if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")
    exit(0)


@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    for j in range(50):
        for qubit in range(qubit_count - 1):
            #ry(0.1, qvector[qubit])
            #rz(0.2, qvector[qubit+1])
            x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)

T0=time()
# for q in qubit_count:     
    
result=cudaq.sample(kernel, qubit_count, shots_count=10)
elaT=time()-T0

# print("num of gpus:", cudaq.num_available_gpus())
# print('elaT=%.1fsec , qubit_count=%d, num output=%d '%(elaT,qubit_count,len(result)))
if myrank == 0:
    #1print('counts:',result)
    print('\nelaT=%.1fsec , nq=%d, target=%s, ranks=%d, num output=%d '%(elaT,qubit_count,target,ranks,len(result)))
    
cudaq.mpi.finalize()
