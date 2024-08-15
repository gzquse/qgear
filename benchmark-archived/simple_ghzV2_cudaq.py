#!/usr/bin/env python3

# mpirun -np 4 --allow-run-as-root python3 cuquantum_backends.py

import cudaq
from time import time

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=35, type=int, help='pair: nq_addr nq_data, space separated ')

    parser.add_argument('-n','--numShots', default=10000, type=int, help='num of shots')
    parser.add_argument('-r','--numRepeat', default=10, type=int, help='num of CX repeats')
    parser.add_argument('-t','--cudaqTarget',default="nvidia-mgpu", choices=['qpp-cpu','nvidia','nvidia-mgpu','nvidia-mqpu'], help="CudaQ target backend")

    args = parser.parse_args()
    #cudaq.mpi.initialize()
    # myrank = cudaq.mpi.rank()
    # if myrank==0:
    #     for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

#...!...!....................
@cudaq.kernel
def kernel(qubit_count: int, nRep: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    for j in range(nRep):
        for qubit in range(qubit_count - 1):
            ry(0.1, qvector[qubit])
            rz(0.2, qvector[qubit+1])
            x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    target=args.cudaqTarget
    nRep=args.numRepeat
    
    # ranks = cudaq.mpi.num_ranks()
    # myrank = cudaq.mpi.rank()

    cudaq.set_target(target)
    shots=args.numShots
    #? print(cudaq.draw(kernel, nq, nRep)) 
    
    # if myrank == 0:
    #     print('\n start nq=%d, target=%s, ranks=%d, aval GPUs=%d nRep=%d shots=%d ... '%(nq,target,ranks,cudaq.num_available_gpus(),nRep,shots))
        
    if cudaq.num_available_gpus() == 0:
        print("This example requires a GPU to run. No GPU detected.")
        exit(0)

    T0=time()
    result=cudaq.sample(kernel, nq, nRep, shots_count=shots)
    elaT=time()-T0

    # if myrank == 0:
    #     if args.verb>1: print('counts:',result)
    #     print('\nelaT= %.1f sec , nq=%d, nRep=%d, shots=%d, target=%s, ranks=%d, numSol=%d '%(elaT,nq,nRep,shots,target,ranks,len(result)))
    
# cudaq.mpi.finalize()
