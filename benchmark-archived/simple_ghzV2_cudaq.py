#!/usr/bin/env python3

# mpirun -np 4 --allow-run-as-root python3 cuquantum_backends.py
# mpirun -np 4 python3 ./simple_ghzV2_cudaq.py --numShots 10 --cudaqTarget nvidia-mgpu --numRepeat 10 --numQubits 33

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
    cudaq.set_target(args.cudaqTarget)
    cudaq.mpi.initialize()
    args.myrank = cudaq.mpi.rank()
    args.ranks=cudaq.mpi.num_ranks()
    if args.myrank==0:
         for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))

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
    nCX=nRep*(nq-1)
    
    shots=args.numShots
    #? print(cudaq.draw(kernel, nq, nRep)) 

    if args.myrank==0:
        print('\n start nq=%d, target=%s, ranks=%d, aval GPUs=%d nRep=%d,  nCX=%d shots=%d ... '%(nq,target,args.ranks,cudaq.num_available_gpus(),nRep,nCX,shots))
        
    if cudaq.num_available_gpus() == 0:
        print("This example requires a GPU to run. No GPU detected.")
        exit(0)

    T0=time()
    result=cudaq.sample(kernel, nq, nRep, shots_count=shots)
    elaT=time()-T0

    if args.myrank==0:
        if args.verb>1: print('counts:',result)
        print('\nelaT= %.1f sec , nq=%d, nRep=%d,  nCX=%d  shots=%d, target=%s, ranks=%d, numSol=%d '%(elaT,nq,nRep,nCX,shots,target,args.ranks,len(result)))
    
    cudaq.mpi.finalize()
