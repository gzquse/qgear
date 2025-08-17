#!/usr/bin/env python3

# inside podman-hpc , maxNQ=33 for -N1
# salloc -N 1 --gpus-per-task=1 --ntasks-per-node=1   -t 4:00:00 -q interactive -A nintern -C gpu  --gpu-bind=none
# mpirun -np 4 python3   RND_ker1.py -q 33 -r 10 

# takes about 110 sec, inside Shifter
# srun -N 1 -n 4 shifter bash -l launch.sh " RND_ker1.py -q -31 -r 50 "

import cudaq
from time import time

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=-31, type=int, help=' negative: bits in numRank - value')

    parser.add_argument('-n','--numShots', default=10000, type=int, help='num of shots')
    parser.add_argument('-r','--numRepeat', default=50, type=int, help='num of CX repeats')

    args = parser.parse_args()
    # util functions:
    is_power_of_two = lambda n: n > 0 and (n & (n - 1)) == 0
    bits_needed = lambda x: x.bit_length() if x != 0 else 1
    
    if 0:  # in shfter
        args.gpuOpt="mgpu,fp32"
        cudaq.set_target("nvidia", option=args.gpuOpt)        
    else: # in podman-hpc
        cudaq.set_target("nvidia-mgpu")
        cudaq.mpi.initialize() 
    args.myRank = cudaq.mpi.rank()
    args.numRanks = cudaq.mpi.num_ranks()
    assert is_power_of_two(args.numRanks)
    #print('qqq',args.numQubits)
    if args.numQubits<0:
        args.numQubits=bits_needed(args.numRanks) - args.numQubits
    
    if args.myRank==0:
        for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

    
#...!...!....................
@cudaq.kernel
def pseudoRnd_circ(qubit_count: int, nRep: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    # Use poor man pseudo-random generator based on  prime numbers
    prime1 = 7919  # A larger prime number
    prime2 = 104729  # Another larger prime number
    M = qubit_count-1  # To ensure random values are within the range of qubits

    for j in range(nRep):
        for i in range(M):
            iq1= (i * prime1 + prime2) % M
            #iq1=i
            iq2=(iq1+1) %M
            ry(0.1*i, qvector[iq1])
            rz(0.2*i, qvector[iq2])
            x.ctrl(qvector[iq1], qvector[iq2])
    mz(qvector)
    

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    nRep=args.numRepeat
    shots=args.numShots
    nCX=nRep*(nq-1)
    
    if args.myRank==0 :
        print('adj-gpu RND nq=%d  nCX=%d  shots=%d  numRank=%d ...'%(nq,nCX,shots,args.numRanks))
        if nCX<20:
            print(cudaq.draw(pseudoRnd_circ, nq, nRep))


    t = time()
    counts =cudaq.sample(pseudoRnd_circ, nq, nRep,shots_count=shots)
    t2 = time() - t
    cudaq.mpi.finalize()

    if args.myRank==0:
        #counts.dump()
        num_sol=len(counts)
        
        print('adj-gpu RND nq=%d  nCX=%d  shots=%d  numRank=%d  num_sol=%d  elaT= %.1f sec'%(nq,nCX,shots,args.numRanks,num_sol, t2))
        for i,res in enumerate(counts):
            print('sol=%d %s'%(i,res))
            if i>4: break
        print('M0:done')
         
