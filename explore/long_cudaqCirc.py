#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' compare time to assembly long cudaq circuit '''

import sys,os
import numpy as np
from time import time
import cudaq

import argparse
#...!...!..................
def get_parser(backName=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=3, type=int, help='pair: nq_addr nq_data, space separated ')
   
    parser.add_argument('-n','--numShots', default=1000, type=int, help='num of shots')
    parser.add_argument('-k','--numCX', default=4, type=int, help='num of CX gates')
    
  
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

#...!...!....................
def generate_random_pairs(k, nq):
    pairs = [[]]*k
    for i in range(k):
        while True:
            pair = np.random.choice(nq, 2, replace=False)
            if pair[0] == pair[1]: continue  # Ensuring the pair values are different
            pairs[i]=pair
            break
    return np.array(pairs)

#...!...!....................
def circ_extend(N, flat_qpair):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(N)
    kernel.h(q[0])
    for i in range(N - 1):
      kernel.cx(q[i], q[i + 1])

    # Applying operations based on qpair
    ng=len(flat_qpair)      
    for i in range(0, ng, 2):
        #print('rrr',i,flat_qpair[i], flat_qpair[i+1])
        kernel.cx(q[flat_qpair[i]], q[flat_qpair[i+1]])

    kernel.mz(q)
    return kernel


#...!...!....................
@cudaq.kernel
def circ_instance(N: int, flat_qpair: list[int]):
    qvector = cudaq.qvector(N)
    h(qvector[0])
    for i in range(1, N):
        x.ctrl(qvector[0], qvector[i])
        
    # Applying operations based on qpair
    ng=len(flat_qpair)      
    for i in range(0, ng, 2):
        x.ctrl(qvector[flat_qpair[i]], qvector[flat_qpair[i+1]])
    
    mz(qvector)



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer_simulator_statevector')  #     assert len(args.padSymbols)==2
    nq=args.numQubits
    cudaq.set_target("nvidia")
    shots=args.numShots
    print('got GPU, run %d shots'%shots)
    qpairs = generate_random_pairs(args.numCX, nq)
    #qpairs = [[0, 1], [2, 1], [1, 0]]
    if args.numCX<5: print(qpairs)
   
    # Flatten qpair list 
    fpairs = [int(qubit) for pair in qpairs for qubit in pair]

    
    print('\nM:run instance...')
    if nq<6: print(cudaq.draw(circ_instance, nq,fpairs))
   
    T0=time()
    result = cudaq.sample(circ_instance, nq,fpairs, shots_count=shots)
    print('  run done elaT=%.1f sec'%(time()-T0))
    if nq<6: print(result)
    

    print('\nM:run extend...')
    T0=time()
    qKer=circ_extend(nq,fpairs)   
    if nq<6:  print(cudaq.draw(qKer))
    counts = cudaq.sample(qKer, shots_count=shots)
    print('  run done elaT=%.1f sec'%(time()-T0))
    if nq<6: counts.dump()
   

