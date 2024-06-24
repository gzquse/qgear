#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' compare time for 2 methods of assembly & execute a long random cudaq circuit '''

import sys,os
import numpy as np
from time import time
import cudaq

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=3, type=int, help='pair: nq_addr nq_data, space separated ')
   
    parser.add_argument('-n','--numShots', default=1001000, type=int, help='num of shots')
    parser.add_argument('-k','--numCX', default=4, type=int, help='num of CX gates')
    parser.add_argument('-t','--cudaqTarget',default="nvidia", choices=['qpp-cpu','nvidia','nvidia-mgpu','nvidia-mqpu'], help="CudaQ target backend")
  
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
def circ_func(N, flat_qpair,angles):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(N)
    kernel.h(q[0])
    for i in range(N - 1):
      kernel.cx(q[i], q[i + 1])

    # Applying operations based on qpair
    ng=len(angles)      
    for i in range(ng):
        j=2*i
        kernel.ry(angles[i],q[flat_qpair[j]] )
        kernel.rz(-angles[i],q[flat_qpair[j+1]] )
        kernel.cx(q[flat_qpair[j]], q[flat_qpair[j+1]])
        
    kernel.mz(q)
    return kernel


#...!...!....................
@cudaq.kernel
def circ_decor(N: int, flat_qpair: list[int], angles: list[float]):
    qvector = cudaq.qvector(N)
    h(qvector[0])
    for i in range(N - 1):
        x.ctrl(qvector[i], qvector[i+1])
        
    # Applying operations based on qpair
    ng=len(angles)      
    for i in range(ng):
        j=2*i
        ry(angles[i],qvector[flat_qpair[j]] )
        rz(-angles[i],qvector[flat_qpair[j+1]] )
        x.ctrl(qvector[flat_qpair[j]], qvector[flat_qpair[j+1]])
        
    mz(qvector)



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    cudaq.set_target(args.cudaqTarget)
    shots=args.numShots
    print('use target=%s, run %d shots'%(args.cudaqTarget,shots))
    qpairs = generate_random_pairs(args.numCX, nq)
    yangles = np.random.uniform(0, np.pi, args.numCX)
    #qpairs = [[0, 1], [2, 1], [1, 0]]
    if args.numCX<5: print(qpairs)
   
    # Flatten qpair list 
    fpairs = [int(qubit) for pair in qpairs for qubit in pair]
    fangles = [float(x) for x in yangles ]
    
    print('\nM:case: circ_decor()...')
    if nq<6: print(cudaq.draw(circ_decor, nq,fpairs, fangles))
    
    T0=time()
    counts = cudaq.sample(circ_decor, nq,fpairs, fangles, shots_count=shots)
    print('  assembled & run  elaT= %.1f sec'%(time()-T0))
    if nq<6:  counts.dump()
    str0=counts.most_probable()
    print('numSol:%d  MPV %s: %d\n'%(len(counts),str0,counts[str0]))

    #exit(0)  # skip slower version
    print('\nM:case: circ_func()...')
    T0=time()
    qKer=circ_func(nq,fpairs,fangles)
    print('  assembled  elaT= %.1f sec, run ...'%(time()-T0))
    if nq<6:  print(cudaq.draw(qKer))
    T0=time()
    counts = cudaq.sample(qKer, shots_count=shots)
    print('  run  elaT= %.1f sec'%(time()-T0))
    if nq<6:  counts.dump()
    # <class 'cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.SampleResult'>
    str0=counts.most_probable()
    print('numSol:%d  MPV %s: %d'%(len(counts),str0,counts[str0]))


