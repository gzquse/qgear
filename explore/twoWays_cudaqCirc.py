#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' compare 2 methods of assembly cudaq circuit '''

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
   
    parser.add_argument('-n','--numShots', default=1000, type=int, help='num of shots')
    parser.add_argument('-t','--cudaqTarget',default="nvidia", choices=['qpp-cpu','nvidia','nvidia-mgpu','nvidia-mqpu'], help="CudaQ target backend")
    
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

#...!...!....................
def ghz_extend(N):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(N)
    kernel.h(q[0])
    for i in range(N - 1):
      kernel.cx(q[i], q[i + 1])
 
    kernel.mz(q)
    return kernel


#...!...!....................
@cudaq.kernel
def ghz_instance(N: int):
    qvector = cudaq.qvector(N)
    h(qvector[0])
    for i in range(1, N):
        x.ctrl(qvector[0], qvector[i])
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
    print('got %s, run %d shots'%(cudaq.get_target(),shots))

    print('\nM:run instance...')
    if nq<6: print(cudaq.draw(ghz_instance, nq))
    T0=time()
    result = cudaq.sample(ghz_instance, nq, shots_count=shots)
    print('  run done elaT=%.1f sec'%(time()-T0))
    if nq<6: print(result)


    print('\nM:run extend...')
    T0=time()
    qKer=ghz_extend(nq)   
    if nq<6:  print(cudaq.draw(qKer))
    counts = cudaq.sample(qKer, shots_count=shots)
    print('  run done elaT=%.1f sec'%(time()-T0))
    if nq<6: counts.dump()
   
