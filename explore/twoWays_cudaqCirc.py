#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' compare 2 methods of assembly cudaq circuit '''

import sys,os
import numpy as np
from time import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
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

def ghz_qiskit(nq):
    qr = QuantumRegister(nq, 'q')
    cr = ClassicalRegister(nq, 'c')
    qc = QuantumCircuit(qr,cr)
    qc.h(qr[0])
    for i in range(1, nq):
        qc.cx(qr[0], qr[i])
    qc.barrier()
    for i in range(nq):
        qc.measure(qr[i],cr[i])
    return qc


#...!...!....................
def ghz_obj(nq):
    qc= cudaq.make_kernel()
    qr = qc.qalloc(nq)
    qc.h(qr[0])
    for i in range(1, nq):
        qc.cx(qr[0], qr[i])
    qc.mz(qr)
    return qc


#...!...!....................
@cudaq.kernel
def ghz_kernel(N: int):
    qr = cudaq.qvector(N)
    h(qr[0])
    for i in range(1, N):
        x.ctrl(qr[0], qr[i])
    mz(qr)



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    shots=args.numShots

    if nq<6:        # show qiskit for reference
        qc1=ghz_qiskit(nq)
        print(qc1)
        backend = AerSimulator()
        T0=time()
        results = backend.run(qc1, shots=shots).result()
        counts = results.get_counts(0)
        elaT=time()-T0
        print('  run qiskit_obj done elaT=%.1f sec'%(time()-T0))    
        print(counts)
     
    #... do the same in cudaq
    cudaq.set_target(args.cudaqTarget)
    
    print('\ngot %s, run %d shots'%(cudaq.get_target(),shots))

    print('\nM:run circ-cudaq_obj...')
    T0=time()
    qc2=ghz_obj(nq)
   
    if nq<6:  print(cudaq.draw(qc2))
    counts = cudaq.sample(qc2, shots_count=shots)
    print('  run cudaq_obj done elaT=%.1f sec, target=%s'%(time()-T0,args.cudaqTarget))
    counts.dump()      
 
    print('\nM:run circ-cudaq_kernel...')
    if nq<6: print(cudaq.draw(ghz_kernel, nq))
    T0=time()
    result = cudaq.sample(ghz_kernel, nq, shots_count=shots)
    print('  run done elaT=%.1f sec, target=%s'%(time()-T0,args.cudaqTarget))
    print(result)


