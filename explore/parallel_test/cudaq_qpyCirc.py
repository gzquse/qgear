#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 Qiskit to CUDA quantum converter
run on single gpu node
'''
import sys,os

import cudaq
from qiskit import qpy
import numpy as np
from time import time

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-f',"--qpyName",  default='out/bell.qpy',help='full path to QPY circuit file')
   
    parser.add_argument('-n','--numShots', default=800, type=int, help='num of shots')
    
  
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    return args

#...!...!..................
def qiskit_to_cudaq(file_name): 
    with open(file_name, 'rb') as handle:
        qc = qpy.load(handle)[0]
      
    qubit_count = qc.num_qubits
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)
    
    # Translate Qiskit operations to CUDAQ
    for op in qc:
        gate = op.operation.name
        params = op.operation.params
        qIdxL = [q._index for q in op.qubits]
        if gate == 'h':
            kernel.h(qubits[qIdxL[0]])
        elif gate == 'p':
            kernel.rz(params[0], qubits[qIdxL[0]])
        elif gate == 'ry':
            kernel.ry(params[0], qubits[qIdxL[0]])
        elif gate == 'sx':
            kernel.rx(np.pi/2., qubits[qIdxL[0]])
        elif gate == 'cx':
            kernel.cx(qubits[qIdxL[0]], qubits[qIdxL[1]])
        elif gate == 'measure':
            kernel.mz(qubits[qIdxL[0]])
        else:
            print('ABORT; uknown gate',gate); exit(99)

    return qc,kernel,qubit_count 
   

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
   
    qc,qKer,nq=qiskit_to_cudaq(args.qpyName)

    if nq < 6:
        print(qc)
        print(cudaq.draw(qKer))

    print('M: run cudaq-circuit  on GPU, shots=%d'%args.numShots)
    cudaq.set_target("nvidia")
    T0=time()
    results = cudaq.sample(qKer, shots_count=args.numShots)
    elaT=time()-T0
    print('M:  ended elaT=%.1f sec'%(elaT))
    if nq < 6:
        print(results)
    else:
        print("count:", len(results))
    
