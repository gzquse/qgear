#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Example of GHZ circuit simulated on Aer'''
import sys,os


import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import time

import argparse
#...!...!..................
def get_parser(backName=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=3, type=int, help='pair: nq_addr nq_data, space separated ')

   
    parser.add_argument('-n','--numShots', default=800, type=int, help='num of shots')
    
  
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

#...!...!....................
def ghzCirc(nq=2):
    """Create a GHZ state preparation circuit."""
    qc = QuantumCircuit(nq, nq)
    qc.h(0)
    for i in range(1, nq):
        qc.cx(0, i)
    qc.measure(range(nq), range(nq))
    return qc

#...!...!....................
def rndCirc(nq=2):
    from qiskit.circuit.random import random_circuit
    qc = random_circuit(nq,nq, measure=True)
    return qc
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer')  #     assert len(args.padSymbols)==2

    #qc = ghzCirc(args.numQubits)
    qc = rndCirc(args.numQubits)

    if args.numQubits<6:  print(qc)

    backend = AerSimulator()
    shots=args.numShots
    
    # run the simulation for all images
    print('M: job nqTot=%d started ...'%qc.num_qubits)
    qcT=transpile(qc,backend)
    T0=time()
    results = backend.run(qcT, shots=shots).result()
    counts = results.get_counts(0)
    elaT=time()-T0
    print('M: QCrank simu   shots=%d   ended elaT=%.1f sec'%(shots,elaT))
    if args.numQubits<4: 
        print('counts:',counts)
    else:
        print('counts size:',len(counts))
