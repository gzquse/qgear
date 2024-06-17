#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Example of   writeing/reading serialized circuit using QPY format '''
import sys,os


import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from time import time
from qiskit import qpy
from qiskit.circuit import Parameter

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
    qc = QuantumCircuit(nq, nq, name='bell', metadata={'test1': True,'test2':23.45})
    qc.h(0)
    for i in range(1, nq):
        qc.cx(0, i)
    qc.measure(range(nq), range(nq))
    return qc

#...!...!....................
def rndCirc(nq=2):
    from qiskit.circuit.random import random_circuit
    qc = random_circuit(nq,nq, measure=True)
    qc.name='horse'
    return qc

#...!...!....................
def paramCirc():
    nq=2
    angle = Parameter("angle")  # undefined number
    theta = Parameter("$\Theta$")
    qc = QuantumCircuit(nq,nq, name='dog', metadata={'city':'richmond','test2':22.33})
    qc.ry(angle, 0)
    qc.h(1)
    qc.cx(1,0)
    qc.rz(theta,1)
    qc.x(0)
    qc.measure(range(nq), range(nq))
    return qc,angle,theta


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer') 
    backend = AerSimulator()

    #.... pick one 
    #qc = ghzCirc(args.numQubits)
    qc = rndCirc(args.numQubits)
    #qc,angle,theta = paramCirc()

    print('meta:',qc.metadata)
    if args.numQubits<6:  print(qc)
    qcT=transpile(qc,backend, basis_gates=['p','sx','cx','ry'])
    if args.numQubits<6:  print(qcT)
    print('M1: parameters:',qc.parameters)
    
    # . . . . . . . . . . . . . . . . . . .

    circF='out/%s.qpy'%qc.name
    with open(circF, 'wb') as fd:
        qpy.dump(qcT, fd)
    print('\nSaved circ:',circF)
        
    with open(circF, 'rb') as fd:
        qc1 = qpy.load(fd)[0]
    
    # . . . . . . . . . . . . . . . . . . .
    if args.numQubits<6:  print(qc1)
    print('meta:',qc1.metadata)
    qc1Par=qc1.parameters  # <class 'qiskit.circuit.parametertable.ParameterView'>
    print('M2: parameters:',qc1Par,type(qc1Par),'len=',len(qc1Par))
       
    if len(qc1.parameters)>0:
        qc1=qc1.assign_parameters({ angle: 1.23,theta:44.55 })  # works for many paramaters
        #1qc1=qc1.assign_parameters([11.22])  # wokrs for 1 parameter
    if args.numQubits<6:  print(qc1)
    shots=args.numShots

    # run the simulation for all images
    print('M: job nqTot=%d started ...'%qc1.num_qubits)

    T0=time()
    results = backend.run(qc1, shots=shots).result()
    counts = results.get_counts(0)
    elaT=time()-T0
    print('M: QCrank simu   shots=%d   ended elaT=%.1f sec'%(shots,elaT))
    if args.numQubits<4: 
        print('counts:',counts)
    else:
        print('counts size:',len(counts))
