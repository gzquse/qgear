#!/usr/bin/env python3

# mpirun -np 4 --allow-run-as-root python3 cuquantum_backends.py

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from time import time

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=8, type=int, help='pair: nq_addr nq_data, space separated ')

    parser.add_argument('-n','--numShots', default=10, type=int, help='num of shots')
    parser.add_argument('-r','--numRepeat', default=2, type=int, help='num of CX repeats')
    parser.add_argument('-t','--cudaqTarget',default="nvidia-mgpu", choices=['qpp-cpu','nvidia','nvidia-mgpu','nvidia-mqpu'], help="CudaQ target backend")

    args = parser.parse_args()
    for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

#...!...!....................
def qiskit_circ(nq: int, nRep: int):
    qc = QuantumCircuit(nq)
    qc.h(0)
    for j in range(nRep):
        for qubit in range(nq - 1):
            qc.ry(0.1, qubit)
            qc.rz(0.2, qubit)
            qc.cx(qubit, qubit + 1)

    qc.measure_all()
    return qc



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

    qc=qiskit_circ(nq, nRep)

    print(qc)
    shots=args.numShots
    backend = AerSimulator()
       

    T0=time()
    #???result=cudaq.sample(kernel,  shots_count=shots)
    elaT=time()-T0
