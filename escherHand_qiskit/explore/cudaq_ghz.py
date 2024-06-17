#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Qiskit to CUDA quantum single gpu node'''
import sys,os

import cudaq
from qiskit import qpy
#=================================
#=================================
#  M A I N 
#=================================
#=================================

def qiskit_to_cudaq(file_name, shots):
    
    with open(file_name, 'rb') as handle:
        qc = qpy.load(handle)
    
    quantum_data = qc[0]
    qubit_count = quantum_data.num_qubits
    cudaq.set_target("nvidia")

    kernel = cudaq.make_kernel()

    qubits = kernel.qalloc(qubit_count)
    # Translate Qiskit operations to CUDAQ

    for instruction in quantum_data:
        gate = instruction.operation.name
        qubit_indices = [q._index for q in instruction.qubits]
        if gate == 'h':
            kernel.h(qubits[qubit_indices[0]])
        elif gate == 'cx':
            kernel.cx(qubits[qubit_indices[0]], qubits[qubit_indices[1]])
        elif gate == 'measure':
            kernel.mz(qubits[qubit_indices[0]])
        else:
            print('ABORT; uknown gate',gate); exit(99)


    results = cudaq.sample(kernel, shots_count=shots)
    if qubit_count < 6:
        print(cudaq.draw(kernel))
        print(results)
    else:
        print("count:", len(results))
        
if __name__ == "__main__":
    qiskit_to_cudaq("ghz.qpy", 1000)
    #qiskit_to_cudaq("out/bell.qpy", 1000)
    #qiskit_to_cudaq("out/horse.qpy", 1000)
    
