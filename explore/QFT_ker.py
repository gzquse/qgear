#!/usr/bin/env python3
import cudaq
import numpy as np
from typing import List
# Define kernels for the Quantum Fourier Transform and the Inverse Quantum Fourier Transform
@cudaq.kernel
def quantum_fourier_transform2(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the QFT.'''
    qubit_count = len(qubits)
    # Apply Hadamard gates and controlled rotation gates.
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])

@cudaq.kernel
def inverse_qft(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the inverse QFT.'''
    cudaq.adjoint(quantum_fourier_transform2, qubits)

@cudaq.kernel
def verification_example(input_state : List[int]):
    '''Args:
    input_state (list[int]): specifies the input state to be transformed with QFT and the inverse QFT.  '''
    qubit_count = len(input_state)
    # Initialize qubits.
    qubits = cudaq.qvector(qubit_count)

    # Initialize the quantum circuit to the initial state.
    for i in range(qubit_count):
        if input_state[i] == 1:
            x(qubits[i])

    # Apply the quantum Fourier Transform
    quantum_fourier_transform2(qubits)

    # Apply the inverse quantum Fourier Transform
    inverse_qft(qubits)

# The state to which the QFT operation is applied to. The zeroth element in the list is the zeroth qubit.
input_state = [1, 0, 0]


# Number of decimal points to round up the statevector to.
precision = 2

shots = 10000
print(cudaq.draw(verification_example, input_state))
print(cudaq.sample(verification_example, input_state, shots_count=shots))
# Print the statevector to the specified precision
statevector = np.array(cudaq.get_state(verification_example, input_state))
print(np.round(statevector, precision)) # The result should be the input state