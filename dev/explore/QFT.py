#!/usr/bin/env python3

import cudaq
import numpy as np
from typing import List


@cudaq.kernel
def quantum_fourier_transform(input_state: List[int]):
    '''Args:
    input_state (list[int]): specifies the input state to be Fourier transformed.  '''

    qubit_count = len(input_state)

    # Initialize qubits.
    qubits = cudaq.qvector(qubit_count)

    # Initialize the quantum circuit to the initial state.
    for i in range(qubit_count):
        if input_state[i] == 1:
            x(qubits[i])

    # Apply Hadamard gates and controlled rotation gates.
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])

#Can be changed to 'nvidia' for single gpu, 'nvidia-mgpu' for multi-GPU or quantum hardware.
cudaq.set_target("nvidia-mgpu")

# The state to which the QFT operation is applied to. The zeroth element in the list is the zeroth qubit.
input_state = [1, 0, 1]

# Number of decimal points to round up the statevector to.
precision = 2

# Draw the quantum circuit.
print(cudaq.draw(quantum_fourier_transform, input_state))

# Print the statevector to the specified precision
statevector = np.array(cudaq.get_state(quantum_fourier_transform, input_state))
print(np.round(statevector, precision))