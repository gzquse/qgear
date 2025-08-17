import cudaq
import numpy as np
from typing import List
import random 
from time import time

wires = 28
cudaq.set_target("nvidia", option='mgpu,fp32')

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

# The state to which the QFT operation is applied to. The zeroth element in the list is the zeroth qubit.
input_state = [random.choice([0, 1]) for i in range(wires)]
T0=time()
# Print the statevector to the specified precision
statevector = np.array(cudaq.get_state(quantum_fourier_transform, input_state))
print(time()-T0)
