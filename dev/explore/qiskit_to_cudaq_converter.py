import numpy as np
from qiskit import QuantumCircuit

def extract_qiskit_info(qc: QuantumCircuit, filename: str):
    qregs = qc.qregs[0]
    nq = qc.num_qubits

    # Construct a dictionary with address:index of the qregs objects
    qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}

    # Lists to store gate information
    flat_qpair = []
    angles = []

    # Extract Qiskit operations information
    for op in qc:
        gate = op.operation.name
        params = op.operation.params if op.operation.params else []
        qAddrL = [hex(id(q)) for q in op.qubits]
        qIdxL = [qregAddrD[a] for a in qAddrL]
        
        # Flatten qIdxL for pairs
        if gate in ['cx']:
            flat_qpair.extend(qIdxL)
        elif gate in ['ry', 'rz']:
            angles.extend(params)

    # Save the extracted information to a .npy file
    np.save(filename, {'flat_qpair': flat_qpair, 'angles': angles})

# Example usage
# qc = ...  # Define or load your Qiskit QuantumCircuit here
# extract_qiskit_info(qc, 'qiskit_info.npy')
