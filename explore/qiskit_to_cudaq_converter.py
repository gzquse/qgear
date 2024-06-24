import numpy as np

def extract_qiskit_info(qc, filename):
    qregs = qc.qregs[0]
    nq = qc.num_qubits

    # Construct a dictionary with address:index of the qregs objects
    qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}

    # Lists to store gate information
    gate_info = []

    # Extract Qiskit operations information
    for op in qc:
        gate = op.operation.name
        params = op.operation.params
        qAddrL = [hex(id(q)) for q in op.qubits]
        qIdxL = [qregAddrD[a] for a in qAddrL]
        gate_info.append((gate, qIdxL, params))
    
    # Save the extracted information to a .npy file
    np.save(filename, gate_info