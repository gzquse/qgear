import cudaq

#...!...!....................
def qiskit_to_cudaq(qc):
    qregs=qc.qregs[0]
    #print(qregs)
    # Construct a dictionary with address:index of the qregs objects
    qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}
    #print('qregAddrD:',qregAddrD)

    nq = qc.num_qubits
    # single gpu
    cudaq.set_target("nvidia")
    gpu_counts = 1
    # multiple gpus TODO

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(nq)
    #print('CCC, nq=',nq,qubits)
    # Translate Qiskit operations to CUDAQ
    for op in qc:
        #print('\nop',op)
        gate = op.operation.name
        params = op.operation.params
        qAddrL = [hex(id(q)) for q in op.qubits]
        qIdxL=[ qregAddrD[a] for a in qAddrL]
        #print('gg',gate)
        if gate == 'h':
            kernel.h(qubits[qIdxL[0]])
        elif gate == 'cx':
            kernel.cx(qubits[qIdxL[0]], qubits[qIdxL[1]])
        elif gate == 'ry':
            kernel.ry(params[0], qubits[qIdxL[0]])
        elif gate == 'rz':
            kernel.rz(params[0], qubits[qIdxL[0]])
        elif gate == 'measure':
            kernel.mz(qubits[qIdxL[0]])
        elif gate == 'barrier':
            continue
            #print('MISSING : kernel.barrier')
        else:
            print('ABORT; uknown gate',gate); exit(99)

    return kernel
