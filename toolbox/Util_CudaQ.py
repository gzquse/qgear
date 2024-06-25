import cudaq
import traceback
import numpy as np

# Function to reverse the keys
def reverse_key(key):
    return key[::-1]

def process_dict(d):
    mapped_dict = {}
    # Reverse the keys in the initial dictionary
    reversed_dict = {reverse_key(k): v for k, v in d.items()}
    # Map the reversed keys to the values in the mapped_dict
    mapped_reversed_dict = {k: mapped_dict[k] for k in reversed_dict if k in mapped_dict}
    return mapped_reversed_dict

def string_to_dict(raw_string):
    # Convert raw string to dictionary
    raw_string = ''.join(filter(lambda x: x.isdigit() or x == ':' or x.isspace(), raw_string))
    #print("raw_string", raw_string)
    raw_list = raw_string.split()
    #print("raw_list:", raw_list)
    mapped_dict = {}
    for item in raw_list:
        key, value = item.split(":")
        mapped_dict[key] = int(value)

    # Create a new dictionary with reversed keys
    rev = {reverse_key(k): v for k, v in mapped_dict.items()}
    return rev

def counts_cudaq_to_qiskit(resL): # input cudaq results
    #... format cudaq counts to qiskit version
    probsBL=[0]*len(resL) # prime the list
    for i,res in enumerate(resL):
        buf = ""
        buf = res.__str__()
        probsBL[i] = string_to_dict(buf)
    return probsBL

#...!...!....................
def qiskit_to_cudaq(qc):
    qregs = qc.qregs[0]
    nq = qc.num_qubits

    # Construct a dictionary with address:index of the qregs objects
    qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}

    # Create CUDAQ kernel and allocate qubits
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(nq)

    # Define a mapping of Qiskit gate names to CUDAQ kernel methods
    gate_map = {
        'h': lambda qubits, qIdxL, params: kernel.h(qubits[qIdxL[0]]),
        'cx': lambda qubits, qIdxL, params: kernel.cx(qubits[qIdxL[0]], qubits[qIdxL[1]]),
        'ry': lambda qubits, qIdxL, params: kernel.ry(params[0], qubits[qIdxL[0]]),
        'rz': lambda qubits, qIdxL, params: kernel.rz(params[0], qubits[qIdxL[0]]),
        'measure': lambda qubits, qIdxL, params: kernel.mz(qubits[qIdxL[0]])
    }

    # Translate Qiskit operations to CUDAQ
    for op in qc:
        gate = op.operation.name
        params = op.operation.params
        qAddrL = [hex(id(q)) for q in op.qubits]
        qIdxL = [qregAddrD[a] for a in qAddrL]

        if gate in gate_map:
            gate_map[gate](qubits, qIdxL, params)
        elif gate == 'barrier':
            continue
        else:
            print('ABORT; unknown gate', gate)
            exit(99)
    
    return kernel

def cudaq_run_parallel_qpu(qKerL, shots, qpu_count):
    count_futures = {kernel: [] for kernel in qKerL} 
    # Distribute kernels across available GPUs
    for i, kernel in enumerate(qKerL):
        gpu_id = i % qpu_count
        count_futures[kernel].append(cudaq.sample_async(kernel, shots_count=shots, qpu_id=gpu_id))
    # Retrieve and print results
    return [counts.get() for futures in count_futures.values() for counts in futures]

def cudaq_run(qKerL, shots):
    return [cudaq.sample(kernel, shots_count=shots) for kernel in qKerL]

#...!...!....................
@cudaq.kernel
def circ_kernel(qubit_count: int, gate_len: int, gate_type: list[int], flat_qpair: list[int], angles: list[float]):
    qvector = cudaq.qvector(qubit_count)

    for i in range(gate_len):
        j = 2 * i
        q0 = qvector[flat_qpair[j]]
        
        if gate_type[i] == 0:
            h(q0)
        elif gate_type[i] == 1:
            ry(angles[i], q0)
        elif gate_type[i] == 2:
            rz(angles[i], q0)
        elif gate_type[i] == 3:
            q1 = qvector[flat_qpair[j + 1]]
            x.ctrl(q0, q1)
    
    mz(qvector)


#...!...!....................
def qiskit_to_gateList(qcL):
    nCirc=len(qcL)
    qc=qcL[0]
    
    nGate=len(qc)  # this is overestimate, includes barriers & measurements
    print('nGate',nGate)

    # pre-alocate memory
    nqL = [qc.num_qubits for qc in qcL]
    gate_len=np.zeros(shape=(nCirc),dtype=int)
    gate_type=np.zeros(shape=(nCirc,nGate),dtype=int)
    gate_qid=np.zeros(shape=(nCirc,nGate,2),dtype=int)
    gate_angle=np.zeros(shape=(nCirc,nGate),dtype=float)
    """
        h: 0
        ry: 1
        rz: 2
        cx: 3
    """
    for j,qc in enumerate(qcL):
        qregs = qc.qregs[0]
        nq = qc.num_qubits
        # Construct a dictionary with address:index of the qregs objects
        qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}
        k=0 # gate counter per circuit
        for op in qc:
            qAddrL = [hex(id(q)) for q in op.qubits]
            qIdxL = [qregAddrD[a] for a in qAddrL]
            gate = op.operation.name
            params = op.operation.params
            # set first gate counter is the real gate name mapping
            if gate == 'h':
                gate_type[j,k]=0
                gate_qid[j,k]= [qIdxL[0], 0]
            elif gate == 'ry':
                gate_type[j,k]=1
                gate_angle[j,k]=params[0]
                gate_qid[j,k]= [qIdxL[0], 0]
            elif gate == 'rz':
                gate_type[j,k]=2
                gate_angle[j,k]=params[0]
                gate_qid[j,k]= [qIdxL[0], 0]
            elif gate == 'cx':
                gate_type[j,k]=3
                gate_qid[j,k]= qIdxL
            elif gate == 'barrier' or gate == 'measure':
                continue           
            else:
                print('ABORT; unknown gate', gate)
                exit(99) 
            k+=1
        gate_len[j]=k  # remember number of gates per circuit
    # print({
    #     'gate_len': gate_len,
    #     'gate_type': gate_type,
    #     'gate_qid': gate_qid,
    #     'gate_angle': gate_angle,
    #     'num_qubits': nqL,
    # })
    return {
        'gate_len': gate_len,
        'gate_type': gate_type,
        'gate_qid': gate_qid,
        'gate_angle': gate_angle,
        'num_qubits': nqL,
    }




