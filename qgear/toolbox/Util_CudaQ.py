import cudaq
import numpy as np
from typing import List

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
def qiskit_to_cudaq(qc):  # Object method, circuit will be very slow
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

#...!...!....................
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
def circ_kernel(num_qubit: int, num_gate: int, gate_type: list[int], angles: list[float]):
    qvector = cudaq.qvector(num_qubit)
    
    for i in range(num_gate):
        j = 3 * i
        gateId=gate_type[j]
        q0 = qvector[gate_type[j+1]]
        
        if gateId == 1:
            h(q0)
        elif gateId == 2:
            ry(angles[i], q0)
        elif gateId == 3:
            rz(angles[i], q0)
        elif gateId == 4:
            q1 = qvector[gate_type[j + 2]]
            x.ctrl(q0, q1)
        elif gateId == 5:
            #martin_add_measurement
            continue
        elif gateId == 6:  # controlled phase
            cr1(angles[i], [qvector[gate_type[j + 2]]], q0)
        elif gateId == 7:  # swap
            swap(q0, qvector[gate_type[j + 2]])
    
    mz(qvector)

#...!...!....................
@cudaq.kernel
def qft(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the QFT.'''
    qubit_count = len(qubits)
    # Apply Hadamard gates and controlled rotation gates.
    for i in range(qubit_count):
        h(qubits[i])
        for j in range(i + 1, qubit_count):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qubits[j]], qubits[i])

#...!...!....................
@cudaq.kernel
def inverse_qft(qubits: cudaq.qview):
    '''Args:
    qubits (cudaq.qview): specifies the quantum register to which apply the inverse QFT.'''
    cudaq.adjoint(qft, qubits)

#...!...!....................
@cudaq.kernel
def qft_kernel(input_state: List[int]):
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
    qft(qubits)

    # Apply the inverse quantum Fourier Transform
    #inverse_qft(qubits)
    
#...!...!....................
def qiskit_to_gateList(qcL):
    nCirc=len(qcL)
    qc=qcL[0]
    
    nGate=len(qc)  # this is overestimate, includes barriers & measurements
    print('qiskit_to_gateList: nGate',nGate)

    # pre-allocate memory
    circ_type=np.zeros(shape=(nCirc,2),dtype=np.int32) # [num_qubit, num_gate]
    gate_type=np.zeros(shape=(nCirc,nGate,3),dtype=np.int32) # [gate_type, qubit1, qubit2] 
    gate_param=np.zeros(shape=(nCirc,nGate),dtype=np.float32)

    m={'h': 1, 'ry': 2,  'rz': 3, 'cx':4, 'measure':5, 'cp': 6, 'swap': 7} # mapping of gates

    for j,qc in enumerate(qcL):
        qregs = qc.qregs[0]
        nq = qc.num_qubits
        assert  nGate>=len(qc)  # to be sure we reserved enough space
        
        # Construct a dictionary with address:index of the qregs objects
        qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}
        k=0 # gate counter per circuit
        for op in qc:
            gate = op.operation.name
            params = op.operation.params
            qIdxL = [qc.find_bit(q).index for q in op.qubits]

            if gate == 'h':
                gate_type[j,k]= [m[gate],qIdxL[0], 0]
            elif gate == 'ry':
                gate_param[j,k]=params[0]
                gate_type[j,k]= [m[gate],qIdxL[0], 0]
            elif gate == 'rz':
                gate_param[j,k]=params[0]
                gate_type[j,k]= [m[gate],qIdxL[0], 0]
            elif gate == 'cx':
                gate_type[j,k]= [m[gate]]+qIdxL
            elif gate == 'cp':
                gate_param[j,k]=params[0]   
                gate_type[j,k]= [m[gate]]+qIdxL
            elif gate == 'swap':
                gate_type[j,k]= [m[gate]]+qIdxL
            elif gate == 'barrier':                
                continue
            elif  gate == 'measure':
                continue # Martin, ADD ME
            else:
                print('ABORT; unknown qiskit gate', gate)
                exit(99) 
            k+=1
        circ_type[j]=[nq,k]  # remember number of gates per circuit
    outD={'circ_type': circ_type, 'gate_type': gate_type, 'gate_param': gate_param}
    md={'gate_map':m,'num_qubit':nq,'num_gate':nGate,'num_circ':nCirc}
    return outD, md


