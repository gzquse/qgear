import cudaq
import traceback

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
    raw_list =1 
    #print("raw_list:", raw_list)
    mapped_dict = {}
    for item in raw_list:
        key, value = item.split(":")
        mapped_dict[key] = int(value)

    # Create a new dictionary with reversed keys
    rev = {reverse_key(k): v for k, v in mapped_dict.items()}
    return rev

#...!...!....................
def qiskit_to_cudaq(qc):
    qregs=qc.qregs[0]
    #print(qregs)
    # Construct a dictionary with address:index of the qregs objects
    qregAddrD = {hex(id(obj)): idx for idx, obj in enumerate(qregs)}
    #print('qregAddrD:',qregAddrD)

    nq = qc.num_qubits
    # single gpu
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
