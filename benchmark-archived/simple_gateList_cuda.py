#!/usr/bin/env python3
# genarates parametrized circuit 

# mpirun -np 4 --allow-run-as-root python3 cuquantum_backends.py

import cudaq
from time import time
import numpy as np

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=5, type=int, help='circuit width  ')
    parser.add_argument('-i','--numCirc', default=1, type=int, help='num of circuits  in to the job')

    parser.add_argument('-n','--numShots', default=10, type=int, help='num of shots')
    parser.add_argument('-k','--numCX', default=4, type=int, help='num of CX gates')
    parser.add_argument('-t','--cudaqTarget',default="nvidia-mgpu", choices=['qpp-cpu','nvidia','nvidia-mgpu','nvidia-mqpu'], help="CudaQ target backend")

    args = parser.parse_args()
    cudaq.mpi.initialize()
    myrank = cudaq.mpi.rank()
    if myrank==0:
        for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))

    assert args.numQubits>=2
    return args

#...!...!....................
def random_qubit_pairs(nq, k):
    # draw 2 different elements out of a set     
    # Generate all possible pairs (excluding self-pairs)
    all_pairs = np.array([(i, j) for i in range(nq) for j in range(nq) if i != j])
    
    # Randomly select k pairs from the list of all pairs
    selected_indices = np.random.choice(len(all_pairs), k, replace=True)
    pairs = all_pairs[selected_indices]    
    return pairs

    
#...!...!....................
def generate_random_gateList_np(args):
    nCirc=args.numCirc
    nGate=3*args.numCX
    nq=args.numQubits
    m={'h': 1, 'ry': 2,  'rz': 3, 'cx':4, 'measure':5 } # mapping of gates
     
    # pre-allocate memory
    circ_type=np.zeros(shape=(nCirc,2),dtype=np.int32) # [num_qubit, num_gate]
    gate_type=np.zeros(shape=(nCirc,nGate,3),dtype=np.int32) # [gate_type, qubit1, qubit2] 
    gate_param=np.random.uniform(0, np.pi, size=(nCirc, nGate)).astype(np.float32)

    t_ry=np.full(( args.numCX,1), m['ry'] )
    t_rz=np.full(( args.numCX,1), m['rz'] )
    t_cx=np.full(( args.numCX,1), m['cx'] )
 
    for j in range(nCirc):
         qpairs = random_qubit_pairs(nq,args.numCX)
         rpairs=qpairs[:,::-1]
                  
         circ_type[j]=[nq,nGate]
         
         gate_type[j,0::3]=np.concatenate((t_ry, qpairs), axis=1)  # Shape ( k,3)
         gate_type[j,1::3]=np.concatenate((t_rz, rpairs), axis=1) 
         gate_type[j,2::3]=np.concatenate((t_cx, qpairs), axis=1)

         gate_param[j,2::3]=0  # CX has no parameter, to make it look nice
         
    return circ_type,gate_type, gate_param
         

#...!...!....................
@cudaq.kernel
def ghz_kernel(qubit_count: int, nRep: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    for j in range(nRep):
        for qubit in range(qubit_count - 1):
            ry(0.1, qvector[qubit])
            rz(0.2, qvector[qubit+1])
            x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)

#...!...!....................
@cudaq.kernel
def param_kernel(num_qubit: int, num_gate: int, gate_type: list[int], angles: list[float]):
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

    mz(qvector)

#=================================
#=================================
#  M A I N 
#=================================
#=================================

circ_typeV,gate_typeV, gate_paramV=generate_random_gateList_np(args)
print('M:  gen_rand  elaT= %.1f sec '%(time()-T0))
print('M: shapes circ_typeV,gate_typeV, gate_paramV',circ_typeV.shape,gate_typeV.shape, gate_paramV.shape)

if __name__ == "__main__":
    args=get_parser()
    ranks = cudaq.mpi.num_ranks()
    myrank = cudaq.mpi.rank()
    T0=time()
    target=args.cudaqTarget   
    nc=args.numCirc
    shots=args.numShots
   
    cudaq.set_target(target)
    if cudaq.num_available_gpus() == 0:
        print("This code requires a GPU to run. No GPU detected.")
        exit(0)

    kerTime=0  # only GPU kernle time
    T00=time()  # wall time start
    for i in range(nc):
        # Convert values to Python int and assign to a, b
        nq, num_gate = map(int,circ_typeV[i] )
        # Convert to list of integers
        gate_type=list(map(int,gate_typeV[i].flatten()))
        gate_param=list(map(float,gate_paramV[i]))
        assert num_gate<=len(gate_param)
        prOn= nq<6 and i==0 or args.verb>1
    
        if myrank == 0:
            print('\n start nq=%d, target=%s, ranks=%d, aval GPUs=%d nGate=%d shots=%d ... '%(nq,target,ranks,cudaq.num_available_gpus(),num_gate,shots))

        if prOn:
            print(cudaq.draw(param_kernel, nq, num_gate, gate_type, gate_param))

        # .... run kernel on GPUs .....
        T0=time()
        result=cudaq.sample(ghz_kernel, nq, 50, shots_count=shots)
        elaT=time()-T0
        
        kerTime+=elaT
        if myrank == 0 and i==0:
            if args.verb>1: print('counts:',result)
            print('\nelaT= %.1f sec , nq=%d, nGate=%d, shots=%d, target=%s, ranks=%d, numSol=%d '%(elaT,nq,num_gate,shots,target,ranks,len(result)))

    #......  all circuits done ....
    wallTime=time()-T00
    if myrank == 0:
        print('M: done %d circuits, wallT=%.1f sec  kerT=%.1f sec'%(nc,wallTime, kerTime))
     
    cudaq.mpi.finalize()