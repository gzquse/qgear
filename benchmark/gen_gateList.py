#!/usr/bin/env python3
''' 
generate random CX-blocks circuit and store serialize gate-list as HD5

This is how CX-block looks like, with randomized qubits IDs and angles θ, φ

      ░ ┌───────┐      ░ 
q_0: ─░─┤ Ry(θ) ├──■───░─
      ░ ├───────┤┌─┴─┐ ░ 
q_1: ─░─┤ Rz(φ) ├┤ X ├─░─
      ░ └───────┘└───┘ ░ 

'''

import os,hashlib,sys
import numpy as np
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from Util_H5io4 import write4_data_hdf5
from pprint import pprint

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=5, type=int, help='circuit width')
   
    parser.add_argument('-k','--numCX', default=4, type=int, help='num of CX gates')
    parser.add_argument('-i','--numCirc', default=2, type=int, help='num of circuits  in to the job')
    parser.add_argument("--expName",  default=None,help='(optional) ')

    # IO paths
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--outPath",default='out/',help="(optional) redirect all outputs ")
    
    args = parser.parse_args()
    # make arguments  more flexible
    if args.basePath=='env': args.basePath= os.environ['Cudaq_dataVault']
    if args.basePath!=None and args.outPath!=None :
        args.outPath=os.path.join(args.basePath,'circ') 

    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath, exist_ok=True)
    assert args.numQubits>=2
    return args

#...!...!....................
def show_CX_block():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    # Define the parameters theta and phi
    theta = Parameter('θ')
    phi = Parameter('φ')
   
    # Create a quantum circuit with one qubit
    qc = QuantumCircuit(2)
   
    # Apply the Ry and Rz gates with parameters
    qc.barrier()
    qc.ry(theta, 0)
    qc.rz(phi, 1)
    qc.cx(0,1)
    qc.barrier()
    
    # Draw the circuit
    print(qc.draw())

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
def generate_random_gateList(args):
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
                  
         #if args.numCX<9: print(qpairs.T)
         #if args.numCX<9: print(rpairs.T)  # reversed pairs
         circ_type[j]=[nq,nGate]
         
         gate_type[j,0::3]=np.concatenate((t_ry, qpairs), axis=1)  # Shape ( k,3)
         gate_type[j,1::3]=np.concatenate((t_rz, rpairs), axis=1) 
         gate_type[j,2::3]=np.concatenate((t_cx, qpairs), axis=1)

         gate_param[j,2::3]=0  # CX has no parameter, to make it look nice
         
         #print(j, 'gate_type:\n',gate_type[j].T)  # type is OK
         #print(j, 'gate_param:\n',gate_param[j].T)  # type is OK
    outD={'circ_type': circ_type, 'gate_type': gate_type, 'gate_param': gate_param}
    md={'gate_map':m, 'num_cx':args.numCX, 'num_qubit':nq,'num_gate':nGate,'num_circ':nCirc}
    return outD,md 

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    show_CX_block()

    T0=time()
    outD,MD=generate_random_gateList(args)
    print('M:  gen_rand  elaT= %.1f sec '%(time()-T0))

    MD['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    
    if args.expName==None:
        MD['short_name']='rcirc_'+MD['hash']
    else:
        MD['short_name']=args.expName

    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.gate_list.h5')
    write4_data_hdf5(outD,outF,MD)

    pprint(MD)
    print('\n time numactl --cpunodebind=0 --membind=0    ./run_gateList.py  --expName %s  -b qiskit-cpu '%(MD['short_name']))
    print('time     ./run_gateList.py  --expName %s  -b nvidia    # 1 GPU'%(MD['short_name']))
    print(' time     ./run_gateList.py  --expName %s  -b nvidia-mqpu   # all GPUs parallel'%(MD['short_name']))
    print('M:done')

    
    
