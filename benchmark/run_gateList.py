#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
 INPUT: circuit defined by gteList , hd5
 Action:
 - opens reads .gate_list.h5
 - qiskit circ constructed from  getList
 - run on all CPUs on a node
 - saves updated HD5
'''

import psutil

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
import os
from time import time
from pprint import pprint
from toolbox.Util_Qiskit import qiskit_circ_gateList
from qiskit_aer import AerSimulator
from toolbox.Util_CudaQ import circ_kernel
import cudaq

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='rblock_a946df',help='[.gate_list.h5]  defines list of circuits to run')
    parser.add_argument('-n','--numShots',type=int, default=101000, help="(optional) shots per circuit")
    parser.add_argument("-t", "--target", default="nvidia", choices=['qiskit-cpu','tensornet','nvidia-mgpu','nvidia-mqpu','nvidia','qpp-cpu'], help="cudaQ target settings")

    # IO paths
    parser.add_argument("--basePath",default='env',help="head path for set of experiments, or env")
    parser.add_argument("--inpPath",default=None,help="input circuits location")
    parser.add_argument("--outPath",default=None,help="all outputs from experiment")
       
    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['Cudaq_dataVault']
    if args.inpPath==None: args.inpPath=os.path.join(args.basePath,'circ') 
    if args.outPath==None: args.outPath=os.path.join(args.basePath,'meas') 

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def run_qiskit_aer():
    job=backend.run(qcL,shots=args.numShots)
    #print('   job id:%s , running ...'%job.job_id())
    result=job.result()
    probsBL=result.get_counts()
    return len(probsBL[0])

#...!...!....................
def run_cudaq(num_qpus):
    # converter  and run circuits one by one
    resL=[0]* nCirc # prime the list
    
    for i in range(nCirc):
        # Convert values to Python int and assign to a, b
        num_qubit, num_gate = map(int,gateD['circ_type'][i] )
        # Convert to list of integers
        gate_type=list(map(int,gateD['gate_type'][i].flatten()))
        gate_param=list(map(float,gateD['gate_param'][i]))
        assert num_gate<=len(gate_param)
        prOn= num_qubit<6 and i==0 or args.verb>1
        
        if prOn:   print(cudaq.draw(circ_kernel, num_qubit, num_gate, gate_type, gate_param))    
        if target == "nvidia-mgpu" or target == "nvidia":    
            results = cudaq.sample(circ_kernel,num_qubit, num_gate, gate_type, gate_param, shots_count=args.numShots)
        elif target == "nvidia-mqpu":
            gpu_id = i % num_qpus
            futures = cudaq.sample_async(circ_kernel, shots_count=args.numShots, qpu_id=gpu_id)
            # Retrieve and print results
            results = [counts.get() for counts in futures]
        resL[i]=results
    
    return len(results)


#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__": 
    args=get_parser()
    target = args.target
    inpF=args.expName+'.gate_list.h5'
    gateD,MD=read4_data_hdf5(os.path.join(args.inpPath,inpF))
    nCirc=MD['num_circ']
    if args.verb>=2:
        print('M: MD:');  pprint(MD)
    
    T0=time()
    if 'qiskit' in target:
        qcL=qiskit_circ_gateList(gateD,MD)
        print('\nM:  gen_circ  elaT= %.1f sec '%(time()-T0))
        MD.pop('gate_map')
        #....  excution using backRun(.) .....
        backend = AerSimulator()
    else:
        cudaq.set_target(target)
        target = cudaq.get_target()
        # only get qpus not gpus
        num_qpus = target.num_qpus()
    

    #... auxil MD , filled partially
    #XMD['submit']={'backend': backend.name,'num_circ':nCirc}

    print('job started, nCirc=%d  nq=%d  shots/circ=%d   target=%s ...'%(nCirc,MD['num_qubit'],args.numShots,target))
        
    T0=time()
    if 'qiskit' in target:
        resLen=run_qiskit_aer()
    else:
        resLen=run_cudaq(num_qpus)
    elaT=time()-T0
    load1, _, _ = psutil.getloadavg()
    
    print('M:  ended elaT=%.1f sec, end_load1=%.1f\n'%(elaT,load1))
    MD['run_cpu']={'num_cpu':os.cpu_count(),'elapsed_time':elaT,'cpu_load_1min':load1,'num_mstrings':resLen,'target':target}

    MD['short_name']+='_cpu%d'%MD['run_cpu']['num_cpu']
    expD={} # no arrays to save
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,MD)
    
    print('M:done qiskit %s  elaT %.1f'%(MD['short_name'],elaT))

    pprint(MD)
