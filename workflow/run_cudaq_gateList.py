#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
 INPUT: pair of hd5 + qpy
 Action:
 - opens hd5, reads .gate_list.h5
 - circ-kernel constructed from  getList
 - run on single gpu node
 - compares results from CPU
 - saves updated HD5
'''
import pdb
# python3 -m pdb run_cudaq_qpyCircs.py  --expName exp_84adce

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
from toolbox.Util_Qiskit import  import_QPY_circs
import os
from time import time
from pprint import pprint
from toolbox.Util_CudaQ import  circ_kernel

from toolbox.logger import log
import cudaq
import traceback
from  Util_EscherHands_ver0  import  make_qcrankObj
from  simple_qcrank_EscherHands_backAer import evaluate
from toolbox.Util_CudaQ import counts_cudaq_to_qiskit

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='exp_i14brq',help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument('-n','--numShots',type=int, default=None, help="(optional) shots per circuit")
    parser.add_argument('-i','--numSample', default=None, type=int, help='(optional) num of images to be processed')
    parser.add_argument("-t", "--target", default="nvidia", choices=['tensornet','nvidia-mgpu','nvidia-mqpu','nvidia'], help="cudaQ target settings")


    parser.add_argument("--inpPath",default='out/',help="input circuits location")
    parser.add_argument("--outPath",default='out/',help="all outputs from experiment")
    
    args = parser.parse_args()
    # make arguments  more flexible
   
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args


#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__": 
    args=get_parser()
    
    inpF=args.expName+'.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.outPath,inpF))
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)

    inpF2=inpF.replace('.h5','.gate_list.h5')
    gateD,_=read4_data_hdf5(os.path.join(args.outPath,inpF2))
    if args.numShots==None:
        shots = expMD['submit']['num_shots']
    else:
        shots=args.numShots        
    assert shots <1024*1024  ,' empirical limit of GPU shots using CudaQ'

    nCirc=expMD['payload']['num_sample']

     # select backend
    target = args.target
    gpu_count = cudaq.num_available_gpus()
    cudaq.set_target(target)
    
    print('M: run %d cudaq-circuit on %d GPUs, %d shots/circ'%(nCirc,gpu_count,shots))

    # converter  and run circuits one by one
    resL=[0]* nCirc # prime the list
    T0=time()
    for i in range(nCirc):
        #print('\nM:run circ ',i)
       
        # Convert values to Python int and assign to a, b
        num_qubit, num_gate = map(int,gateD['circ_type'][i] )
        # Convert to list of integers
        gate_type=list(map(int,gateD['gate_type'][i].flatten()))
        gate_param=list(map(float,gateD['gate_param'][i]))
        assert num_gate<=len(gate_param)
        prOn= num_qubit<6 and i==0 or args.verb>1
        
        if prOn:   print(cudaq.draw(circ_kernel, num_qubit, num_gate, gate_type, gate_param))        
        results = cudaq.sample(circ_kernel,num_qubit, num_gate, gate_type, gate_param, shots_count=shots)
        resL[i]=results
        if prOn:
            print('  assembled & run  elaT= %.1f sec, circ=%d  raw counts:'%(i,time()-T0))
            results.dump()          


    elaT=time()-T0
    print('M: simulated %d circs, elaT=%.1f sec, post-processing ...'%(nCirc,elaT))
    
    #... format cudaq counts to qiskit version
    probsBL=counts_cudaq_to_qiskit(resL)
   
    pp0 = probsBL[0]
    if num_qubit<6:
        print('counts: %s'%pp0)
    else:
        print("counts size: %d"%len(pp0))

    #... recover qcrankObj
    qcrankObj=make_qcrankObj( expMD)
    if args.verb>=2: print(qcrankObj.circuit)            
    u_data=expD['u_input']
    _,u_reco_gpu,res_data_gpu=evaluate(probsBL,expMD,qcrankObj,u_data,args.verb)

    #.... append GPU results
    expMD['short_name']='gpu_'+expMD['short_name']
    expMD['run_gpu']={'num_gpu':gpu_count,'elapsed_time':elaT,'cudaq_target':target}
    if 'run_cpu' in expMD:  expD['u_reco_cpu']=expD['u_reco']  # rename CPU results, just for comparioson
    expD['u_reco']=u_reco_gpu

     #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,expMD)
    print('\n   ./plot_EscherHands.py  --expName %s  -Y '%(expMD['short_name']))
    print('M:done')

