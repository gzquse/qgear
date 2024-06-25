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

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='exp_i14brq',help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument('-n','--numShots',type=int, default=None, help="(optional) shots per circuit")
    parser.add_argument('-i','--numSample', default=None, type=int, help='(optional) num of images to be processed')
    parser.add_argument("-m", "--target", default="nvidia", help="GPU settings")

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
    nq=expMD['qiskit_transp']['num_qubit']

    nCirc=expMD['payload']['num_sample']
    # converter list of circ
    countsL=[0]* nCirc # prime the list
    T0=time()
    for i in range(nCirc):
        print('\nM:run circ ',i)
        prOn= nq<6 and i==0 or args.verb>1
        ng=int(gateD['gate_len'][i])
        gate_type = gateD['gate_type'][i]
        gate_qid=gateD['gate_qid'][i]
        gate_angle=gateD['gate_angle'][i]
        qubit_count=int(gateD['num_qubits'][0][i])
        # Flatten qpair list
        fpairs = [int(qubit) for pair in gate_qid for qubit in pair]
        fangles = [float(x) for x in gate_angle]
        fgate_type = [int(x) for x in gate_type]
        # print("num_qubits:", qubit_count, type(qubit_count))
        # print("gate_type:", fgate_type, type(fgate_type))
        # print("ng:", ng, type(ng))
        # print("fpairs:", fpairs, type(fpairs))
        # print("fangles:", fangles, type(fangles))
        if prOn:   print(cudaq.draw(circ_kernel, qubit_count, ng, fgate_type, fpairs, fangles))
        counts = cudaq.sample(circ_kernel, qubit_count, ng, fgate_type, fpairs, fangles, shots_count=shots)

        print('  assembled & run  elaT= %.1f sec'%(time()-T0))
        if prOn:  counts.dump()          
        countsL[i]=counts

    print('M: converted %d circ, elaT=%.1f sec'%(nCirc,time()-T0))
    
