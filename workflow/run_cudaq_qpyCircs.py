#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
 INPUT: pair of hd5 + qpy
 Action:
 - opens hd5, reads qpy
 - converts qcL to qKerL (Qiskit -->CudaQ
 - run on single gpu node
 - compares results from CPU
 - saves updated HD5
'''
#import pdb
# python3 -m pdb run_cudaq_qpyCircs.py  --expName exp_84adce

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5
from toolbox.Util_Qiskit import  import_QPY_circs
import os
from time import time
from pprint import pprint
from toolbox.Util_CudaQ import qiskit_to_cudaq
import cudaq

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='exp_i14brq',help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument("--inpPath",default='out/',help="input circuits location")
    parser.add_argument("--outPath",default='out/',help="all outputs from  experiment")
    
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

    
    shots = expMD['submit']['num_shots']
    nq=expMD['qiskit_transp']['num_qubit']

    qcL=import_QPY_circs(expMD,args)
    nCirc=len(qcL)
    
    # converter list of circ
    qKerL=[0 for i in range(nCirc)] # prime the list
    for i in range(nCirc):        
        qKerL[i]=qiskit_to_cudaq(qcL[i])

    print('M: converted %d circ'%nCirc)
    if nq<6:
        print(cudaq.draw(qKerL[0]))

    print('M: run %d cudaq-circuit  on GPU'%nCirc)
    resL=[0 for i in range(nCirc)] # prime the list
    cudaq.set_target("nvidia")
    T0=time()
    for i in range(nCirc): 
        resL[i] = cudaq.sample(qKerL[i], shots_count=shots)

    elaT=time()-T0
    print('M:  ended elaT=%.1f sec'%(elaT))
    res0=resL[0]
    if nq<6:
        print('counts:',res0)
    else:
        print("counts size", len(res0))

    '''  TO DO
    1) convert resL[] to format: 
{'0000': 523, '1011': 1983, '1010': 2005, '0100': 3307, '1101': "
 "1387, '0001': 1551, '1111': 2274, '1100': 1988, '0110': 2055, '1000': 4127, "
 "'0010': 2864, '0111': 2093, '0011': 2985, '1001': 16, '0101': 581, '1110': "
 '2261}'
 
    2) call  u_true,u_reco,res_data=evaluate(probsBL,MD,qcrankObj,u_data)
    imported form simple_qcrank_EscherHands_backAer

    3) verify the bitstrings are mapped the same way
    '''
        
    print('M:done')

  
