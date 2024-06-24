#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
 INPUT: pair of hd5 + qpy
 Action:
 - opens hd5, reads qpy
 - converts qcL:Qiskit --> serializes numpy-list of gates
 - saves updated HD5
'''
import pdb
# python3 -m pdb run_cudaq_qpyCircs.py  --expName exp_84adce

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
from toolbox.Util_Qiskit import  import_QPY_circs
from toolbox.Util_CudaQ import qiskit_to_gateList
import os
from time import time
from pprint import pprint

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

    qcL=import_QPY_circs(expMD,args)
       
    # converter list of circ
    T0=time()
    outD=qiskit_to_gateList(qcL)
    print('M: converted %d circ, elaT=%.1f sec'%(len(qcL),time()-T0))
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.gate_list.h5')
    write4_data_hdf5(outD,outF)
    print('\n   ./run_cudaq_gateList.py  --expName %s  '%(expMD['short_name']))
    print('M:done')

