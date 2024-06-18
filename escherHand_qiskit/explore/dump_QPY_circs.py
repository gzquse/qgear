#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5
from toolbox.Util_Qiskit import   import_QPY_circs, qiskit_to_cudaq
import os
from pprint import pprint
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

    qcL=import_QPY_circs(expMD,args)

    print('M:got %d circuits'%len(qcL))
    
    # converter
    shots = expMD['submit']['num_shots']
    # single circuit
    qc1 = qcL[0]
    # FAILED (Cannot get Qubit Object details correctly)
    elaT, gpu_counts = qiskit_to_cudaq(qcL, shots)

    print("Running time: %s, GPU counts: %d"%elaT, gpu_counts)

    print('M:done')

  
