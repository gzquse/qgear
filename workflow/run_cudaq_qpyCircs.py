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
from toolbox.Util_CudaQ import qiskit_to_cudaq, string_to_dict
from workflow.simple_qcrank_EscherHands_backAer import construct_random_input
from toolbox.logger import log
import cudaq
import traceback

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='exp_i14brq',help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument("--inpPath",default='out/',help="input circuits location")
    parser.add_argument("--outPath",default='out/',help="all outputs from  experiment")
    
    args = parser.parse_args()
    # make arguments  more flexible
   
    for arg in vars(args):  log.info( 'myArg:');print(arg, getattr(args, arg))
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
        log.info('M:expMD:');  pprint(expMD)

    
    shots = expMD['submit']['num_shots']
    nq=expMD['qiskit_transp']['num_qubit']

    qcL=import_QPY_circs(expMD,args)
    nCirc=len(qcL)
    
    # converter list of circ
    qKerL=[0 for i in range(nCirc)] # prime the list
    for i in range(nCirc):        
        qKerL[i]=qiskit_to_cudaq(qcL[i])

    log.info('M: converted %d circ'%nCirc)
    if nq<6:
        print(cudaq.draw(qKerL[0]))

    log.info('M: run %d cudaq-circuit  on GPU'%nCirc)
    resL=[0 for i in range(nCirc)] # prime the list
    cudaq.set_target("nvidia")
    try:
        T0=time()
        for i in range(nCirc):
            if args.verb>=2: log.info('start circ %d'%i)
            resL[i] = cudaq.sample(qKerL[i], shots_count=shots)
        elaT=time()-T0
    except Exception as e:
        log.error("Cuda sampling error: %s", e, exc_info=True)
    log.info('M:  ended elaT=%.1f sec'%(elaT))
    # format cudaq counts to qiskit version
    # Apply the function to each dictionary in the list
    #log.info("resL:", resL)
    try:
        for i,res in enumerate(resL):
            res = res.__str__()
            resL[i] = string_to_dict(res)
        res0 = resL[0]
        if nq<6:
            log.info('counts: %s'%res0)
        else:
            log.info("counts size: %d"%len(res0))
    except Exception as e:
        log.error("counts string convertion error: %s", e)
        
    # construct data
    probsBL = resL
    MD['run_gpu']={'num_gpu':cp.cuda.runtime.getDeviceCount(),'elapsed_time':elaT}
    #u_data,f_data= construct_random_input(MD,args.verb)
    u_data=expD['u_data']
    f_data=
    # try:
    #     u_true,u_reco,res_data=evaluate(probsBL,expMD,qcrankObj,u_data)
    # except Exception as e:
    #     logger.error("An error occurred during evaluate: %s", e, exc_info=True)
    #u_true,u_reco,res_data=evaluate(probsBL,MD,qcrankObj,u_data)
    '''  TO DO
 
    2) call  u_true,u_reco,res_data=evaluate(probsBL,MD,qcrankObj,u_data)
    imported form simple_qcrank_EscherHands_backAer

    3) verify the bitstrings are mapped the same way

    3.1) fix  Segmentation fault for  514 CX-circuit: ./simple_qcrank_EscherHands_backAer.py --nqAddr 8 -e -i 2

    4) add run time to MD

    5) save hd5 with new results under different name
    '''
        
    log.info('M:done')

  
