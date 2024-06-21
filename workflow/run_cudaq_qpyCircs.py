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
import pdb
# python3 -m pdb run_cudaq_qpyCircs.py  --expName exp_84adce

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
from toolbox.Util_Qiskit import  import_QPY_circs
import os
from time import time
from pprint import pprint
from toolbox.Util_CudaQ import qiskit_to_cudaq, string_to_dict
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
    parser.add_argument('-n','--numShots',type=int,default=None, help="(optional) shots per circuit")

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

    if args.numShots==None:
        shots = expMD['submit']['num_shots']
    else:
        shots=args.numShots
    assert shots <1024*1024  ,' empirical limit on A100'
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

    
    # preset
    cudaq.set_target("nvidia-mqpu")
    target = cudaq.get_target()
    gpu_count = target.num_qpus()
    
    log.info('M: run %d cudaq-circuit on %d GPUs, %d shots/circ'%(nCirc,gpu_count,shots))

    resL=[0]*nCirc  # prime the list
    try:
        T0=time()
        for i in range(nCirc):
            if args.verb>=2: print('start circ %d'%i)
            resL[i] = cudaq.sample(qKerL[i], shots_count=shots)
        elaT=time()-T0
    except Exception as e:
        log.error("Cuda sampling error: %s", e, exc_info=True)
    print('M:  ended elaT=%.1f sec'%(elaT))
    
    #... format cudaq counts to qiskit version
    probsBL=[0]*nCirc # prime the list
    for i,res in enumerate(resL):
        res = res.__str__()
        probsBL[i] = string_to_dict(res)
    pp0 = probsBL[0]
    if nq<6:
        print('counts: %s'%pp0)
    else:
        print("counts size: %d"%len(pp0))

    #... recover qcrankObj
    qcrankObj=make_qcrankObj( expMD,False,False)
    if args.verb>=2: print(qcrankObj.circuit)            
    u_data=expD['u_input']
    _,u_reco_gpu,res_data_gpu=evaluate(probsBL,expMD,qcrankObj,u_data,args.verb)

    #.... append GPU results
    expMD['short_name']='gpu_'+expMD['short_name']
    expMD['run_gpu']={'num_gpu':gpu_count,'elapsed_time':elaT}
    if 'run_cpu' in expMD:  expD['u_reco_cpu']=expD['u_reco']  # rename CPU results, just for comparioson
    expD['u_reco']=u_reco_gpu

     #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,expMD)
    print('\n   ./plot_EscherHands.py  --expName %s  -Y '%(expMD['short_name']))
    print('M:done')


    
    '''  TO DO
 
    2)DONE  call  u_true,u_reco,res_data=evaluate(probsBL,MD,qcrankObj,u_data)
    imported form simple_qcrank_EscherHands_backAer

    3) SKIP verify the bitstrings are mapped the same way

    3.1) OPEN ISSUE fix  Segmentation fault for  514 CX-circuit: ./simple_qcrank_EscherHands_backAer.py --nqAddr 8 -e -i 2

    4)DONE  add run time to MD

    5) DONE  save hd5 with new results under different name
    '''
        
   

  
