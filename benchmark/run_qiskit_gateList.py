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
from gen_gateList import qiskit_circ_gateList
from qiskit_aer import AerSimulator

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='rblock_a946df',help='[.gate_list.h5]  defines list of circuits to run')
    parser.add_argument('-n','--numShots',type=int, default=101000, help="(optional) shots per circuit")

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


#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__": 
    args=get_parser()
    
    inpF=args.expName+'.gate_list.h5'
    gateD,MD=read4_data_hdf5(os.path.join(args.inpPath,inpF))
    
    if args.verb>=2:
        print('M: MD:');  pprint(MD)

    T0=time()
    qcL=qiskit_circ_gateList(gateD,MD)
    print('\nM:  gen_circ  elaT= %.1f sec '%(time()-T0))
    MD.pop('gate_map')
    nCirc=len(qcL)
    
    #....  excution using backRun(.) .....
    backend = AerSimulator()

    #... auxil MD , filled partially
    MD['submit']={'backend': backend.name,'num_circ':nCirc}

    print('job started, nCirc=%d  nq=%d  shots/circ=%d at %s ...'%(nCirc,qcL[0].num_qubits,args.numShots,backend))
        
    T0=time()
    job=backend.run(qcL,shots=args.numShots)
    #print('   job id:%s , running ...'%job.job_id())
    result=job.result()
    probsBL=result.get_counts()
    elaT=time()-T0
    load1, _, _ = psutil.getloadavg()
    
    print('M:  ended elaT=%.1f sec, end_load1=%.1f\n'%(elaT,load1))
    MD['run_cpu']={'num_cpu':os.cpu_count(),'elapsed_time':elaT,'cpu_load1':load1}

    MD['short_name']+='_cpu%d'%MD['run_cpu']['num_cpu']
    expD={} # no arrays to save
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,MD)

    print('M:done qiskit %s  elaT %.1f'%(MD['short_name'],elaT))

