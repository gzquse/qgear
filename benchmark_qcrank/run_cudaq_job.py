#!/usr/bin/env python3
__author__ = "Jan Balewski + Martin"
__email__ = "janstar1122@gmail.com"

'''
QCrank

Use CudaQ  GPU  simulator
Input is serialized gate list from Util_CudaQ: qiskit_to_gateList()

Run simulations w/o submit/retrieve of the job
Simulations run locally.
Records meta-data containing  job_id
HD5 arrays contain input images
Dependence: cudaq
'''

from time import time
import sys,os
import numpy as np
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

import cudaq
from toolbox.Util_CudaQ import circ_kernel

import argparse

#...!...!..................
def get_parser(backName="ibmq_qasm_simulator",provName="local sim",doMathOp=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath",default='out/',help="input packed image")
    parser.add_argument("--outPath",default='out/',help="raw outputs from experiment")
    parser.add_argument("--circName",  default=None,help='gate-list file name')
    
    # .... job running
    parser.add_argument('-n','--numShotPerAddr',type=int,default=400, help="shots per address of QCrank, or as-is if negative")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")

    args = parser.parse_args()
    args.backend='device-mgpu'  # not tested
    # make arguments  more flexible
    if args.basePath=='env': args.basePath= os.environ['Cudaq_dataVault']
    if args.basePath!=None:
        args.outPath=os.path.join(args.basePath,'jobs')
    
        
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args


#...!...!..................
def canned_qcrank_inp(args):
    inpF=args.cannedExp+'.qcrank_inp.h5'
    bigD,md=read4_data_hdf5(os.path.join(args.outPath,inpF))

    sd={}
    sd['num_shots']=args.numShotPerAddr*md['payload']['seq_len']
    md['submit']=sd

    return bigD,md

#...!...!....................
def make_qcrank(md, barrier=True):
    
    pmd=md['payload']
    nq_addr=pmd['nq_addr']
    nq_data=pmd['nq_fdata'] 
    
    #.... create parameterized QCrank  circ
    qcrankObj = qcrank.ParametrizedQCRANK(
        nq_addr, nq_data,
        qcrank.QKAtan2DecoderQCRANK,
            keep_last_cx=True, barrier=barrier,
        measure=True, statevec=False,
        reverse_bits=True  # to match Qiskit littleEndian
    )
    return  qcrankObj

#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer')

    inpF=args.circName+'.gate_list.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.outPath,inpF))


    circ_typeV=expD.pop('circ_type')
    gate_typeV=expD.pop('gate_type')
    gate_paramV=expD.pop('gate_param')
    
    nCirc=circ_typeV.shape[0]  # make it generic, despite only 1 circuit is expected

    T0=time()
    for i in range(nCirc):
        print('\nM:run circ ',i)

        # Convert values to Python int and assign to a, b
        num_qubit, num_gate = map(int,circ_typeV[i] )
        # Convert to list of integers
        gate_type=list(map(int,gate_typeV[i].flatten()))
        gate_param=list(map(float,gate_paramV[i]))
        assert num_gate<=len(gate_param)
        prOn= num_qubit<6 and i==0 or args.verb>1
        if prOn:   print(cudaq.draw(circ_kernel, num_qubit, num_gate, gate_type, gate_param))

    not_tested_on_GPU
    # ----- submission ----------
    T0=time.time()
    job =  backend.run(qcEL,shots=numShots)
    jid=job.job_id()

    print('submitted JID=',jid,backend ,'\n do %.2fM shots , wait for execution ...'%(numShots/1e6))
    
    harvest_ibmq_backRun_submitMeta(job,expMD,args)
    T1=time.time()
    print(' job done, elaT=%.1f min'%((T1-T0)/60.))
    
    print('M: got results')
    harvest_backRun_results(job,expMD,expD)
    
    outF=os.path.join(args.outPath,expMD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,expMD)

    print('   ./postproc_exp.py --expName   %s --showPlots b c   \n'%(expMD['short_name']))
 

     
