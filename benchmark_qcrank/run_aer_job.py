#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
QCrank

Use Qiskit ideal  simulator
use all-to-all connectivity.
Run simulations w/o submit/retrieve of the job
Simulations run locally.
Records meta-data containing  job_id
HD5 arrays contain input images
Dependence: qpixl, qiskit
'''

import time
import sys,os
import numpy as np
from pprint import pprint

from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
from qiskit import  transpile
from toolbox.Util_ibm import harvest_ibmq_backRun_submitMeta, harvest_backRun_results, harvest_circ_transpMeta
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_Qiskit import  circ_depth_aziz


sys.path.append(os.path.abspath("/daan_qcrank/py"))
from qpixl import qcrank
import argparse

#...!...!..................
def get_parser(backName="ibmq_qasm_simulator",provName="local sim",doMathOp=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath",default='out/',help="input packed image")
    parser.add_argument("--outPath",default='out/',help="raw outputs from experiment")
    parser.add_argument("--cannedExp",  default='canImg_b2_32_32',help='packed image name')
    parser.add_argument("--expName",  default=None,help='(optional) replaces IBMQ jobID assigned during submission by users choice')
    
    # .... job running
    parser.add_argument('-n','--numShotPerAddr',type=int,default=400, help="shots per address of QCrank, or as-is if negative")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
    parser.add_argument( "-G","--exportGateList", action='store_true', default=False, help="exprort gate list for CudaQ executiomn ")

    args = parser.parse_args()
    args.backend='aer'
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
    
    expD,expMD=canned_qcrank_inp(args)
        
    pprint(expMD)
    numShots=expMD['submit']['num_shots']
    
    #... adjust parser settings
    #1args.outPath=os.path.join(args.basePath,'meas')    
    #assert os.path.exists(args.outPath)
    
    # ------  circuit generation -------
    qcrankObj=make_qcrank( expMD)
    qcP=qcrankObj.circuit
    nqTot=qcP.num_qubits
    
    print('M: circuit has %d qubits'%(qcP.num_qubits))
    circ_depth_aziz(qcP,text='circ_orig')
    
    if args.verb>0 and qcP.num_qubits<7 or args.verb>1 :
        print('M:.... PARAMETRIZED IDEAL CIRCUIT ..............')        
        print(circuit_drawer(qcP, output='text',cregbundle=True))

    backend = AerSimulator()

    # transpilation i snot needed fro Aer, but we do it here for convenience for CudaQ
    qcT=transpile(qcP,backend, basis_gates=['cx','ry','h']) 
             
    if  qcP.num_qubits<6:
        print('M:.... PARAMETRIZED TRANSPILED CIRCUIT .............. for',backend)
        print(qcT.draw(output='text',idle_wires=False))  # skip ancilla

    harvest_circ_transpMeta(qcT,expMD,args.backend)
     
    # -------- bind the data to parametrized circuit  -------
    f_data=expD['inp_fdata']
    qcrankObj.bind_data(f_data, max_val=expMD['payload']['qcrank_max_fval'])
    
    # generate the instantiated circuits
    qcEL = qcrankObj.instantiate_circuits()
    nCirc=len(qcEL)
    print('M: execution-ready %d circuits on %d qubits on %s'%(nCirc,nqTot,backend.name))                        

    if args.exportGateList:
        from  toolbox.Util_CudaQ import qiskit_to_gateList
        gateD,mapD=qiskit_to_gateList(qcEL)        
        expD.update(gateD)
        expMD.update(mapD)
        outF=os.path.join(args.outPath,expMD['short_name']+'.gate_list.h5')
        write4_data_hdf5(expD,outF,expMD)
        print('   ./run_cudaq_job.py --circName   %s  -n 500 -E   \n'%(expMD['short_name']))
        exit(0)
 
    if not args.executeCircuit:
        pprint(expMD)
        print('NO execution of circuit, use -E to execute the job')
        exit(0)

    # ----- submission ----------
    if args.executeCircuit:
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
 

     
