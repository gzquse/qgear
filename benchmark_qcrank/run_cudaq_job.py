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

from time import time, localtime
import sys,os
import numpy as np
from pprint import pprint
from toolbox.Util_ibm import harvest_circ_transpMeta
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_IOfunc import dateT2Str
import cudaq
from toolbox.Util_CudaQ import circ_kernel, counts_cudaq_to_qiskit, qiskit_to_gateList
from toolbox.Util_Qiskit import pack_counts_to_numpy
from toolbox.Util_Qiskit import  circ_depth_aziz
from qiskit import  transpile
from qiskit_aer import AerSimulator
import hashlib
import argparse

# sys.path.append(os.path.abspath("/daan_qcrank/py"))
from toolbox.qpixl import qcrank

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("-b", "--backend", default="nvidia", choices=['qiskit-cpu','tensornet','nvidia-mqpu','nvidia','qpp-cpu'], help="cudaQ target settings")
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath",default='out/',help="input packed image")
    parser.add_argument("--outPath",default='out/',help="raw outputs from experiment")
    parser.add_argument("-c", "--circName",  default='canImg_b2_32_32', help='gate-list file name')
    parser.add_argument("--expName",  default=None,help='(optional) replaces jobID assigned during submission by users choice')
    parser.add_argument('-t','--target-option',default='fp32,mgpu',help='target options')
    # .... job running
    parser.add_argument('-n','--numShotPerAddr',type=int,default=400, help="shots per address of QCrank, or as-is if negative")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=True, help="may take long time, test before use ")

    args = parser.parse_args()

    # make arguments more flexible
    if args.basePath=='env': args.basePath= os.environ['Cudaq_dataVault']
    if args.basePath!=None:
        args.outPath=os.path.join(args.basePath,'jobs')
    # default cudaQ settings
    args.myRank,  args.numRank = 0,1
    if args.backend=='nvidia': 
        cudaq.mpi.initialize()
        args.myRank  = int(os.environ['SLURM_PROCID'])
        args.numRank = int(os.environ['SLURM_NTASKS'])
    if args.myRank == 0:
        rank_print('myArg-program:',parser.prog)
        for arg in vars(args):  rank_print('myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args

# Add this helper function at the start of the file, after imports
def rank_print(*args, **kwargs):
    if cudaq.mpi.rank() == 0:
        print(*args, **kwargs)

#...!...!..................
def canned_qcrank_inp(args):
    inpF=args.circName+'.qcrank_inp.h5'
    bigD,md=read4_data_hdf5(os.path.join(args.outPath,inpF))

    sd={}
    sd['num_shots']=args.numShotPerAddr*md['payload']['seq_len']
    md['submit']=sd

    return bigD,md

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

#...!...!....................
def harvest_cudaq_backRun_submitMeta(md,args):
    sd=md['submit']
    sd['backend']= args.backend
    t1=localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time())

    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN
    name='cudaq_'+ md['hash']
    md['short_name']=name if args.expName==None else args.expName

def run_cudaq(shots,nc=1):
    # converter  and run circuits one by one
    resL=[0]* nc # prime the list
    for i in range(nc):
        # Convert values to Python int and assign to a, b
        num_qubit, num_gate = map(int,gateD['circ_type'][i] )
        # Convert to list of integers
        gate_type=list(map(int,gateD['gate_type'][i].flatten()))
        gate_param=list(map(float,gateD['gate_param'][i]))
        assert num_gate<=len(gate_param)
        prOn= num_qubit<6 and i==0 or args.verb>1

        if prOn:   print(cudaq.draw(circ_kernel, num_qubit, num_gate, gate_type, gate_param))    

        results = cudaq.sample(circ_kernel,num_qubit, num_gate, gate_type, gate_param, shots_count=shots)
        resL[i]=results  # store  bistsrings
                        
    return resL
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    target = args.backend
    expD,expMD=canned_qcrank_inp(args) 
    
    if args.myRank == 0:
        pprint(expMD)
    numShots=expMD['submit']['num_shots']
    cudaq.set_target(target)
    num_gpus = cudaq.num_available_gpus()
    used_qgus=num_gpus
    if args.verb and args.myRank == 0:
        rank_print('M: use %d of  %d seen gpus'%(used_qgus,num_gpus))
    # ------  circuit generation -------
    qcrankObj=make_qcrank(expMD)
    qcP=qcrankObj.circuit
    nqTot=qcP.num_qubits
    
    rank_print('M: circuit has %d qubits'%(qcP.num_qubits))
    circ_depth_aziz(qcP,text='circ_orig')
    backend = AerSimulator()
    # transpilation i snot needed fro Aer, but we do it here for convenience for CudaQ
    qcT=transpile(qcP,backend, basis_gates=['cx','ry','h']) 
             
    if  qcP.num_qubits<6 and args.myRank == 0:
        rank_print('M:.... PARAMETRIZED TRANSPILED CIRCUIT .............. for',backend)
        rank_print(qcT.draw(output='text',idle_wires=False))  # skip ancilla

    harvest_circ_transpMeta(qcT,expMD,args.backend)
     
    # -------- bind the data to parametrized circuit  -------
    f_data=expD['inp_fdata']
    qcrankObj.bind_data(f_data, max_val=expMD['payload']['qcrank_max_fval'])
    
    # generate the instantiated circuits
    qcEL = qcrankObj.instantiate_circuits()
    nCirc=len(qcEL)
    rank_print('M: execution-ready %d circuits on %d qubits on %s'%(nCirc,nqTot,args.backend))                        

    outD,md=qiskit_to_gateList(qcEL)        
    inpF=os.path.join(args.outPath,args.circName+'.gate_list.h5')
    
    md['short_name'] = args.circName
    if args.myRank == 0:
        write4_data_hdf5(outD,inpF,md)
    gateD,MD=read4_data_hdf5(inpF,args.verb)
    if args.verb: print('M: job %s started, nCirc=%d  nq=%d  shots/circ=%d  on target=%s ...'%(args.circName,nCirc,MD['num_qubit'],args.numShotPerAddr,target))

    T0=time()
    resL=run_cudaq(numShots)
    elaT=time()-T0

    rank_print('RCQ: done', len(resL[0]), args.backend, '\n done %.2fM shots' % (numShots/1e6))

    harvest_cudaq_backRun_submitMeta(expMD,args)
    
    countsL=counts_cudaq_to_qiskit(resL)
    pp0 = countsL[0]
        # collect job performance info
    qa={}
    
    qa['status']='JobStatus.DONE'
    qa['num_circ']=nCirc


    qa['num_clbits']=len(next(iter(pp0.keys())))
    qa['device']='GPU'
    qa['method']='statevector'
    qa['noise']='ideal'
    qa['shots']=numShots
    qa['time_taken']=elaT
    
    if args.myRank == 0:
        rank_print(' job done, elaT=%.1f min'%((elaT)/60.))
        rank_print('M: got results')
        rank_print("counts size: %d"%len(pp0))
        rank_print('job QA'); pprint(qa)
    expMD['job_qa']=qa
    pack_counts_to_numpy(expMD,expD,countsL)

    if args.myRank == 0:
        outF=os.path.join(args.outPath,expMD['short_name']+'.h5')
        write4_data_hdf5(expD,outF,expMD)
    rank_print('   ./postproc_exp.py --expName   %s --showPlots b c   \n'%(expMD['short_name']))
    if args.backend=='nvidia': 
        cudaq.mpi.finalize()
 