#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
 INPUT: circuit defined by gteList , hd5
 Action:
 - opens reads .gate_list.h5
 - qiskit circ  constructed from  getList
   and  run on all CPUs 
 - cudaq-kernel  constructed from  getList
   and  run on all GPUs
 - saves updated HD5
'''

import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5
from toolbox.Util_IOfunc import  write_yaml, dateT2Str,  get_gpu_info, get_cpu_info
import psutil # for w-load
import os
from time import time, sleep
from pprint import pprint
from toolbox.Util_Qiskit import qiskit_circ_gateList
from qiskit_aer import AerSimulator
from toolbox.Util_CudaQ import circ_kernel, qft_kernel
import random
import cudaq

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-e","--expName",  default='mac10q',help='[.gate_list.h5]  defines list of circuits to run')
    parser.add_argument('-n','--numShots',type=int, default=10240, help="(optional) shots per circuit")
    parser.add_argument("-b", "--backend", default="nvidia", choices=['qiskit-cpu','tensornet','nvidia-mgpu','nvidia-mqpu','nvidia','qpp-cpu'], help="cudaQ target settings")
    parser.add_argument('-q','--qft',type=int, default=0, help="(optional) enable qft circuit")

    # IO paths
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath",default='out/',help="input circuits location")
    parser.add_argument("--outPath",default='out/',help="all outputs from experiment")
       
    args = parser.parse_args()
    # make arguments  more flexible
    if args.basePath=='env': args.basePath= os.environ['Cudaq_dataVault']
    if args.basePath!=None:
        args.inpPath=os.path.join(args.basePath,'circ') 
        args.outPath=os.path.join(args.basePath,'meas') 

    # this is complex logic of fetching rank either from srun or mpich
    args.myRank,  args.numRank = 0,1
    if args.backend=='qiskit-cpu':  # use srun ranks
        args.myRank  = int(os.environ['SLURM_PROCID'])
        args.numRank = int(os.environ['SLURM_NTASKS'])
        args.cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        args.tasks_per_node = int(os.environ['SLURM_NTASKS_PER_NODE'])
    if args.backend=='nvidia-mgpu':  # use `mpich -np 4` executed inside podman
        cudaq.mpi.initialize()
        args.myRank = cudaq.mpi.rank()
        args.numRank = cudaq.mpi.num_ranks()
       
                
    if args.myRank==0:
        for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
        if not os.path.exists(args.outPath):
            os.makedirs(args.outPath, exist_ok=True)
    else: args.verb=0 # absolute silence for ranks>0
    assert os.path.exists(args.inpPath)
    
    return args

#...!...!....................
def run_qiskit_aer(shots):
    job=backend.run(qcL,shots=shots)
    #print('   job id:%s , running ...'%job.job_id())
    result=job.result()
    probsBL=result.get_counts()
    return probsBL

#...!...!....................
def run_cudaq(shots,num_qpus,qft=0,nc=1):
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
        input_state = [random.choice([0, 1]) for i in range(num_qubit)]

        if prOn and not qft:   print(cudaq.draw(circ_kernel, num_qubit, num_gate, gate_type, gate_param))    
        if target == "nvidia-mgpu"  or  target == "nvidia":  
            target2='adj-gpu' if target == "nvidia-mgpu" else 'one-gpu'
            if qft:
                results = cudaq.sample(qft_kernel, input_state, shots_count=shots)
            else:
                results = cudaq.sample(circ_kernel,num_qubit, num_gate, gate_type, gate_param, shots_count=shots)
            resL[i]=results  # store  bistsrings
            
        elif target == "nvidia-mqpu":  # 4 GPUs parallel OR 1
            target2='par-gpu'
            gpu_id = i % num_qpus
            if qft:
                results = cudaq.samsample_asyncple(qft_kernel, input_state, shots_count=shots, qpu_id=gpu_id)
            else:
                results = cudaq.sample_async(circ_kernel, num_qubit, num_gate, gate_type, gate_param, shots_count=shots, qpu_id=gpu_id)
            # Retrieve  results - this is where the time is used???
            resL[i]=results.get() # store bistsrings
                        
    print('RCQ: done',len(resL[0]),target2)
    return resL,target2

#...!...!....................
def input_shard(bigD,args):
    if args.verb>0: print('Shard for rank=%d of %d'%(args.myRank,args.numRank))
    totSamp=bigD['circ_type'].shape[0]
    assert totSamp%args.numRank==0
    shardSize=totSamp//args.numRank
    if args.verb>0: print(' select %d-shard of size %d'%(args.myRank,shardSize))
    iOff=args.myRank*shardSize
    for xx in bigD:
        arr=bigD[xx]
        #print(xx,arr.shape)
        bigD[xx]=arr[iOff:iOff+shardSize]
    return shardSize
    

#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
  
    args=get_parser()
    target = args.backend
         
    inpF=args.expName+'.gate_list.h5'
    gateD,MD=read4_data_hdf5(os.path.join(args.inpPath,inpF),args.verb)
    
    if target=='qiskit-cpu':  # use srun ranks
        shardSize=input_shard(gateD,args)
        MD['num_circ']=shardSize
        MD['my_rank']=args.myRank
        MD['num_rank']=args.numRank
        MD['cores']=args.cores
        MD['tasks_per_node']=args.tasks_per_node 
        MD['qft']=args.qft     
        
    nCirc=MD['num_circ']    
    if args.verb>=2:
        print('M:pre MD:');  pprint(MD)
       
    if 'qiskit' in target:
        if args.verb: print('M: will run %d circ on CPUs numRank=%d ...'%(nCirc,args.numRank))
        T0=time()
        qcL=qiskit_circ_gateList(gateD,MD)
        if args.verb: print('\nM:  gen_circ  elaT= %.1f sec '%(time()-T0))
        #....  excution using backRun(.) .....
        backend = AerSimulator()
    else:
        cudaq.set_target(target)
        # only get qpus not gpus
        num_qpus = cudaq.get_target().num_qpus()
        used_qpus=num_qpus if  target == "nvidia-mqpu" else 1
        if args.verb: print('M: use %d of  %d seen qpus'%(used_qpus,num_qpus))
                
    shots=args.numShots
    if args.verb: print('M: job %s started, nCirc=%d  nq=%d  shots/circ=%d  on target=%s ...'%(MD['short_name'],nCirc,MD['num_qubit'],shots,target))
        
    T0=time()
    if 'qiskit' in target:
        resL=run_qiskit_aer(shots)
        MD['cpu_info']=get_cpu_info(verb=0)
        target2='par-cpu'
    else:
        resL,target2=run_cudaq(shots,num_qpus,args.qft)
        MD['num_qpus']=num_qpus
        MD['gpu_info']=get_gpu_info(verb=0)
        

    elaT=time()-T0
    load1, _, _ = psutil.getloadavg()
    if args.verb: print('M:  %s ended elaT=%.1f sec, numRank=%d end_load1=%.1f\n'%(MD['short_name'],elaT,args.numRank,load1))
    MD.update({'elapsed_time':elaT,'target':target,'date':dateT2Str()})
    MD['num_meas_strings']=[ len(x) for x in resL]
    MD['target2']=target2
    MD['short_name']+='_'+target2
    MD['num_shots']=shots
    # add postfix for cpu tasks
    if target2 == 'par-cpu':
        MD['short_name']+='_c'+str(MD['cores'])+'_tp'+str(MD['tasks_per_node'])
    elif target2 == 'adj-gpu':
        MD['short_name']+='_s'+str(MD['num_shots'])
    if args.qft:
        MD['short_name']+='_qft'+str(MD['qft'])
    MD.pop('gate_map')
    MD['cpu_1min_load']=load1
    if args.numRank>1: MD['short_name']+='_r%d.%d'%(args.myRank,args.numRank)
    if args.myRank==0:  # dump some bitstrings
        res0=resL[0]
        for i,bstr in enumerate(res0):
            print('bstr:',bstr,res0[bstr])
            if i >10: break
    
    if target=='nvidia-mgpu':
        print('myRank:%d of %d'%(args.myRank,  args.numRank ))
        # for this case all ranks carry the same information, quit all be rank0
        if args.myRank>0:
            cudaq.mpi.finalize()
            sleep(3)
            exit(0)
 
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.yaml')
    write_yaml(MD,outF)
    
    if args.verb:
        print('M:done  %s  elaT %.1f sec\n'%(MD['short_name'],elaT))
        if 'qiskit' in target:  MD.pop('cpu_info')
        else:  MD.pop('gpu_info')
        pprint(MD)
   
    if target == 'nvidia-mgpu':
        cudaq.mpi.finalize()
  
