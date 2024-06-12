#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
encode 2 sequnces of real numbers with  QCrank using  nq_data=2 qubits and common address qubits


compute vector : averag or difference 
"--mathOp", choices=['add','sub','none'],
(optional) change relative weights

 backend: aer
uses backRun

'''
import numpy as np
from pprint import pprint
import os
from time import time, localtime
from toolbox.Util_IOfunc import dateT2Str
from qiskit import  transpile

from  Util_EscherHands_ver0  import run_qcrank_only, circ_qcrank ,circ_qcrank_and_EscherHands_one,  marginalize_EscherHands_EV
from qiskit.visualization import circuit_drawer
from qiskit_aer import AerSimulator

from toolbox.Util_H5io4 import  write4_data_hdf5
from toolbox.Util_Qiskit import  circ_depth_aziz

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    # .... QCrank
    parser.add_argument("--nqAddr",  default=2, type=int, help='size of address registers')
    parser.add_argument('-i','--numSample', default=50, type=int, help='num of images packed in to the job')
 
    #.... Escher-Hands math
    parser.add_argument('-w','--weight', default=0.5, type=float, help='weight of 1st value')
    parser.add_argument("-m", "--mathOp",  default='add',choices=['add','sub','none'], help=" operation on data lines ")

    # .... job running
    parser.add_argument('-n','--numShotPerAddr',type=int,default=8000, help="shots per address of QCrank")

    parser.add_argument("--expName",  default=None,help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument('-b','--backend',default="aer", help="tasks")
    parser.add_argument( "-F","--fakeSimu", action='store_true', default=False, help="will switch to backend-matched simulator")
    parser.add_argument('--spam_corr',type=int, default=1, help="Mitigate error associated with readout errors, 0=off")    
   
    parser.add_argument( "-B","--noBarrier", action='store_true', default=False, help="remove all bariers from the circuit ")
    parser.add_argument("--outPath",default='out/',help="all outputs from  experiment")
    
    args = parser.parse_args()
    # make arguments  more flexible
    parser.add_argument('--readMit',type=int, default=0, help="Mitigate readout errors, 0=off")
    args = parser.parse_args()
    # make arguments  more flexible
    if 'ibm' in args.backend:
        args.provider='IBMQ_cloud'
    else:
        args.provider='local_sim'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    assert args.readMit==0 # not used, see paper_Ehands_appl/aux/submit_ibmq_sampler.py
    assert  args.backend=='aer'
    return args

#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    pd['nq_addr']=args.nqAddr
    pd['nq_fdata']=2   # two real numbers per address are needed by EscherHands
    pd['num_sample']=args.numSample
    pd['ehand_weight']=args.weight  # sum= w*x1 + (1-w)*x2
    pd['ehand_math_op']=args.mathOp
    pd['max_fval']=np.pi
    pd['seq_len']=1<<args.nqAddr
    md={ 'payload':pd}
    md['short_name']=args.expName

    if args.verb>1:  print('\nBMD:');pprint(md)            
    assert args.weight >=0
    assert args.weight <=1    
    return md

#...!...!....................
def harvest_ibmq_submitMeta(job,md,args):
    sd={}
    sd['job_id']=job.job_id()
    sd['backend']=args.backend #job.backend().name
    sd['num_shots']=args.numShots
    sd['read_err_mit']=args.readMit
    t1=localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time())
    sd['provider']=args.provider
    md['submit']=sd

    if args.expName==None:
        # the  6 chars in job id , as handy job identiffier
        md['hash']=sd['job_id'].replace('-','')[3:9] # those are still visible on the IBMQ-web
        name='exp_'+md['hash']
        #if md['backend']['ideal_simu']: name='isim_'+md['hash']
        md['short_name']=name
    else:
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
        md['short_name']=args.expName

#...!...!....................
def construct_random_input(md):
    pmd=md['payload']
    n_addr=pmd['seq_len']
    nq_data=pmd['nq_fdata'] ; assert nq_data==2
    n_img=pmd['num_sample']
    pprint(md)
    
    # generate user data , float random
    eps=1e-4 # just in case
    udata = np.random.uniform(-1.+eps, 1.-eps, size=(n_addr, nq_data, n_img))

    if 0:  # testing edge cases
        idata=np.array([5]) # tmp
        assert udata.shape==(4,2,1)
        xdata=np.array([[0.7,0.0,0,0],[0.0,-0.4,0,0]]).T
        #print('ss1',udata.shape, xdata.shape)
        udata=xdata[:, :, np.newaxis]
        print(xdata,udata.shape)

    fdata=np.arccos(udata) # encode user data for qcrank
    print('input udata=',udata.shape,'\n',repr(udata[...,:3].T))
    return udata,fdata

#...!...!....................
def evaluate(probsBL,md,qcrankObj,udata_inp):
    pmd=md['payload']
    smd=md['submit']
    nq_addr=pmd['nq_addr']
    nq_data=pmd['nq_fdata']
    seq_len=pmd['seq_len']
    nSamp=pmd['num_sample']
    
    mathOp=pmd['ehand_math_op']
    W=pmd['ehand_weight']
    udata_true=np.zeros_like(udata_inp)
    udata_rec=np.zeros_like(udata_inp)
    addrBitsL = [nq_data+i  for i in range(nq_addr)]

    assert nq_data==2 # needed by EscherHands
    print('udata',udata_true.shape)
    print('raw inp: '); pprint(udata_inp[...,:3].T)
      
    if mathOp=='none':
        udata_true=udata_inp.copy()
        if 1: # the convoluted way, using qcrank decoder
            # ....  since QCrank maxVal=pi, we get the reco angles already 
            fdata_rec =  qcrankObj.decoder.angles_from_yields(probsBL)
            udata_rec=np.cos(fdata_rec)
            #print('raw1 rec: '); pprint(udata_rec.T)
        else: # quicker way,  output is just in probabilities
            for ic in range( nSamp):  # EscherHands did its job
                udata_rec[:,0,ic]= marginalize_EscherHands_EV( addrBitsL, probsBL[ic], dataBit=0)    
                udata_rec[:,1,ic]= marginalize_EscherHands_EV( addrBitsL, probsBL[ic], dataBit=1)
            #print('raw2 rec: '); pprint(udata_rec.T)
             
    if mathOp=='add' or mathOp=='sub':
        
        for ic in range( nSamp):  # EscherHands did its job
            udata_rec[:,0,ic]= marginalize_EscherHands_EV( addrBitsL, probsBL[ic], dataBit=0)    
            udata_rec[:,1,ic]= marginalize_EscherHands_EV( addrBitsL, probsBL[ic], dataBit=1)                
        udata_true[:,1]=udata_inp[:,0] *  udata_inp[:,1]
        
    # QCrank is reversing order of data qubits somewhere, so here I un-do it by flippin 0/1 in udata_inp
    if mathOp=='add':
        udata_true[:,0]= W*udata_inp[:,1] + (1-W)* udata_inp[:,0] 
 
    if mathOp=='sub':
        udata_true[:,0]= W* udata_inp[:,1] - (1-W)*udata_inp[:,0] 
 
    #.... common    
    print('\nmath=%s rec: '%mathOp); pprint(udata_rec[...,:3].T)        
    print('\nmath=%s true: '%mathOp); pprint(udata_true[...,:3].T)
    res_data=udata_true - udata_rec

    L2=np.linalg.norm(res_data)
    print('mathOp=%s  nCirc=%d L2=%.2g\n'%(mathOp, nCirc,L2))

    c1=L2<0.06
    c2=np.max(np.abs(res_data)) <0.07
    mOK=c2 and c1
    if mOK: print('---- PASS ----  ')
    else:
        print('  *** FAILED **** L2=%r  res=%r'%(c1,c2))
        if nCirc>10: print('for large nCirc it may be just 1 outlier')
        
    print('\ndump 1st circ'); ic=0
    mathOp2='mult'
    if mathOp=='none': mathOp=' x1 '; mathOp2=' x2 '
    print('summary mathOp=%s  W=%.2f  L2=%.2f shots=%d backend=%s\n i  input: x1      x2      true(%s)   meas(%s)   res(%s)     true(%s)  meas(%s)   res(%s)'%(pmd['ehand_math_op'],W,L2,smd['num_shots'],smd['backend'],mathOp,mathOp,mathOp,mathOp2,mathOp2,mathOp2))
    for i in range(seq_len):
        print('%2d    %6.2f    %6.2f      %6.2f      %6.2f      %6.2f         %6.2f      %6.2f      %6.2f'%\
              (i,udata_inp[i,0,ic],udata_inp[i,1,ic],udata_true[i,0,ic],udata_rec[i,0,ic],res_data[i,0,ic],udata_true[i,1,ic],udata_rec[i,1,ic],res_data[i,1,ic]))

    return udata_true,udata_rec,res_data
  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    MD=buildPayloadMeta(args)
    args.numShots=args.numShotPerAddr*MD['payload']['seq_len']

    # pprint(MD)

    u_data,f_data= construct_random_input(MD)
    
    if 0:  # test only QCrank
        run_qcrank_only(f_data,MD,sampler)
        exit(0)

    #....  circuit generation .....
    qcrankObj,qcEL=circ_qcrank_and_EscherHands_one(f_data, MD,barrier=not args.noBarrier)
    qc1=qcEL[0]
    print('M: circuit has %d qubits'%(qc1.num_qubits))
    circ_depth_aziz(qc1,text='circ_orig')
    if args.verb>0: print(circuit_drawer(qc1.decompose(), output='text',cregbundle=True))

    #....  excution using backRun(.) .....
    print('M: acquire backend:',args.backend)
       
    print('M: acquire backend:',args.backend)
    backend = AerSimulator()
    qcTL =qcEL
    
    qc1=qcTL[0];  nCirc=len(qcTL)
    print(qc1.draw(output='text',idle_wires=False))  # skip ancilla
    depthTC,opsTC=circ_depth_aziz(qc1,'transpiled')
        
    #... auxil MD , filled partially
    MD['submit']={'backend': backend.name,'num_circ':nCirc}
    MD[ 'qiskit_transp']={'num_qubit': qcEL[0].num_qubits,'2q_depth':depthTC['2q'],'num_2q':opsTC['cx']}
       
    print('job started, nCirc=%d  nq=%d  shots/circ=%d at %s ...'%(nCirc,qc1.num_qubits,args.numShots,backend))
    T0=time()
    job=backend.run(qcTL,shots=args.numShots)
    print('   job id:',job.job_id())
    result=job.result()
    elaT=time()-T0
    print('M:  ended elaT=%.1f sec'%(elaT))
    jobMD=result.metadata    
    
    harvest_ibmq_submitMeta(job,MD,args)
    if args.fakeSimu: MD['submit'][ "noise_model"]=noisy_backend.name
     
    probsBL=result.get_counts()#result.quasi_dists
    if args.verb>1: print('M:qprobs:%s'%(probsBL[0]))
    
    u_true,u_reco,res_data=evaluate(probsBL,MD,qcrankObj,u_data)

    pprint(MD)
    expD={'u_input': u_data,'u_true':u_true,'u_reco':u_reco}
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,MD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,MD)

    print('\n   ./plot_EscherHands.py  --expName %s  -Y '%(MD['short_name']))
