__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from time import time, localtime
from pprint import pprint
from qiskit.result.utils import marginal_distribution
import numpy as np

from qgear.toolbox.Util_Qiskit import pack_counts_to_numpy
from qgear.toolbox.Util_IOfunc import dateT2Str

#...!...!....................
def harvest_ibmq_backRun_submitMeta(job,md,args):
    sd=md['submit']
    sd['job_id']=job.job_id()
    sd['backend']= job.backend().name # V2

    t1=localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time())
    #sd['provider']=args.provider
    #sd['api_type']='circuit-runner' #IBM-spceciffic , the  alternative is 'sampler'

    if args.expName==None:
        # the  6 chars in job id , as handy job identiffier
        md['hash']=sd['job_id'].replace('-','')[3:9] # those are still visible on the IBMQ-web
        name='ibm_'+md['hash']
        md['short_name']=name
    else:
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
        md['short_name']=args.expName


#...!...!....................
def harvest_backRun_results(job,md,bigD):  # many circuits
    jobRes = job.result()
    resL=jobRes.results
    nCirc=len(resL)  # number of circuit in the job

    #print('jr:'); pprint(jobRes)
    #1qc=job.circuits()[ic]  # transpiled circuit
    #1ibmMD=jobRes.metadata ; print('tt nC',nCirc,type(ibmMD))

    #nqc=len(resL)  # number of circuit in the job
    countsL=jobRes.get_counts()
    jstat=str(job.status())
    res0=resL[0]
    if nCirc==1:
        countsL=[countsL]  # this is poor design

    #print('ccc1b',countsL[0])
    #print('meta:'); pprint(res0._metadata)

    # collect job performance info
    qa={}

    qa['status']=jstat
    qa['num_circ']=nCirc

    try :
        ibmMD=res0.metadata
        for x in ['num_clbits','device','method','noise']:
            #print(x,ibmMD[x])
            qa[x]=ibmMD[x]

        qa['shots']=res0.shots
        qa['time_taken']=res0.time_taken
    except:
        print('MD1 partially missing')

    if 'num_clbits' not in qa:  # use alternative input
        head=res0.header
        #print('head2');  pprint(head)
        qa['num_clbits']=len(head.creg_sizes)

    print('job QA'); pprint(qa)
    md['job_qa']=qa
    pack_counts_to_numpy(md,bigD,countsL)
    return bigD


#...!...!....................
def marginalize_qcrank_EV(  addrBitsL, probsB,dataBit):
    #print('MQCEV inp bits:',dataBit,addrBitsL)
    # ... marginal distributions for 2 data qubits, for 1 circuit
    assert dataBit not in addrBitsL
    bitL=[dataBit]+addrBitsL
    #print('MQCEV bitL:',bitL)
    probs=marginal_distribution(probsB,bitL)

    #.... comput probabilities for each address
    nq_addr=len(addrBitsL)
    seq_len=1<<nq_addr
    mdata=np.zeros(seq_len)
    fstr='0'+str(nq_addr)+'b'
    for j in range(seq_len):
        mbit=format(j,fstr)
        mbit0=mbit+'0'; mbit1=mbit+'1'
        m1=probs[mbit1] if mbit1 in probs else 0
        m0=probs[mbit0] if mbit0 in probs else 0
        m01=m0+m1
        #print(j,mbit,'sum=',m01)
        p=m1/m01 if m01>0 else 0
        mdata[j]=p
    return 1-2*mdata


#...!...!....................
def harvest_circ_transpMeta(qc,md,transBackN):
    nqTot=qc.num_qubits

    try:  # works for qiskit  0.45.1
        physQubitLayout = qc._layout.final_index_layout(filter_ancillas=True)
    except:
        physQubitLayout =[ i for i in range(nqTot)]

    #print('physQubitLayout'); print(physQubitLayout)

    #.... cycles & depth ...
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )

    #.....  gate count .....
    opsOD=qc.count_ops()  # ordered dict
    opsD={ k:opsOD[k] for k in opsOD}

    n1q_g=0
    for xx in ['ry','h','r','u2','u3']:
        if xx not in opsD: continue
        n1q_g+=opsD[xx]

    n2q_g=0
    for xx in ['cx','cz','ecr']:
        if xx not in opsD: continue
        n2q_g+=opsD[xx]

    #... store results
    tmd={'num_qubit': nqTot,'phys_qubits':physQubitLayout}
    tmd['transpile_backend']=transBackN

    tmd['2q_gate_depth']=len2
    tmd['1q_gate_count']=n1q_g
    tmd[ '2q_gate_count']= n2q_g
    md['transpile']=tmd

    md['payload'].update({'num_qubit':nqTot , 'num_clbit':qc.num_clbits})
    print('circ_transpMeta:'); pprint(tmd)
