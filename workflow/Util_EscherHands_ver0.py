import numpy as np
from pprint import pprint

from qiskit import QuantumCircuit
import sys,os
from time import time
from qiskit.result.utils import marginal_distribution

sys.path.append(os.path.abspath("/daan_qcrank/py"))
from qpixl import qcrank

#...!...!....................
def circ_qcrank_and_EscherHands_one(fdata,md, barrier=True):
    # stores only 1 Ehands block
    pmd=md['payload']
    nq_addr=pmd['nq_addr']
    nq_data=pmd['nq_fdata']
    mathOp=pmd['ehand_math_op']
    assert mathOp in ['add','sub','none']

    pprint(md)
    qcrankObj,qcL=circ_qcrank(fdata,md,barrier=barrier, measure=False)    

    for qc in qcL:
        if barrier: qc.barrier()
    
        if mathOp!='none':  # .... append EscherHands circuit
            alpha=np.arccos( 1- 2* pmd['ehand_weight'])
            qA=0     # assign adder qubit
            qM=qA+1     # assign multiplier qubit
            assert nq_data==2
            
            # ... ctrl(0)-Rz(pi,1) ...
            qc.rz(np.pi/2,qM)
            qc.cx(qA,qM)
            qc.rz(-np.pi/2,qM)
            
            # ... ctrl(1)-Ry(alpha,0)
            qc.ry(alpha/2,qA)
            if mathOp=='sub': qc.x(qM)  
            qc.cx(qM,qA)
            if mathOp=='sub': qc.x(qM)
            qc.ry(-alpha/2,qA)
            if barrier: qc.barrier()
        
        qc.measure_all()
    return qcrankObj,qcL



#...!...!....................
def make_qcrankObj( md,barrier, measure):
    pmd=md['payload']
    nq_addr=pmd['nq_addr']
    nq_data=pmd['nq_fdata']
    # create parameterized circ
    qcrankObj = qcrank.ParametrizedQCRANK(
        nq_addr, nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=True, barrier=barrier,
        measure=measure, statevec=False, 
        reverse_bits=True  # to match Qiskit littleEndian 
    )
    return qcrankObj

#...!...!....................
def circ_qcrank(data_org, md,barrier=True, measure=True):

    qcrankObj=make_qcrankObj( md,barrier, measure)
    pmd=md['payload']
    nq_addr=pmd['nq_addr']
    nq_data=pmd['nq_fdata']
    max_fval=pmd['max_fval']
    #print('www',data_org.T)
    assert np.min(data_org)>=0
    assert np.max(data_org)<=max_fval
    
 

    if 0:
        print('.... PARAMETRIZED IDEAL CIRCUIT .............. num_addr=%d'%(1<<nq_addr))
        print(qcrankObj.circuit.draw())
  
    # bind the data
    #print('dorg:',data_org.shape)
    qcrankObj.bind_data(data_org, max_val=max_fval)

    # generate the instantiated circuits
    circs = qcrankObj.instantiate_circuits()

    if 0:
        print('.... FIRST INSTANTIATED CIRCUIT .............. of %d'%(len(circs)))
        print(circs[0])
    return qcrankObj,circs


#...!...!....................
def run_qcrank_only(f_data,MD,sampler):        
        qcrankObj,qcL=circ_qcrank(f_data,MD)
        
        T0=time()
        job = sampler.run(qcL) 
        result=job.result()
        jobMD=result.metadata
        nCirc=MD['payload']['num_sample']
        qprobsL=result.quasi_dists
        
        for ic in range(nCirc):
            if ic<3: print('ic=%d job-meta:'%ic,jobMD[ic],' nclbit=%d'%qcL[0].num_clbits)
            
          
        print('RQO:qprobsL0:%s'%(qprobsL[0]))
        probsBL=measL_int2bits(qprobsL,qcL[0].num_clbits)
        # decode results
        angles_rec =  qcrankObj.decoder.angles_from_yields(probsBL)

        print('\nM: minAngle=%.3f, maxAngle=%.3f  should be in range [0,pi]\n'%(np.min(angles_rec),np.max(angles_rec)))
        # .... next step does nothing, since maxVal=pi 
        max_fval=MD['payload']['max_fval']
        data_rec = qcrankObj.decoder.angles_to_fdata(angles_rec, max_val=max_fval)
        #print('ss',f_data.shape,  angles_rec.shape,data_rec.shape)
        
        img=0
        print(f'.... ORIGINAL DATA ..............')
        print(f'org img={img}\n', f_data[..., img])
        print(f'reco img={img}\n', data_rec[..., img])
        print('.... DIFFERENCE ..............')
        print(f'diff img={img}\n', f_data[..., img] - data_rec[..., img])
        n_addr=1<<MD['payload']['nq_addr']

        shots=result.metadata[ic]['shots']
        print('....L2 distance = sqrt( sum (res^2)), shots=%d  ndf=%d  max_fval=%.1f'%(shots,n_addr,max_fval))

        for img in range(0,nCirc):
            print('img=%d L2=%.2g'%(img, np.linalg.norm(f_data[..., img] - data_rec[..., img])))

#...!...!....................
def marginalize_EscherHands_EV(  addrBitsL, probsB,dataBit):
    # ... marginal distributions for 2 data qubits, for 1 circuit
    bitL=[dataBit]+addrBitsL
    #print('MEH bitL:',bitL)
    probs=marginal_distribution(probsB,bitL)
    #print(bitL,'aaa1',probs)
    nq_addr=len(addrBitsL)
    seq_len=1<<nq_addr
    mdata=np.zeros(seq_len)
    for j in range(seq_len):
        mbit=format(j,'0'+str(nq_addr)+'b')
        #print(j,mbit)
        mbit0=mbit+'0'; mbit1=mbit+'1'
        m1=probs[mbit1] if mbit1 in probs else 0
        m0=probs[mbit0] if mbit0 in probs else 0
        p=m1/(m1+m0)
        #print(j,mbit1,m1,m0,p,1-2*p)
        mdata[j]=p
    return 1-2*mdata

