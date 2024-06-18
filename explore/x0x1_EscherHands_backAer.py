#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Functionality:
--input_x  takes 2 real number in range [-1,1]
--weight  takes 1 real number in range [0,1]
--subtract  is binary switch

Output:   
  y0=w*x0 + (1-w)*x1  OR  y=w*x0 - (1-w)*x1 
  y1=x0*x1  regardless on the 'w' 

Summary:
compute on QPU a  gradient and  average for a pair of real values stored as sqrt(amplitudes) on 2 qubits, w/o using any ancilla.

notes for sum:
https://docs.google.com/document/d/1yiKcocSj7qq2Paa9DiR2W_f5eYqml5IOBfwyC0k7bUc/edit?usp=sharing

notes for difference:
https://docs.google.com/document/d/17imQIXXlJdt3-nkfXjxYvVlrI5C7f7rBJJO4QI8PToo/edit?usp=sharing

     ┌──────────┐ ░                 ┌─────────┐ ┌───┐┌────────────┐ ░ ┌─┐   
q_0: ┤ Ry(θ[0]) ├─░──────────────■──┤ Ry(α/2) ├─┤ X ├┤ Ry(-0.5*α) ├─░─┤M├───
     ├──────────┤ ░ ┌─────────┐┌─┴─┐├─────────┴┐└─┬─┘└────────────┘ ░ └╥┘┌─┐
q_1: ┤ Ry(θ[1]) ├─░─┤ Rz(π/2) ├┤ X ├┤ Rz(-π/2) ├──■─────────────────░──╫─┤M├
     └──────────┘ ░ └─────────┘└───┘└──────────┘                    ░  ║ └╥┘
c: 2/══════════════════════════════════════════════════════════════════╩══╩═
                                                                       0  1 

'''

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Parameter,ParameterVector
from qiskit_aer import AerSimulator
from qiskit.result.utils import marginal_distribution
from qiskit.visualization import circuit_drawer
from time import time, sleep
import numpy as np
import os

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_shot',type=int,default=4000, help="shots")

    parser.add_argument('-w','--weight', default=0.5, type=float, help='weight of 1st value')
    parser.add_argument('-s','--subtract', action='store_true', help='subtract instead of add ')
    parser.add_argument('-x','--input_x', default=[0.3,0.9], nargs=2, type=float, help='pair of input values')
    parser.add_argument( "-r","--randomData",   action='store_true', default=False, help="(optional)replace all inputs by random values")
     
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
   

    if args.randomData:
        print('Random inputs')
        eps=0.005
        args.input_x=np.random.uniform(-1.+eps, 1.-eps, size=len(args.input_x))
        args.weight=np.random.uniform(eps, 1.-eps)


   
    return args

#...!...!....................
def data_2_angles(xV,W):
    nx=2
    assert len(xV)==nx
    assert min(xV) >=-1
    assert max(xV) <=1
    assert W >=0
    assert W <=1
    
    thetaV=np.arccos(xV)
    alpha=np.arccos(1.- 2*W)
    print('D2A: xV:',xV,' W:',W)
    print('D2A: thV:',thetaV,' alpha:',alpha)
    return thetaV,alpha

#...!...!....................
def circ_EscherHand_2q(doSub=False):  # true: for difference
    ''' computes:
            EV[p(q0)]= (x0 + x1)/2
            EV[p(q1)]= x0 * x1
    '''
    # Define the parameters alpha and theta
    alpha = Parameter('α')
    thetas = ParameterVector('θ',length=2)
    nq=2
    # Create a Quantum Circuit with 2 qubits and 2 classical bits for measurement
    qc = QuantumCircuit(nq, nq)
    
    # feature map:
    for i in range(nq): qc.ry(thetas[i], i)

    # superoperator computing weighted sum
    qc.barrier()
    
    # ... ctrl(0)-Rz(pi,1) ...
    qc.rz(np.pi/2,1)
    qc.cx(0,1)
    #Xqc.rz(-np.pi/2,1)
    
    # ... ctrl(1)-Ry(alpha,0)
    qc.ry(alpha/2,0)
    if doSub: qc.x(1)
    qc.cx(1,0)
    if doSub: qc.x(1)
    qc.ry(-alpha/2,0)
    qc.barrier()

    # Measure qubit 1 into classical bit 1
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc, thetas, alpha

#...!...!....................
def marginal_prob(counts,qid):   # qid qubit index
    n0=0;n1=0
    # measured bits are LSBF
    for key,val in counts.items():
        # Reverse the bit string to convert from MSBF to LSBF
        keyr=key[::-1]
        mbit=keyr[qid]
        if mbit=='1': n1+=val
        else: n0+=val
    mprob=n1/(n0+n1)
    #print('n0,n1',n0,n1,'mprob=',mprob)
    return mprob


#...!...!....................
def ana_exp_sum(counts,doSub=False, qid=0):  # qubit index
     
    mprob=marginal_prob(counts,qid)
    xV,w=args.input_x,args.weight
    print('\nInput: xV:',xV,' w:',w)
    wV=[w,1-w]
    if not doSub:
        tsum= wV[0]*xV[0] + wV[1]*xV[1]
        print('true add : y= + %.1f * %.2f  +  %.1f * %.2f   ==>  y=%.2f'%(wV[0],xV[0],wV[1],xV[1],tsum))
        qStr='  sum q%d'%qid
    else:
        tsum= wV[0]*xV[0] - wV[1]*xV[1]
        print('true subtract: y=  %.1f * %.2f  -  %.1f * %.2f   ==>  y=%.2f'%(wV[0],xV[0],wV[1],xV[1],tsum))
        qStr='  sub q%d'%qid
    msum=1-2*mprob
    diff=tsum-msum
  
    if  abs(diff)<0.03 : okStr='--- PASS ---'
    elif  abs(diff)<0.1  : okStr='.... poor ....' 
    else: okStr='*** FAILED ***'

    okStr+=qStr
    
    print('Eval:  prob=%.3f, tEV=%.3f,  mEV=%.3f   delta=%.3f    %s\n'%(mprob,tsum,msum,diff,okStr))
  

#...!...!....................
def ana_exp_prod(counts,qid=1):  # qubit index
    mprob=marginal_prob(counts,qid)
    xV=args.input_x
    print('\nInput: xV:',xV)
    tprod=xV[0] *xV[1]
    print('true prod : y=  %.2f  *  %.2f   ==>  y=%.2f'%(xV[0],xV[1],tprod))
    qStr='  prod q%d'%qid
    mprod=1-2*mprob
    diff=tprod-mprod
  
    if  abs(diff)<0.03 : okStr='--- PASS ---'
    elif  abs(diff)<0.1  : okStr='.... poor ....' 
    else: okStr='*** FAILED ***'

    okStr+=qStr
    
    print('Eval:  prob=%.3f, tEV=%.3f,  mEV=%.3f   delta=%.3f    %s\n'%(mprob,tprod,mprod,diff,okStr))
  

    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()

    thetas_data,alpha_data=data_2_angles(args.input_x,args.weight)
    doSub=args.subtract
    qcP, thetasP, alphaP=circ_EscherHand_2q(doSub)
    
    print(circuit_drawer(qcP, output='text'))
    
    if 0: # Draw the circuit using the latex source
        latex_source = circuit_drawer(qcP, output='latex_source',cregbundle=False)
        print(latex_source)
        just_latex

    shots=args.num_shot
    backend = AerSimulator()
    
    param_dict = {alphaP: alpha_data, thetasP: thetas_data}
    # Bind the values to the circuit
    qcE = qcP.assign_parameters(param_dict)

    print(qcE)
    qcT = transpile(qcE, backend=backend, optimization_level=3, seed_transpiler=42)
  
    print('job started, nq=%d at'%qcT.num_qubits,backend)
    T0=time()
    job=backend.run(qcE,shots=shots, dynamic=True)
    result=job.result()
    elaT=time()-T0
   
    print('M: run ended elaT=%.1f sec'%(elaT))
    ic=0
    counts=result.get_counts(ic)       
    print('M:counts:%s'%(counts))

    ana_exp_sum(counts,doSub)
    ana_exp_prod(counts)

    # https://qiskit.org/documentation/stubs/qiskit.visualization.circuit_drawer.html
    
    print('M: done, shots:',shots,backend)
