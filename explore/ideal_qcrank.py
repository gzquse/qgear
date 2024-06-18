#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Example script that runs QCRANK on the simulator.'''
import sys,os

sys.path.append(os.path.abspath("/daan_qcrank/py"))
from qpixl import qcrank

import numpy as np
from qiskit_aer import AerSimulator
from time import time

import argparse
#...!...!..................
def get_parser(backName=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)

    parser.add_argument('-q','--numQubits', default=[2,2], type=int,  nargs='+', help='pair: nq_addr nq_data, space separated ')

    parser.add_argument('-i','--numImages', default=2, type=int, help='num of images packed in to the job')

    # Qiskit:
    parser.add_argument('-b','--backendName',default=backName,help="backend for computations, always runs against external IBMQ (use fakeSubmit for local comoute)  " )
    parser.add_argument('-n','--numShots', default=800, type=int, help='num of shots')
    parser.add_argument( "-E","--execDecoding", action='store_true', default=False, help="do not decode job output")
 
  
    args = parser.parse_args()
    args.outPath='out/'
    args.provider='IBMQ'

    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.outPath)
    assert len(args.numQubits)==2

    return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer')  #     assert len(args.padSymbols)==2
    
    # set up example parameters ---------------------------------------------------
    n_img = args.numImages               # how many images to process, was 1
    nq_addr, nq_data = args.numQubits  
    max_val = np.pi            # maximum data value
    shots = args.numShots           # number of shots to sample
    keep_last_cx = True     # keep the last cnot or remove
    fake_lima =  False       # T:run on noisy 5q lima simu. or F:noisefree simulator
    qcrank_opt= True     # T: optimal,  F: not optimal w/ cx-gates being parallele
    # ------------------------------------------------------------------------------
    # Derived sizes ---------------------------------------------------------------
    n_addr = 2**nq_addr         # number of different addresses
    n_pix = nq_data * n_addr    # total number of pixels
    # ------------------------------------------------------------------------------

    # generate float random data
    data = np.random.uniform(0, max_val, size=(n_addr, nq_data, n_img))
    if args.verb>2:
        print('input data=',data.shape,repr(data))

    if fake_lima:
        backend = FakeLima()
        assert nq_addr+nq_data <= 5
    else:
        backend = AerSimulator()

    # set up experiments
    param_qcrank = qcrank.ParametrizedQCRANK(
        nq_addr,
        nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=keep_last_cx,
        measure=True,
        statevec=False,
        reverse_bits=True,   # to match Qiskit littleEndian
        parallel= qcrank_opt
    )
    qc=param_qcrank.circuit
    cxDepth=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    print(f'.... PARAMETRIZED CIRCUIT .............. n_pix={n_pix},  qcrank_opt={qcrank_opt}, cx-depth={cxDepth}')
    nqTot=qc.num_qubits
    print(' gates count:', qc.count_ops())

    if args.verb>2 or nq_addr<4:
        print(param_qcrank.circuit.draw())


    if fake_lima:
        param_qcrank.transpile(backend=backend, optimization_level=3)
        print(f'.... TRANSPILED PARAMETRIZED CIRCUIT .............. n_pix={n_pix}')
        print(param_qcrank.circuit.draw())

    # bind the data
    param_qcrank.bind_data(data, max_val=max_val)
    # generate the instantiated circuits
    data_circs = param_qcrank.instantiate_circuits()
    if args.verb>2 or nq_addr<4:
        print(f'.... FIRST INSTANTIATED CIRCUIT .............. of {n_img}')
        print(data_circs[0].draw())

    
    T0=time()    
    # run the simulation for all images
    print('M: job nqTot=%d started ...'%nqTot)
    results = [backend.run(c, shots=shots).result() for c in data_circs]
    counts = [r.get_counts(c) for r, c in zip(results, data_circs)]
    elaT=time()-T0
    print('M: QCrank simu nqTot=%d  shots=%d  nImg=%d  ended elaT=%.1f sec'%(nqTot,shots,n_img,elaT))
    
    if not args.execDecoding:
        print('NO evaluation of job output, use -E to execute decoding')
        exit(0)

    # decode results
    angles_rec =  param_qcrank.decoder.angles_from_yields(counts)  

    print('\nM: minAngle=%.3f, maxAngle=%.3f  should be in range [0,pi]\n'%(np.min(angles_rec),np.max(angles_rec)))

    data_rec = param_qcrank.decoder.angles_to_fdata(angles_rec, max_val=max_val)

    if args.verb>2:
        print(f'.... ORIGINAL DATA .............. n_img={n_img}')
        for i in range(n_img):
            print(f'org img={i}\n', data[..., i])
        print(f'.... RECONSTRUCTED DATA ..........  n_img={n_img}')
        for i in range(n_img):
            print(f'reco img={i}\n', data_rec[..., i])
            #print(f'reco img={i}\n', angles_rec[..., i]/np.pi*max_val)
        print('.... DIFFERENCE ..............')
        for i in range(n_img):
            print(f'diff img={i}\n', data[..., i] - data_rec[..., i])

    print('....L2 distance = sqrt( sum (res^2)), shots=%d  ndf=%d '%(shots,n_addr))
    for i in range(n_img):
        print('img=%d L2=%.2g'%(i, np.linalg.norm(data[..., i] - data_rec[..., i])))
