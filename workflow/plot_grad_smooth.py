#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from toolbox.Util_H5io4 import  read4_data_hdf5
import os
from toolbox.PlotterBackbone import PlotterBackbone
from pprint import pprint

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("--expName",  default='ibm_q7rmac',help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument("--outPath",default='out/',help="all outputs from  experiment")
    
    args = parser.parse_args()
    # make arguments  more flexible
   
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args

    
#...!...!....................
def plot_org_func(ax,md, expD):    
    pprint(md)
    pmd=md['payload']
    smd=md['submit']
    tmd=md['qiskit_transp']
    imgOp=pmd['image_op']
    
    ofunc=expD['org_func']
    xBins=ofunc[0]
    
    ax.step(xBins,ofunc[1], c='blue', where='mid', label=r' f(x)')
    if imgOp=='grad':    
        ax.step(xBins,ofunc[2], c='r', where='mid',label=r"$\nabla$ f(x) ")

    if imgOp=='conv':    
        ax.step(xBins,ofunc[2], c='g', where='mid',label=" g(x) ")
        ax.step(xBins,ofunc[1]*ofunc[2], '--',c='r', where='mid',label=" f(g)*g(x) ")

    if imgOp=='smooth':    
        halfNoise=pmd['noise_ampl_half']
        # Adding a yellow band around the step function
        xBins2=np.hstack([xBins[0],xBins])
        ofunc2=np.hstack([ofunc[1][0],ofunc[1]])
        ax.fill_between(xBins2, ofunc2-halfNoise, ofunc2+halfNoise, step='post', color='yellow', alpha=0.8,label='noise %.1f'%halfNoise)
        
    tit='Ground truth, seq_len: %d '%(pmd['seq_len'])
    ax.set(title=tit,ylabel='arb. uni.')
    ax.grid(True)
    ax.legend(loc='center left')
    ax.axhline(0,ls='--')

    #...  aux info about the job
    if 'noise_model' in smd:
        backN='fake %s'%smd['noise_model']
    else:
        backN=smd['backend']
    txt=''
    txt+='\n%s , op=%s'%(backN,imgOp)
    txt+='\nshots/addr: %d, images: %d'%(smd['num_shots']/pmd['seq_len'],pmd['num_sample'])
    txt+='\nshots/img : %d'%smd['num_shots']
    txt+='\nseq len: %d, qubits: %d'%(pmd['seq_len'],tmd['num_qubit'])
    txt+='\n2q gates: %d, depth: %d'%(tmd['num_2q'],tmd['2q_depth'])
    
    ax.text(0.45,0.05,txt,transform=ax.transAxes,color='m')

    return

#...!...!....................
def plot_reco_grad(ax,md,expD,im=0):    
    pmd=md['payload']
    smd=md['submit']
    #tmd=md['qiskit_transp']

    ofunc=expD['org_func']
    xBins=ofunc[0]
    mCol='g'
    if im==0:
        tfunc=ofunc[2]  # gradient
        tdLab=r"$true  ~\nabla$ f(x)"
        mCol='r'
    if im==1:
        tfunc=-ofunc[1]  # -f(z)
        tdLab='true -f(x) '
    if im==2:
        tfunc=np.zeros_like(xBins)
        tdLab='true 0-value '
        
    ureco=expD['u_reco']
    rfunc0=ureco[:,0,im]
    
    ax.step(xBins,rfunc0, 'o',c='k',where='mid',label=r'meas')
    ax.step(xBins,tfunc, c=mCol, where='mid',label=tdLab)
    spa=smd['num_shots']/pmd['seq_len']
    tit='Measured op=%s , %d shots/addr %s'%(pmd['image_op'],spa,md['short_name']) 
    ax.set(title=tit,ylabel='meas expectation value')
    ax.grid(True)
    ax.legend(loc='upper left')
    if im!=2: ax.axhline(0,ls='--')

    return
 
    #...  info in middle column
    txt=''
    if 'noise_model' in smd:
        txt+='\nfake : %s'%(smd['noise_model'])
    
    ax=axL[0,1]
    ax.text(0.10,0.1,txt,transform=ax.transAxes,color='m')

#...!...!....................
def plot_reco_smooth(ax,md,expD,im=0):    
    pmd=md['payload']
    smd=md['submit']
    qid=1  # for output_sum
       
    ofunc=expD['org_func']
    xBins=ofunc[0]  # x
    tfunc=ofunc[1]  # f(x)

    uinp=expD['u_input'][...,im]
    ureco=expD['u_reco']
    rfunc0=ureco[:,qid,im]
    
    ax.step(xBins,rfunc0, 'o',c='k',where='mid',label=r'meas')
    ax.step(xBins,tfunc, c='r', where='mid',label='true f(x)')

    lCol=['b','g']
    for i in range(2):
        ax.step(xBins,uinp[:,i],'--' ,c=lCol[i],where='mid',label='inp %c'%(97+i))
    spa=smd['num_shots']/pmd['seq_len']
    tit='Measured op=%s , %d shots/addr %s'%(pmd['image_op'],spa,md['short_name']) 
    ax.set(title=tit,ylabel='meas expectation value')
    ax.grid(True)
    ax.legend(loc='upper left',title='image=%d'%im)
    ax.axhline(0,ls='--')

    return
     
#...!...!....................
def plot_reco_conv(ax,md,expD,im=0):    
    pmd=md['payload']
    smd=md['submit']
    qid=1  # for output_prod
    ofunc=expD['org_func']
    xBins=ofunc[0]  # x
    tfunc1=ofunc[1]  # f(x)
    tfunc2=ofunc[2]  # g(x)
    if im==0:
        s1=1; s2=1; tLab='true f(x) * g(x)'
    if im==1:
        s1=-1; s2=1; tLab='true [-f(x)] * g(x)'
    if im==2:
        s1=1; s2=-1; tLab='true  f(x) * [-g(x)]'            
    tconv=s1*s2*tfunc1*tfunc2
    
    
    uinp=expD['u_input'][...,im]
    ureco=expD['u_reco']
    rfunc=ureco[:,qid,im]
    #print('rfunc:',rfunc)
    
    ax.step(xBins,rfunc, 'o',c='k',where='mid',label=r'measured')
    ax.step(xBins,tconv, c='r', where='mid',label=tLab)

    lCol=['b','g']
    #for i in range(2):
    #    ax.step(xBins,uinp[:,i],'--' ,c=lCol[i],where='mid',label='inp %c'%(97+i))
    spa=smd['num_shots']/pmd['seq_len']
    tit='Meas img=%d op=%s , %d shots/addr %s'%(im,pmd['image_op'],spa,md['short_name']) 
    ax.set(title=tit,ylabel='meas expectation value')
    ax.grid(True)
    ax.legend(loc='upper center')
    ax.axhline(0,ls='--')

    return
     
    
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__": 
    args=get_parser()
    np.set_printoptions(precision=2)

    inpF=args.expName+'.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.outPath,inpF))
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print('M:expD counts sample:');  print(expD['counts_raw'][0][:20],'sum=', np.sum(expD['counts_raw'][0]))
        stop2

    
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    plot=PlotterBackbone(args)
    axL2=plot.blank_separate2D(nrow=2,ncol=2, figsize=(12,8),figId=2)
   
    plot_org_func(axL2[0,0],expMD,expD)
        
    imgOp=expMD[ 'payload']['image_op']
    axL=[axL2[0,1],axL2[1,0],axL2[1,1]]
    
    for im in range(3):    
        if imgOp=='grad': plot_reco_grad(axL[im],expMD,expD,im)
        if imgOp=='smooth': plot_reco_smooth(axL[im],expMD,expD,im)
        if imgOp=='conv': plot_reco_conv(axL[im],expMD,expD,im)
        
    plot.display_all()
    print('M:done')

  
