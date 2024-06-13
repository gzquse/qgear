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


    parser.add_argument("--expName",  default='exp_i14brq',help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument("--outPath",default='out/',help="all outputs from  experiment")
    
    args = parser.parse_args()
    # make arguments  more flexible
   
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args


#...!...!....................
def compute_correlation_and_draw_line(ax, x_data, y_data,xLR=[]):
    """Compute correlation and draw a line at the angle of correlation."""
    correlation = np.corrcoef(x_data, y_data)[0, 1]

    # Line representing correlation - slope based on correlation
    # y = mx + c, where m is the correlation coefficient
    # We pass through the mean of the points for the line of best fit
    mean_x, mean_y = np.mean(x_data), np.mean(y_data)
    ax.plot(mean_x,mean_y,'Dr')
    m = correlation * np.std(y_data) / np.std(x_data)
    c = mean_y - m * mean_x

    # Points for the line
    if len(xLR)==0:
        x = np.array([min(x_data), max(x_data)])
    else:
        x=np.array(xLR)
    y = m * x + c

    ax.plot(x, y, 'r--',label='correlation')

    th=np.arctan(m)
    txt='correl: %.2f,   theta %.0f deg'%(correlation,th/np.pi*180)
    ax.text(0.05,0.78,txt,transform=ax.transAxes,color='r')
    
    
    ax.set_xlim(-1.05,1.05)
    ax.grid(True)
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

    
    return 

#...!...!....................
def plot_2d_histogram(ax,  tdata, res_data):
    """Plot 2D histogram with tdata on x-axis and res_data on y-axis."""
    h = ax.hist2d(tdata, res_data, bins=15, cmap='Blues') #,cmin=0.1)
    plt.colorbar(h[3], ax=ax)
    
    ax.set_ylabel('reco-true')
    ax.set_xlim(-1.05,1.05)
    xx=0.3
    ax.set_ylim(-xx,xx)
    ax.grid()
  
        
#...!...!....................
def plot_histogram(ax,  tdata,res_data):
    """Plot histogram of the difference and annotate mean and std."""
    
    ax.hist(res_data, bins=15, color='salmon', alpha=0.7)
    mean = np.mean(res_data)
    std = np.std(res_data)
    ax.axvline(mean, color='r', linestyle='dashed', linewidth=1)
    ax.annotate(f'Mean: {mean:.2f}\nStd: {std:.3f}', xy=(0.05, 0.85),c='r', xycoords='axes fraction')
    xx=0.3
    ax.set_xlim(-xx,xx)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

    
#...!...!....................
def plot_EhesrHand(axL,md, rdataV, tdataV):    
    pprint(md)
    pmd=md['payload']
    smd=md['submit']
    tmd=md['qiskit_transp']
    
    mathOp=pmd['ehand_math_op']
    W=pmd['ehand_weight']
    
    nobs=2
    print('PL:inp',rdataV.shape)

    topTit=[ 'job: '+md['short_name'],smd['backend'], 'mathOp: %s W: %.1f'%(mathOp,W)]
    x1LabL=[None,'true  $x_a \cdot x_b$'] 
    if mathOp=='add':
        x1LabL[0]='true  ($x_a+x_b)/2$'
        if W!=0.5: x1LabL[0]='true  %.1f*$x_a + %.1f*x_b$'%(W,1-W)
        
    if mathOp=='sub':
        x1LabL[0]='true ($x_a-x_b)/2$'
        if W!=0.5: x1LabL[0]='true  %.1f*$x_a -  %.1f*x_b$'%(W,1-W)
    
    if mathOp=='none':
        topTit[2]='plain QCrank'
        x1LabL=['true  $x_a$','true $x_b$']
        
    
    for i in range(3):  axL[0,i].set_title(topTit[i])
    
    #....... plot data .....
    for iobs in range(nobs):
        rdata=rdataV[:,iobs,:].flatten()
        tdata=tdataV[:,iobs,:].flatten()
        print('tt',tdata.shape)

        ax=axL[iobs,0]
        ax.scatter(tdata,rdata,)
        ax.set(xlabel=x1LabL[iobs],ylabel='reco')
        compute_correlation_and_draw_line(ax, tdata, rdata)
        ax.set_aspect(1.); ax.set_ylim(-1.05,1.05)
        ax.plot([-1,1],[-1,1],ls='--',c='k',lw=0.7)

        ax=axL[iobs,1]
        res_data = rdata - tdata
        plot_2d_histogram(ax,  tdata,res_data)
        tgth=compute_correlation_and_draw_line(ax, tdata, res_data,xLR=[-0.5,0.5])
        ax.axhline(0.,ls='--',c='k',lw=1.0)
        ax.set(xlabel=x1LabL[iobs])
        
        ax=axL[iobs,2]
        plot_histogram(ax, tdata, res_data)
        ax.set(xlabel='reco-true',ylabel='samples')
        ax.axvline(0.,ls='--',c='k',lw=1.0)

    #...  info in middle column
    txt=''
    if 'noise_model' in smd:
        txt+='\nfake : %s'%(smd['noise_model'])
    
    ax=axL[0,1]
    ax.text(0.10,0.1,txt,transform=ax.transAxes,color='m')

    #...  info on last column
    txt=''
    txt+='\nshots/addr : %d'%(smd['num_shots']/pmd['seq_len'])
    txt+='\nimages: %d'%pmd['num_sample']
    txt+='\nshots/img : %d'%smd['num_shots']
    txt+='\nseq len: %d'%pmd['seq_len']
    txt+='\nqubits: %d'%tmd['num_qubit']
    txt+='\n2q gates: %d'%tmd['num_2q']
    txt+='\n2q depth: %d'%tmd['2q_depth']
    
    ax=axL[0,2]
    ax.text(0.70,0.4,txt,transform=ax.transAxes,color='m')
    
    
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

    rdata=expD['u_reco']
    tdata=expD['u_true']
        
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    plot=PlotterBackbone(args)    
    axL=plot.blank_separate2D(nrow=2,ncol=3, figsize=(12,6),figId=1)

    # Call the main function with the data
    plot_EhesrHand(axL,expMD,rdata, tdata)
    plot.display_all()
    print('M:done')

  
