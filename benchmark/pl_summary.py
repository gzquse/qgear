#!/usr/bin/env python3
""" 
 concatenate selected metrics from mutiple jobs

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import PlotterBackbone
from toolbox.Util_IOfunc import  read_yaml
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abc-string listing shown plots")

 # IO paths
    parser.add_argument("--basePath",default='/global/cfs/cdirs/mpccc/balewski/quantDataVault2024/dataCudaQ_june28',help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath",default=None,help="input circuits location")
    parser.add_argument("--outPath",default=None,help="all outputs from experiment")
       
    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['Cudaq_dataVault']
    if args.inpPath==None: args.inpPath=os.path.join(args.basePath,'meas') 
    if args.outPath==None: args.outPath=os.path.join(args.basePath,'post') 

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    args.showPlots=''.join(args.showPlots)

    return args


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    #...!...!....................
    def circ_time(self,md,bigD,figId=1,ax=None,tit=None):
        if ax==None:
            nrow,ncol=1,1       
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(5.5,6))        
            # Create subplots within the specified figure
            ax1, ax2 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

        nqV=bigD['num_qubit']
        runtV=bigD['run_time_1circ']/60.
        dLab=md['run_label']
        nG=nqV.shape[0]
        nC=md['num_cpu_runs']
        kL=[nG,nC] # size of GPU & CPU runs
        nqR=(nqV[0]-0.5,nqV[-1]+0.5)

        tit='circ: %dk CX-gates, Perlmutter: %s'%(md['num_cx']/1000,md['run_day'])
        
        #....  time vs. nq
        for j in range(2):
            k=kL[j]
            ax1.plot(nqV[:k],runtV[:k,j],label=dLab[j], marker='*', linestyle='-')
        ax1.set_yscale('log')
        # Set y-axis formatter
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        ax1.set(xlabel='num qubits',ylabel='one circuit simulation (minutes)',title=tit)
        ax1.set_xlim(nqR)
        ax1.grid()

        ax1.legend(loc='lower right')#, title=lgTit,bbox_to_anchor=(0.5, 1.18), ncol=3)

        #....  gain factor
        gainV=runtV[:nC,1]/runtV[:nC,0]
        ax2.plot(nqV[:nC],gainV, marker='*', linestyle='-', color='red')
        ax2.set(xlabel='num qubits',ylabel='GPU speed-up')
        ax2.set_ylim(0,25)
        ax2.set_xlim(nqR)
        ax2.grid()
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        return    
       
        #.... mis text
        txt1='device: '+md['submit']['backend']
        txt1+='\nexec: %s'%md['job_qa']['exec_date']
        txt1+='\ndist %s um'%(md['payload']['atom_dist_um'])
        txt1+='\nOmaga %.1f MHz'%md['payload']['rabi_omega_MHz']
        txt1+='\nshots/job: %d'%md['submit']['num_shots']
        txt1+='\ntot atoms: %d'%md['payload']['tot_num_atom']
        txt1+='\nnum jobs: %d'%nTime
        txt1+='\nreadErr eps: %s'%md['analyzis']['readErr_eps']
        ax.text(0.02,0.55,txt1,transform=ax.transAxes,color='g')

        return
    
   
        

#...!...!....................
def readOne(expN,path,verb):
    assert os.path.exists(path)
    inpF=os.path.join(path,expN+'.yaml')
    if not os.path.exists(inpF): return 0,0,0
    xMD=read_yaml(inpF,verb)
    nq=float(xMD['num_qubit'])
    runt=float(xMD['elapsed_time'])/float(xMD['num_circ'])
    return nq,runt,xMD
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    nqL=[i for i in range(22,31) ]
    #nqL=[17,18,22,23,24]
    
    nqV=np.array(nqL)
    N=nqV.shape[0]
    runtV=np.zeros(shape=(N,4)) 
    runLabs=[ 'gpu1_10k', 'cpu128_10k', 'gpu1_1M', 'cpu128_1M']
    
    cMD=None; gMD=None
    nC=0 # counts cpu runs
    #...  collect data
    for i in range(N):
        for j in range(4):
            expN='ck%dq_%s'%(nqV[i],runLabs[j])
            nq,runt,xMD=readOne(expN,args.inpPath,i+j==0)
            if gMD==None: gMD=xMD
            if cMD==None and j==1: cMD=xMD
            if j==0: nqV[i]=nq
            if j==1 and nq>0 : nC+=1
            runtV[i,j]=runt
              
    expD={}
    expD['num_qubit']=nqV
    expD['run_time_1circ']=runtV

    expMD={'run_label': runLabs,'num_cpu_runs':nC}
   
    for xx in ['date','hash','num_circ','num_cx','num_gate','gpu_info']:
        expMD[xx]=gMD[xx]
    xx='cpu_info'
    expMD[xx]=cMD[xx]
    expMD['run_day']=expMD['date'].split('_')[0]
    expMD['short_name']='gpuSpeed_'+expMD['run_day']
    pprint(expMD)
    
    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    if 'a' in args.showPlots:
        ax2=plot.circ_time(expMD,expD,figId=1)
        

    plot.display_all(png=1)
    
