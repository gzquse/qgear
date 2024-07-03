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

    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abc-string listing shown plots")

 # IO paths
    parser.add_argument("--basePath",default='/pscratch/sd/g/gzquse/quantDataVault2024/dataCudaQ_test3',help="head path for set of experiments, or 'env'")
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
    def _make_2_canvas(self,figId):
        nrow,ncol=2,1       
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(5.5,7))        
        # Create subplots within the specified figure
        ax1, ax2 = fig.subplots(nrow,ncol, gridspec_kw={'height_ratios': [2, 1]})
        return ax1, ax2

#...!...!....................
    def _add_stateSize_axis(self,ax):
        # Create a twin x-axis on the top for state-vector size
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        
        # Define specific ticks and labels
        ticks = [20, 24, 28, 32]
        tick_labels = [f'{2**n:.1e}' for n in ticks]
       
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(tick_labels)
        ax_top.set_xlabel('state-vector size', color='green')
        ax_top.tick_params(axis='x', colors='green')

#...!...!....................
    def compute_time(self,md,bigD,figId=1):
        ax1,ax2=self._make_2_canvas(figId)

        nqV=bigD['num_qubit']
        runtV=bigD['run_time_1circ']/60.
        dLab=['CudaQ: 1GPU','Qiskit: 32CPUs']
        nG=nqV.shape[0]
        nC=md['num_cpu_runs']
        kL=[nG,nC] # size of GPU & CPU runs
        nqR=(nqV[0]-0.5,nqV[-1]+1.5)
        
        tit='Compute state-vector, %dk CX-gates, PM: %s'%(md['num_cx']/1000,md['run_day'])
        # sampling circuit, 1M shots, ....
        ax1.set(xlabel='num qubits',ylabel='compute end-state (minutes)')
        ax1.set_title(tit, pad=20)  # Adjust the pad value as needed
        
        #....  time vs. nq
        for j in range(1,-1,-1):
            k=kL[j]
            ax1.plot(nqV[:k],runtV[:k,j],label=dLab[j], marker='*', linestyle='-')
        ax1.set_yscale('log')
        # Set y-axis formatter
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax2.set_ylim(0.2,1900)
        ax1.set_xlim(nqR)
        ax1.grid()
        ax1.legend(loc='lower right')#, title=lgTit,bbox_to_anchor=(0.5, 1.18), ncol=3)

        #... draw limiting lines
        ax1.axhline(1440,ls='--',lw=2,c='m')
        ax1.text(20,600,'24h time-out',c='m')

        ax1.axvline(32.5,ls='--',lw=2,c='firebrick')
        ax1.text(31.5,1,'A100 RAM limit',c='firebrick', rotation=90)
        self._add_stateSize_axis(ax1)
        
        #....  gain factor
        gainV=bigD['runt_spedup']
        ax2.plot(nqV[:nC],gainV, marker='*', linestyle='-', color='red')
        ax2.set(xlabel='num qubits',ylabel='GPU/CPU speed-up')
        ax2.set_ylim(0,35)
        ax2.set_xlim(nqR)
        ax2.grid()
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        return    

#...!...!....................
    def sample_time(self,md,bigD,figId=1):
        ax1,ax2=self._make_2_canvas(figId)

        nqV=bigD['num_qubit']
        smpltV=bigD['sample_time']
        dLab=['CudaQ: 1GPU','Qiskit: 32CPUs']
        nG=nqV.shape[0]
        nC=md['num_cpu_runs']
        kL=[nG,nC] # size of GPU & CPU runs
        nqR=(nqV[0]-0.5,nqV[-1]+1.5)
        
        shotsM=bigD['shots'][0,2]/1000000
        
        tit=' %dM shots sample from state-vector, PM: %s'%(shotsM,md['run_day'])
        # sampling circuit, 1M shots, ....
        ax1.set(xlabel='num qubits',ylabel='sample end-state (seconds)')
        ax1.set_title(tit, pad=20)  # Adjust the pad value as needed
        
        #....  time vs. nq
        for j in range(1,-1,-1):
            k=kL[j]
            ax1.plot(nqV[:k],smpltV[:k,j],label=dLab[j], marker='*', linestyle='-')
        ax1.set_yscale('log')
        # Set y-axis formatter
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        #ax2.set_ylim(0.2,1900)
        ax1.set_xlim(nqR)
        ax1.grid()
        ax1.legend(loc='lower right')#, title=lgTit,bbox_to_anchor=(0.5, 1.18), ncol=3)

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
    if not os.path.exists(inpF): return 0,0,{}
    xMD=read_yaml(inpF,verb)
    nq=float(xMD['num_qubit'])
    runt=float(xMD['elapsed_time'])/float(xMD['num_circ'])

    #june29 hack, add numshots to MD
    if '10k' in expN: xMD['num_shots']=10000
    if '1M'  in expN: xMD['num_shots']=1001000
    return nq,runt,xMD

#...!...!....................
def post_process(md,bigD):
    nC=md['num_cpu_runs']
    runtV=bigD['run_time_1circ']
    print('g10k',runtV[:,0])
    smpltV=np.zeros_like(runtV[:,:2])
    for j in range(2):
        smpltV[:,j]=runtV[:,j+2] - runtV[:,j]

    # hack2:  assume slower has more shots
    smpltV=np.abs(smpltV)
    
    print('smpltV:\n',smpltV)
    #... compute gain
    gainV=runtV[:nC,1]/runtV[:nC,0]    
          
    #... store 2ndary data
    bigD['sample_time']=smpltV
    bigD['runt_spedup']=gainV
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    nqL=[i for i in range(18,22) ]
    #nqL=[17,18]+[i for i in range(22,30) ]
    #nqL=[17,18,22,23,24]
    
    nqV=np.array(nqL)
    N=nqV.shape[0]
    runtV=np.zeros(shape=(N,4))
    shotsV=np.zeros(shape=(N,4)) 
    runLabs=[ 'gpu1_10k', 'cpu128_10k', 'gpu1_1M', 'cpu128_1M']
    
    cMD=None; gMD=None
    nC=0 # counts cpu runs
    #...  collect data
    for i in range(N):
        for j in range(4):
            expN='mar%dq_%s'%(nqV[i],runLabs[j])
            nq,runt,xMD=readOne(expN,args.inpPath,i+j==0)
            print(i,j,expN)
            if j in [0,2]:  assert nq>0 
            if nq==0: continue # no data was found
            if gMD==None: gMD=xMD
            if cMD==None and j==1: cMD=xMD
            if j==0: nqV[i]=nq
            if j==1 : nC+=1
            runtV[i,j]=runt
            shotsV[i,j]=xMD['num_shots']
              
    expD={}
    expD['num_qubit']=nqV
    expD['run_time_1circ']=runtV
    expD['shots']=shotsV

    expMD={'run_label': runLabs,'num_cpu_runs':nC}
    
    for xx in ['date','hash','num_circ','num_cx','num_gate','gpu_info']:
        expMD[xx]=gMD[xx]
    xx='cpu_info'
    expMD[xx]=cMD[xx]
    expMD['run_day']=expMD['date'].split('_')[0]
    expMD['short_name']='gpuSpeed_'+expMD['run_day']
    pprint(expMD)

    post_process(expMD,expD)
    
    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.compute_time(expMD,expD,figId=1)
    if 'b' in args.showPlots:
        plot.sample_time(expMD,expD,figId=2)
        

    plot.display_all(png=1)
    
