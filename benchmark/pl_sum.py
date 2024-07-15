#!/usr/bin/env python3
""" 
 concatenate selected metrics from mutiple jobs

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import os, sys
from pprint import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from PlotterBackbone import PlotterBackbone
from Util_IOfunc import  read_yaml
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abc-string listing shown plots")

 # IO paths
    parser.add_argument("--basePath",default='/pscratch/sd/g/gzquse/quantDataVault2024/dataCudaQ_QEra_July12',help="head path for set of experiments, or 'env'")
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
        ticks = [28, 29, 30, 31, 32]
        tick_labels = [f'{2**n:.1e}' for n in ticks]
       
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(tick_labels)
        ax_top.set_xlabel('state-vector size', color='green')
        ax_top.tick_params(axis='x', colors='green')

#...!...!....................
    def compute_time(self,md,bigD,figId=1):
        ax1,ax2=self._make_2_canvas(figId)
        
        nqV=bigD['num_qubit']
        # three dimension
        runtV=bigD['run_time_1circ']/60.
        dLab=md['run_label']
        dCnt=md['run_count']
        details=md['details']
        nT=len(dLab)
        nqR=(nqV[0]-0.5,nqV[-1]+1.5)
        
        #tit='Compute state-vector, %dk CX-gates, PM: %s'%(md['num_cx']/1000,md['run_day'])
        tit='Compute state-vector: %s'%(md['details'][0]['run_day'])
        ax1.set(xlabel='num qubits',ylabel='compute end-state (minutes)')
        ax1.set_title(tit, pad=20)  # Adjust the pad value as needed
        
        #....  time vs. nq
        for j in range(nT):
            for i in range(len(details)):
                k=dCnt[j]
                ax1.plot(nqV[:k],runtV[:k,j,i],label=dLab[j]+' ,cx:'+str(details[i]['num_cx']), marker='*', linestyle='-')
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
        nC=md['num_cpu_runs']
        gainV=bigD['runt_spedup']
        # based on cnot settings
        for i in range(len(gainV[0])):
            ax2.plot(nqV[:nC],gainV.T[i], label='number of cx:'+str(details[i]['num_cx']), marker='*', linestyle='-')
        ax2.set(xlabel='num qubits',ylabel='GPU/CPU speed-up')
        # deal with time out
        # Replace 0.0 values with NaN (Not a Number)
        gainV[gainV == np.inf] = np.nan
        # Find the maximum value ignoring NaNs
        yMX=np.nanmax(gainV)*1.2
        # Replace NaN values with the maximum value
        gainV[np.isnan(gainV)] = yMX
        ax2.set_ylim(0,yMX)
        ax2.set_xlim(nqR)
        ax2.grid()
        ax2.legend(loc='upper right')
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        return    


#...!...!....................
def readOne(expN,path,verb):
    assert os.path.exists(path)
    inpF=os.path.join(path,expN+'.yaml')
    if not os.path.exists(inpF): return 0,0,{}
    xMD=read_yaml(inpF,verb)
    nq=float(xMD['num_qubit'])
    runt=float(xMD['elapsed_time'])/float(xMD['num_circ'])

    return nq,runt,xMD

#...!...!....................
def post_process(md,bigD):
    nC=md['num_cpu_runs']
    runtV=bigD['run_time_1circ']
    
    #... compute par gpu to par cpu speed up
    gainV=runtV[:nC,0]/runtV[:nC,1]       
    #... store 2ndary data
    # if did not get the result the result should be 0 so divide 0 is inf
    bigD['runt_spedup']=gainV

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    nqL=[i for i in range(28,33)]
    
    nqV=np.array(nqL)
    N=nqV.shape[0]
    runLabs=['par-cpu', 'par-gpu', 'adj-gpu']
    cxL=(100, 10000, 20000)
    nT=len(runLabs)
    nCx=len(cxL)
    cntT=[0]*nT  # capture number of finished runs of each target
    mdT=[[None for i in range(nT)] for j in range(nCx)]
    runtV=np.zeros(shape=(N,nT,nCx))
    shotsV=np.zeros(shape=(N,nT,nCx)) 
    
    # prefix of file
    prefix="mar"

    #...  collect data
    for i in range(N):
        for j in range(nT):
            for k in range(nCx):
                expN=prefix+'%dq%dcx_%s'%(nqV[i],cxL[k],runLabs[j])
                # get cpu first task result from all four tasks
                if j!=1: expN+='_r0.4'
                nq,runt,xMD=readOne(expN,args.inpPath,i+j==0)
                if nq==0: continue # no data was found
                if mdT[j][k]==None: mdT[j][k]=xMD
                if j==2: nqV[i]=nq  # it will be adj-gpu
                runtV[i,j,k]=runt
                shotsV[i,j,k]=xMD['num_shots']
            cntT[j]+=1
    print('M: got jobs:',cntT) 
    expD={}
    expD['num_qubit']=nqV
    expD['run_time_1circ']=runtV
    expD['shots']=shotsV

    expMD={'run_label': runLabs,'run_count':cntT}
    # get first target first cx setting's meta data
    # Since every target enjoys same cnot gate set up so we only get just one target which is enough
    xMD=mdT[0]
    # mdT details: [{'date','hash','num_circ','num_cx','num_gate'}]
    expMD['details']=xMD
    
    for entry in expMD['details']:
        entry['run_day'] = entry['date'].split('_')[0]

    expMD['short_name']='gpuSpeed_'+expMD['details'][0]['run_day']

    expMD['run_label']=runLabs
    pprint(expMD)
    expMD['num_cpu_runs']=cntT[0]
    post_process(expMD,expD)
    
    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.compute_time(expMD,expD,figId=1)
    if 'b' in args.showPlots:
        plot.sample_time(expMD,expD,figId=2)
        

    plot.display_all(png=1)
    
