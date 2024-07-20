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
from toolbox.PlotterBackbone import PlotterBackbone
from toolbox.Util_IOfunc import  read_yaml
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")

    parser.add_argument("--outPath",default='out/',help="all outputs from experiment")
       
    args = parser.parse_args()
    # make arguments  more flexible
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
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
    def compute_time(self,bigD,tag1,figId=1):
        nrow,ncol=1,1       
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(5.5,7))        
        ax = self.plt.subplot(nrow,ncol,1)
  
        dataD=bigD[tag1]
        for tag2 in dataD:
            for tag3 in dataD[tag2]:
                print('plot %s %s %s'%(tag1,tag2,tag3))            
                dataE=dataD[tag2][tag3]
                nqV=dataE['nq']
                runtV=dataE['runt']
                dLab='%s.%s'%(tag2,tag3)               
                ax.plot(nqV,runtV,"*-",label=dLab)
        
        tit='Compute state-vector'

        ax.set(xlabel='num qubits',ylabel='compute end-state (minutes)')
        ax.set_title(tit, pad=20)  # Adjust the pad value as needed
        ax.set_yscale('log')
        ax.grid()
        ax.legend(loc='lower right')

#...!...!....................
def readOne(inpF,dataD,verb=1):
    #print('iii',inpF)
    assert os.path.exists(inpF)
    xMD=read_yaml(inpF,verb)
    #print(inpF,xMD['num_qubit'],xMD['elapsed_time'],float(xMD['num_circ']))
    nq=float(xMD['num_qubit'])
    runt=float(xMD['elapsed_time'])/float(xMD['num_circ'])
    nCX=xMD['num_cx']
    #pprint(xMD)

    if 'cpu_info' in xMD:
        tag1='cpu'
        tag2='all-cpu'  # this should change to c32 or c64 for your samples
    if 'gpu_info' in xMD:
        tag1='gpu'
        tag2=xMD[ 'target']
    if tag1 not in dataD: dataD[tag1]={}
    tag3='%sCX'%nCX
    
    if tag2 not in dataD[tag1]: dataD[tag1][tag2]={}
    if tag3 not in dataD[tag1][tag2]: dataD[tag1][tag2][tag3]={'nq':[],'runt':[]}

    head=dataD[tag1][tag2][tag3]
    head['nq'].append(nq)
    head['runt'].append(runt)

#...!...!....................
def find_yaml_files(directory_path, vetoL=None):
    """
    Scans the specified directory for all files with a .h5 extension,
    rejecting files whose names contain any of the specified veto strings.

    Args:
    directory_path (str): The path to the directory to scan.
    vetoL (list): A list of strings. Files containing any of these strings in their names will be rejected.

    Returns:
    list: A list of paths to the .yaml files found in the directory, excluding vetoed files.
    """
    if vetoL is None:
        vetoL = []
   
    h5_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.yaml') and not any(veto in file for veto in vetoL):
                h5_files.append(os.path.join(root, file))
    return h5_files

#...!...!....................            
def sort_end_lists(d, parent_key='', sort_key='nq', val_key='runt'):
    """
    Recursively prints all keys in a nested dictionary.
    Once the sort_key is in dict it triggers sorting both keys.

    Args:
    d (dict): The dictionary to traverse.
    parent_key (str): The base key to use for nested keys (used for recursion).
    sort_key (str): The key indicating the list to sort by.
    val_key (str): The key indicating the list to sort alongside.
    """
    if sort_key in d:
        xV = d[sort_key]
        yV = d[val_key]
        xU, yU = map(list, zip(*sorted(zip(xV, yV), key=lambda x: x[0])))
        print(' %s.%s:%d' % (parent_key, sort_key, len(xU)))
        d[sort_key]=xU
        d[val_key]=yU
        return
    
    for k, v in d.items():
        full_key = '%s.%s' % (parent_key, k) if parent_key else k
        print(full_key)
        if isinstance(v, dict):
            sort_end_lists(v, full_key, sort_key, val_key)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    #corePath='/dataVault2024/dataCudaQ_'  # in podman
    corePath='/global/cfs/cdirs/mpccc/balewski/quantDataVault2024/dataCudaQ_'  # bare metal
    pathL=['July8']
    fileL=[]
    vetoL=['r1.4','r2.4','r3.4', ]
    for path in pathL:
        path2='%s%s/meas'%(corePath,path)
        fileL+=find_yaml_files( path2, vetoL)

    nInp=len(fileL)
    assert nInp>0
    print('found %d input files, e.g.: '%(nInp),fileL[0])

    dataAll={}
    for i,fileN in enumerate(fileL):
        readOne(fileN,dataAll,i==0)
    #pprint(dataAll)
    print('\nM: all tags:')
    sort_end_lists(dataAll)
    
    # ----  just plotting
    args.prjName='jan23'
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.compute_time(dataAll,'cpu',figId=1)
    if 'b' in args.showPlots:
        plot.compute_time(dataAll,'gpu',figId=2)
        
    plot.display_all(png=1)
    
