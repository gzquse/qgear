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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import PlotterBackbone
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from Util_IOfunc import  read_yaml
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-s", "--shift", type=bool, default=True, help="whether shift the dots")

    parser.add_argument("--outPath",default='out',help="all outputs from experiment")
       
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
    def compute_time(self,bigD,tag1,figId=1,shift=False):
        nrow,ncol=1,1       
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(5.5,7))        
        ax = self.plt.subplot(nrow,ncol,1)

        dataD=bigD[tag1]

        for tag2 in dataD:
            for tag3 in dataD[tag2]:
                if '20000CX' in tag3:  # Skip lines with 20,000 CX
                    continue
                print('plot %s %s %s'%(tag1,tag2,tag3))            
                dataE=dataD[tag2][tag3]
                nqV=dataE['nq']
                runtV=dataE['runt']/60.0 # convert time to min
                date=dataE['date']
                dLab='%s'%(tag3)   
                # Extract cores and tasks_per_node from tag3
                if tag1 == 'cpu':
                    cores = tag3.split('_')[1][1:]
                    tasks_per_node = tag3.split('_')[2][2:]
                    
                    if cores == '32' and tasks_per_node == '4':
                        dCol='C1'  # Square for 32 cores and 4 tasks
                    elif cores == '64' and tasks_per_node == '1':
                        marker_style = '^'  # Triangle for 32 cores and 8 tasks
                        dCol='C2'
                    else:
                        marker_style = 'o'  # Default to circle
                        dCol='C3'
                # Set marker style based on cores and tasks_per_node
                if '100CX' in tag3:
                    marker_style = 's'
                elif '10kCX' in tag3:
                    marker_style = '^'
                else:
                    marker_style = 'o'
                
                
                # Introduce a small random shift to avoid overlap
                isFilled=None if '10kCX' in tag3 else 'none'
                # Dont shift when plot gpu! cause it is so fast!
                if shift and tag1 == 'cpu':
                    shift_x = np.random.uniform(-0.1, 0.1, size=len(nqV))
                    shift_y = np.random.uniform(-0.1, 0.1, size=len(runtV))
                    nqV_shifted = nqV + shift_x
                    runtV_shifted = runtV + shift_y
                    if tag1 == 'cpu':
                        ax.plot(nqV_shifted, runtV_shifted, marker=marker_style, linestyle='-', markerfacecolor=isFilled, color=dCol,label=dLab,markersize=9)    
                    elif tag1 == 'gpu':
                        ax.plot(nqV_shifted, runtV_shifted, marker=marker_style, linestyle='-', markerfacecolor=isFilled, label=dLab,markersize=9)    

                else:
                    ax.plot(nqV,runtV,marker=marker_style, markerfacecolor='none', linestyle='-', label=dLab)
        tit='Compute state-vector tag1=%s'%tag1
        # Place the title above the legend
        ax.set(xlabel='num qubits',ylabel='compute end-state (minutes)')
        ax.set_title(tit, pad=50)  # Adjust the pad value as needed
        ax.set_yscale('log')
        ax.grid()
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0., )


def extract_date_from_path(file_path):
    # Split the path into components
    path_components = file_path.split('/')
    
    # Find the component that starts with "dataCudaQ_"
    for component in path_components:
        if component.startswith('dataCudaQ_'):
            # Extract the date part from this component
            date_part = component[len('dataCudaQ_'):]
            return date_part
    
    # If the component is not found, return None
    return None

#...!...!....................
def readOne(inpF,dataD,verb=1):
    assert os.path.exists(inpF)
    date = extract_date_from_path(inpF)
    xMD=read_yaml(inpF,verb)
    #print(inpF,xMD['num_qubit'],xMD['elapsed_time'],float(xMD['num_circ']))
    nq=float(xMD['num_qubit'])
    runt=float(xMD['elapsed_time'])/float(xMD['num_circ'])
    # record the specific details we will use
    cores = xMD['cores'] if 'cores' in xMD else 32
    tasks_per_node = xMD['tasks_per_node'] if 'tasks_per_node' in xMD else 4
    #pprint(xMD)
    if 'cpu_info' in xMD:
        tag1='cpu'
    if 'gpu_info' in xMD:
        tag1='gpu'
    tag2=xMD['target']
    if tag1 not in dataD: dataD[tag1]={}
    num_cx_formatted = "10k" if xMD["num_cx"] == 10000 else f'{xMD["num_cx"]}'
    if tag1=='cpu':
        tag3 = f'{num_cx_formatted}CX_c{cores}_tp{tasks_per_node}'
    elif tag1=='gpu':
        g_tag = tag2.split('-')[1]
        tag3 = f'{g_tag}.{num_cx_formatted}CX'
    if tag2 not in dataD[tag1]: dataD[tag1][tag2]={}
    if tag3 not in dataD[tag1][tag2]: dataD[tag1][tag2][tag3]={'nq':[],'runt':[], 'cores': [], 'tasks_per_node': [], 'date': []}
    
    head=dataD[tag1][tag2][tag3]
    head['nq'].append(nq)
    head['runt'].append(runt)
    head['cores'].append(cores)
    head['tasks_per_node'].append(tasks_per_node)
    head['date'].append(date)

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
        d[sort_key]=np.array(xU)
        d[val_key]=np.array(yU)
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
    corePath='/pscratch/sd/g/gzquse/quantDataVault2024/dataCudaQ_'  # bare metal
    pathL=[ 'July12']
    fileL=[]
    vetoL=['r1.4','r2.4','r3.4', ]
    for path in pathL:
        path2='%s%s/meas'%(corePath,path)
        fileL+=find_yaml_files( path2, vetoL)
    nInp=len(fileL)
    assert nInp>0
    print('found %d input files, e.g.: '%(nInp),fileL[0])
    print(fileL)
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
        plot.compute_time(dataAll,'cpu', figId=1, shift=args.shift)
    if 'b' in args.showPlots:
        plot.compute_time(dataAll,'gpu',figId=2, shift=args.shift)
  
    plot.display_all(png=1)
    
