#!/usr/bin/env python3

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from neuron_inverter_benchmark.torch.toolbox.Plotter_Backbone import roys_fontset
from PlotterBackbone import PlotterBackbone
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib import font_manager
import matplotlib.patches as patches

roys_fontset(plt)
from Util_IOfunc import  read_yaml

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
        fig=self.plt.figure(figId,facecolor='white',figsize=(5.5,7))        
        ax = self.plt.subplot(nrow,ncol,1)
        if 'gpu' in tag1:
            dataD=bigD[tag1.split('-')[1]]
        elif 'qft' in tag1:
            dataD=bigD[tag1]
        else:
            dataD=bigD[tag1]

        for tag2 in dataD:
            for tag3 in dataD[tag2]:
                if '20000CX' in tag3:  # Skip lines with 20,000 CX
                    continue
                # if tag2 == 'nvidia-mqpu':
                #     continue
                print('plot %s %s %s'%(tag1,tag2,tag3))            
                dataE=dataD[tag2][tag3]
                nqV=dataE['nq']
                runtV=dataE['runt']/60.0 # convert time to min
                date=dataE['date']
                dLab='%s'%(tag3)   
                # default color
                dCol='k'
                # Extract cores and tasks_per_node from tag3
                if tag1 == 'cpu':
                    cores = tag3.split('_')[1][1:]
                    tasks_per_node = tag3.split('_')[2][2:]
                    
                    if cores == '32' and tasks_per_node == '4':
                        dCol='#1f77b4'  # Blue for 32 cores and 4 tasks
                        linestyle = '--'  # Dashed line for CPU
                        marker_style = 's'  # Square for long unitary
                        filled = True
                    elif cores == '64' and tasks_per_node == '1':
                        marker_style = '^'  # Triangle for 64 cores and 1 task
                        dCol='#ff7f0e'  # Orange
                        linestyle = '--'  # Dashed line for CPU
                        filled = False
                    else:
                        marker_style = 'o'  # Default to circle
                        dCol='#2ca02c'  # Green
                        linestyle = '--'  # Dashed line for CPU
                        filled = False
                elif tag1 == 'par-gpu':
                    if tag2 == 'nvidia':
                        dCol='#d62728'  # Red
                        linestyle = '-'  # Solid line for GPU
                        marker_style = 'D'  # Diamond for long unitary
                        filled = True
                    elif tag2 == 'nvidia-mqpu':
                        marker_style = 'o'  
                        dCol='#9467bd'  # Purple
                        linestyle = '-'  # Solid line for GPU
                        filled = False
                    else: continue 
                elif tag1 == 'adj-gpu':
                    if tag2 == 'nvidia-mgpu':
                        if '10K' in tag3:
                            marker_style = '^'  
                            dCol='#8c564b'  # Brown
                            linestyle = '-'  # Solid line for GPU
                            filled = True
                        elif '1M' in tag3:
                            marker_style = 'o'  
                            dCol='#e377c2'  # Pink
                            linestyle = '-'  # Solid line for GPU
                            filled = True
                        elif '10M' in tag3:
                            marker_style = '.'  
                            dCol='#7f7f7f'  # Gray
                            linestyle = '-'  # Solid line for GPU
                            filled = False
                        else:
                            marker_style = '1'  
                            dCol='#bcbd22'  # Yellow-green
                            linestyle = '-'  # Solid line for GPU
                            filled = False
                    else: continue
                elif tag1 == qft:
                    if 'fp32' in tag3:
                        if 'pennylane' in tag2:
                            dCol='#bcbd22'
                            marker_style = 's'
                            linestyle = '-'  # Solid line for GPU
                            filled = True
                            dLab = 'QFT fp32 Pennylane'
                        else:
                            dCol='#ff7f0e'  # Cyan
                            marker_style = '^'
                            linestyle = '-'  # Solid line for GPU
                            filled = True
                            dLab = 'QFT fp32 Cudaq'
                    elif 'fp64' in tag3:
                        # dCol='#ff7f0e'  # Yellow-green
                        # marker_style = 'o'  
                        # linestyle = '-'  # Solid line for GPU
                        # filled = False
                        # dLab = 'QFT fp64 Cudaq'
                        continue
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
                    ax.plot(nqV_shifted, runtV_shifted, marker=marker_style, linestyle=linestyle, markerfacecolor=isFilled, color=dCol,label=dLab,markersize=9)     

                else:
                    ax.plot(nqV, runtV, marker=marker_style, linestyle=linestyle, markerfacecolor=isFilled, color=dCol, label=dLab, markersize=9)
        tit='Compute state-vector tag1=%s'%tag1
        tit='c)'
        # Place the title above the legend
        ax.set(xlabel='num qubits',ylabel='one circuit simu time (minutes)')
        ax.set_title(tit, pad=50)  # Adjust the pad value as needed
        ax.set_yscale('log')
        # ax.set_ylim(1e-3, 3.5e+3)
        ax.set_ylim(1e-3, 1e+1)
        ax.set_xlim(27.5,34.5) 
        # ax.set_xticks(range(16, 33, 2))  # Integers from 16 to 33
        ax.grid(False)
        # adjustable
        ax.legend(loc=2, ncol=1, borderaxespad=0., fontsize=8.5)
        # square = patches.Rectangle((30 - 0.2, 1e-3), 0.4, 1e-3, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(square)
    #    ax.axhline(1440, ls='--', lw=2, c='m')
    #    ax.text(30, 600, '24h time-out', c='m')
       
    def compute_time_mixed(self, bigD, figId=1, shift=False):
        nrow, ncol = 1, 1       
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(5.5, 7))        
        ax = self.plt.subplot(nrow, ncol, 1)

        # Plot CPU data
        if 'cpu' in bigD:
            dataD = bigD['cpu']
            for tag2 in dataD:
                for tag3 in dataD[tag2]:
                    if not (('10kCX_c64_tp4' in tag3) or ('100CX_c64_tp1' in tag3)):
                        continue
                        
                    print('plot CPU %s %s' % (tag2, tag3))            
                    dataE = dataD[tag2][tag3]
                    nqV = np.array(dataE['nq'])
                    runtV = np.array(dataE['runt']) / 60.0
                    
                    if '10kCX' in tag3:
                        marker_style, dCol = 'o', '#d62728'  # Red for long unitary
                        linestyle = '--'  # Dashed line for CPU
                        dLab = 'CPU node, long unitary'
                        filled = False
                    else:  # 100CX
                        marker_style, dCol = 's', '#1f77b4'  # Blue for short unitary
                        linestyle = '--'  # Dashed line for CPU
                        dLab = 'CPU node, short unitary'
                        filled = False
                    
                    if shift:
                        shift_x = np.random.uniform(-0.1, 0.1, size=len(nqV))
                        shift_y = np.random.uniform(-0.1, 0.1, size=[len(runtV)])
                        nqV = nqV + shift_x
                        runtV = runtV + shift_y
                        
                    ax.plot(nqV, runtV, marker=marker_style, linestyle=linestyle, 
                           color=dCol, label=dLab, markersize=9, markerfacecolor='none' if not filled else dCol)

        # Add 2^n reference line as an extrapolation of CPU-long unitary
        n_values = np.linspace(32, 34.5, 100)
        scaling_factor = 10**-7.185
        reference_times = scaling_factor * 2**n_values
        ax.plot(n_values, reference_times, 'k--', alpha=0.5, label=None)

        # Add circles to the reference line within the range 27.5 to 30.1
        circle_indices = np.where((n_values >= 32.5) & (n_values <= 34.1))[0]
        circle_indices = np.linspace(circle_indices[0], circle_indices[-1], 4, dtype=int)  # Show only five circles
        ax.plot(n_values[circle_indices], reference_times[circle_indices], 'k--', alpha=0.8, label=None)
        
        # Calculate the angle of the line for the text rotation
        # angle = np.degrees(np.arctan2(reference_times[-1] - reference_times[0], n_values[-1] - n_values[0]))
        ax.text(33, scaling_factor * 2**33 * 0.3, r'scaling $2^n$', zorder=10, color='black', fontsize=10, rotation=23, rotation_mode='anchor', ha='center', va='center')

        # Plot GPU data
        if 'gpu' in bigD:
            dataD = bigD['gpu']
            for tag2 in dataD:
                for tag3 in dataD[tag2]:
                    tag3_clean = tag3.replace('.NoneS', '')
                    
                    if not (('mgpu.10kCX' in tag3_clean) or ('mgpu.100CX' in tag3_clean) or ('gpu.10kCX' in tag3_clean) or ('gpu.100CX' in tag3_clean)):
                        continue
                        
                    print('plot GPU %s %s' % (tag2, tag3_clean))            
                    dataE = dataD[tag2][tag3]
                    nqV = dataE['nq']
                    runtV = dataE['runt'] / 60.0
                    
                    if '10kCX' in tag3_clean:
                        if 'mgpu' in tag3_clean:
                            marker_style, dCol = '^', '#d62728'  # Red for 4-GPU long unitary
                            linestyle = '-'  # Solid line for GPU
                            dLab = '4-GPU, long unitary'
                            filled = True
                        else:
                            marker_style, dCol = 'o', '#d62728'  # Red for 1-GPU long unitary
                            linestyle = '-'  # Solid line for GPU
                            dLab = '1-GPU, long unitary'
                            filled = True
                    else:  # 100CX
                        if 'mgpu' in tag3_clean:
                            marker_style, dCol = '^', '#1f77b4'  # Blue for 4-GPU short unitary
                            linestyle = '-'  # Solid line for GPU
                            dLab = '4-GPU, short unitary'
                            filled = True
                        else:
                            marker_style, dCol = 's', '#1f77b4'  # Blue for 1-GPU short unitary
                            linestyle = '-'  # Solid line for GPU
                            dLab = '1-GPU, short unitary'
                            filled = True
                        
                    ax.plot(nqV, runtV, marker=marker_style, linestyle=linestyle,
                           color=dCol, label=dLab, markersize=9, markerfacecolor='none' if not filled else dCol)

        tit = 'a)'
        ax.set(xlabel='num qubits', ylabel='one circuit simu time (minutes)')
        ax.set_title(tit, pad=50)
        ax.set_yscale('log')
        
        # Modified axis limits
        ax.set_xlim(27.5, 34.5)
        ax.set_ylim(1e-3, 3e3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Show only integer values on x-axis
        
        ax.axhline(1440, ls='--', lw=2, c='m')
        ax.text(31, 1600, '24h time limit', c='m')
        
        ax.grid(False)
        
        # Reorder the legend and change the names
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ['QFT fp64' if 'fp64' in label else 'QFT fp32' if 'fp32' in label else label for label in labels]
        order = [1, 5, 3, 0, 4, 2]  # Adjust the order as needed
        ax.legend([handles[idx] for idx in order], [new_labels[idx] for idx in order], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=2, mode="expand", borderaxespad=0., fontsize=8.5)

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

def extract_qft_from_filename(filename):
    # Split the filename into components
    components = filename.split('_')
    
    # Iterate through the components to find "qft1"
    for component in components:
        if component == "qft1":
            return component
    
    # If "qft1" is not found, return None
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
    if xMD["num_shots"] == 10000:
        num_shots_formatted = "10K"
    elif xMD["num_shots"] == 1000000:
        num_shots_formatted = "1M"
    elif xMD["num_shots"] == 10000000:
        num_shots_formatted = "10M"
    elif xMD["num_shots"] == 100000000:
        num_shots_formatted = "100M"
    else:
        # previous setup is 10240
        num_shots_formatted = None
    if tag1=='cpu':
        tag3 = f'{num_cx_formatted}CX_c{cores}_tp{tasks_per_node}'
    elif tag1=='gpu':
        g_tag = 'gpu' 
        if '-' in tag2:
            parts = tag2.split('-')
            if len(parts) > 1:
                g_tag = parts[1]
        tag3 = f'{g_tag}.{num_cx_formatted}CX.{num_shots_formatted}S'
    if tag2 not in dataD[tag1]: dataD[tag1][tag2]={}
    if tag3 not in dataD[tag1][tag2]: dataD[tag1][tag2][tag3]={'nq':[],'runt':[], 'cores': [], 'tasks_per_node': [], 'date': []}
    
    head=dataD[tag1][tag2][tag3]
    head['nq'].append(nq)
    head['runt'].append(runt)
    head['cores'].append(cores)
    head['tasks_per_node'].append(tasks_per_node)
    head['date'].append(date)

def readOneQFT(inpF,dataD,qft,verb=1):
    assert os.path.exists(inpF)
    date = extract_date_from_path(inpF)
    xMD=read_yaml(inpF,verb)
    #print(inpF,xMD['num_qubit'],xMD['elapsed_time'],float(xMD['num_circ']))
    nq=float(xMD['num_qubit'])
    runt=float(xMD['elapsed_time'])
    tag1=qft
    tag2=xMD['target']
    if tag1 not in dataD: dataD[tag1]={}
    # save for future compare
    # num_cx_formatted = "10k" if xMD["num_cx"] == 10000 else f'{xMD["num_cx"]}'
    num_shots_formatted = "10k" if xMD["num_shots"] == 10000 else 100
    # one time use hard coded
    options = inpF.split('/')[-1].split('_')[-2]
    shots = xMD['num_shots']
    g_tag = 'gpu' 
    if '-' in tag2:
        parts = tag2.split('-')
        if len(parts) > 1:
            g_tag = parts[1]
    tag3 = f'{qft}.{g_tag}.{options}.{num_shots_formatted}S'
    if tag2 not in dataD[tag1]: dataD[tag1][tag2]={}
    if tag3 not in dataD[tag1][tag2]: dataD[tag1][tag2][tag3]={'nq':[],'runt':[], 'shots':[],'date': []}
    head=dataD[tag1][tag2][tag3]
    head['nq'].append(nq)
    head['runt'].append(runt)
    head['shots'].append(shots)
    head['date'].append(date)

#...!...!....................
def find_yaml_files(directory_path, vetoL=None):
    if vetoL is None:
        vetoL = []
   
    yaml_files = []
    if not os.path.exists(directory_path):
        print(f'Warning: Directory not found: {directory_path}')
        return yaml_files
        
    print(f'Scanning path: {directory_path}')
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.yaml') and not any(veto in file for veto in vetoL):
                yaml_files.append(os.path.join(root, file))
    print(f'Found {len(yaml_files)} YAML files')
    return yaml_files

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
    corePath='/pscratch/sd/g/gzquse/quantDataVault2024/dataCudaQ_'  # bare metal Martin
    pathL=['Nov15']
    fileL=[]
    vetoL=['r1.4','r2.4','r3.4']
    
    for path in pathL:
        path2=f'{corePath}{path}/meas'
        fileL+=find_yaml_files(path2, vetoL)
        
    num_input_files=len(fileL)
    if num_input_files == 0:
        print('Error: No input files found')
        sys.exit(1)
        
    print(f'Found {num_input_files} input files')
    if num_input_files > 0:
        print(f'Example file: {fileL[0]}')

    dataAll={}
    dataQFT={}
    for i,fileN in enumerate(fileL):
        qft = extract_qft_from_filename(fileN)
        if not qft:
            readOne(fileN,dataAll,i==0)
        else:
            readOneQFT(fileN,dataQFT,qft,i==0)
    
    #pprint(dataAll)
    print('\nM: all tags:')
    sort_end_lists(dataAll)
    sort_end_lists(dataQFT)
    # ----  just plotting
    args.prjName='jan23'
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.compute_time(dataAll,'cpu', figId=1, shift=args.shift)
    if 'b' in args.showPlots:
        plot.compute_time(dataAll,'par-gpu',figId=2, shift=args.shift)
    if 'c' in args.showPlots:
        plot.compute_time(dataAll,'adj-gpu',figId=3, shift=args.shift)
    if 'd' in args.showPlots:
        plot.compute_time(dataQFT,qft,figId=4, shift=args.shift)
    if 'x' in args.showPlots:  # 'x' for mixed plot
        plot.compute_time_mixed(dataAll, figId=5, shift=args.shift)
    plot.display_all(png=0)
