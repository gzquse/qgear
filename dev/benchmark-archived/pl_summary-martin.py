#!/usr/bin/env python3
"""
concatenate selected metrics from multiple jobs
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import time
import sys, os
from pprint import pprint
from toolbox.Util_H5io4 import write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import PlotterBackbone
from toolbox.Util_IOfunc import read_yaml
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-Y", "--noXterm", dest='noXterm', action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("-p", "--showPlots", default='a', nargs='+', help="abc-string listing shown plots")
    
    # IO paths
    parser.add_argument("--basePath", default='/pscratch/sd/g/gzquse/quantDataVault2024/dataCudaQ_July4', help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath", default=None, help="input circuits location")
    parser.add_argument("--outPath", default=None, help="all outputs from experiment")
       
    args = parser.parse_args()
    if 'env' == args.basePath: args.basePath = os.environ['Cudaq_dataVault']
    if args.inpPath is None: args.inpPath = os.path.join(args.basePath, 'meas')
    if args.outPath is None: args.outPath = os.path.join(args.basePath, 'post')
    
    for arg in vars(args):  print('myArg:', arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    args.showPlots = ''.join(args.showPlots)
    
    return args


class Plotter(PlotterBackbone):
    def __init__(self, args):
        super().__init__(args)
        
    def _make_2_canvas(self, figId):
        nrow, ncol = 2, 1
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(5.5, 7))
        ax1, ax2 = fig.subplots(nrow, ncol, gridspec_kw={'height_ratios': [2, 1]})
        return ax1, ax2

    def _add_stateSize_axis(self, ax):
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ticks = [20, 24, 28, 32]
        tick_labels = [f'{2**n:.1e}' for n in ticks]
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(tick_labels)
        ax_top.set_xlabel('state-vector size', color='green')
        ax_top.tick_params(axis='x', colors='green')

    def compute_time(self, md, bigD, figId=1):
        ax1, ax2 = self._make_2_canvas(figId)

        nqV = bigD['num_qubit']
        runtV = bigD['run_time_1circ'] / 60.
        dLab = ['CudaQ: 1GPU', 'Qiskit: 32CPUs']
       
        nqR = (nqV[0] - 0.5, nqV[-1] + 1.5)
        
        tit = f'Compute state-vector, {md["num_cx"] // 1000}k CX-gates, PM: {md["run_day"]}'
        ax1.set(xlabel='num qubits', ylabel='compute end-state (minutes)')
        ax1.set_title(tit, pad=20)
        
        for j in range(1, -1, -1):
            ax1.plot(nqV, runtV[:, j], label=dLab[j], marker='*', linestyle='-')
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax2.set_ylim(0.2, 1900)
        ax1.set_xlim(nqR)
        ax1.grid()
        ax1.legend(loc='lower right')

        ax1.axhline(1440, ls='--', lw=2, c='m')
        ax1.text(19, 600, '24h time-out', c='m')
        
        ax1.axvline(32.5, ls='--', lw=2, c='firebrick')
        ax1.text(31.5, 1, 'A100 RAM limit', c='firebrick', rotation=90)
        self._add_stateSize_axis(ax1)
        
        gainV = bigD['runt_spedup']
        ax2.plot(nqV, gainV, marker='*', linestyle='-', color='red')
        ax2.set(xlabel='num qubits', ylabel='GPU/CPU speed-up')
        ax2.set_ylim(0, 35)
        ax2.set_xlim(nqR)
        ax2.grid()
        ax2.yaxis.set_major_locator(MaxNLocator(4))

    def other_plot(self, md, bigD, figId=1):
        ax1, ax2 = self._make_2_canvas(figId)
        print('other plot is empty')


def readOne(expN, path, verb):
    inpF = os.path.join(path, f"{expN}.yaml")
    if not os.path.exists(inpF): return 0, 0, {}
    xMD = read_yaml(inpF, verb)
    nq = float(xMD['num_qubit'])
    runt = float(xMD['elapsed_time']) / float(xMD['num_circ'])
    return nq, runt, xMD


def read_all_data(nqV, runLabs, inpPath):
    N = nqV.shape[0]
    runtV = np.zeros(shape=(N, 2))
    gMD, cMD = None, None

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(N):
            for j in range(2):
                expN = f'mar{nqV[i]}q_{runLabs[j]}'
                futures.append(executor.submit(readOne, expN, inpPath, i+j==0))

        for future in as_completed(futures):
            nq, runt, xMD = future.result()
            idx = futures.index(future)
            i, j = divmod(idx, 2)
            if gMD is None: gMD = xMD
            if cMD is None and j == 1: cMD = xMD
            runtV[i, j] = runt
    
    return runtV, gMD, cMD


def post_process(md, bigD):
    runtV = bigD['run_time_1circ']
    gainV = runtV[:, 1] / runtV[:, 0]
    bigD['runt_spedup'] = gainV


if __name__ == '__main__':
    args = get_parser()

    nqL = list(range(18, 29))
    nqV = np.array(nqL)
    runLabs = ['gpu', 'cpu']

    runtV, gMD, cMD = read_all_data(nqV, runLabs, args.inpPath)

    expD = {'num_qubit': nqV, 'run_time_1circ': runtV}
    expMD = {'run_label': runLabs}

    for key in ['date', 'hash', 'num_circ', 'num_cx', 'num_gate', 'gpu_info', 'cpu_info']:
        if key in gMD:
            expMD[key] = gMD[key]
        elif key in cMD:
            expMD[key] = cMD[key]
    expMD['run_day'] = expMD['date'].split('_')[0]
    expMD['short_name'] = f'gpuSpeed_{expMD["run_day"]}'

    post_process(expMD, expD)

    args.prjName = expMD['short_name']
    plot = Plotter(args)
    if 'a' in args.showPlots:
        plot.compute_time(expMD, expD, figId=1)
    if 'b' in args.showPlots:
        plot.other_plot(expMD, expD, figId=1)

    plot.display_all(png=1)
