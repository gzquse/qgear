#!/usr/bin/env python3
""" 
 concatenate selected metrics from mutiple jobs

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import os
import sys
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from Util_H5io4 import write4_data_hdf5, read4_data_hdf5
from PlotterBackbone import PlotterBackbone
from Util_IOfunc import read_yaml


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-Y", "--noXterm", dest='noXterm', action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("-t", "--drawType", default='par-cpu', choices=["par-cpu", "par-gpu", "adj-gpu"], help="choose how to draw")
    parser.add_argument("-c", "--cpu", type=int, default=0, help="dedicated cpu test out")
    parser.add_argument("-p", "--showPlots", default='a', nargs='+', help="abc-string listing shown plots")

    parser.add_argument("--basePath", default='/pscratch/sd/g/gzquse/quantDataVault2024/dataCudaQ_QEra_July12', help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath", default=None, help="input circuits location")
    parser.add_argument("--outPath", default=None, help="all outputs from experiment")

    args = parser.parse_args()
    if 'env' == args.basePath:
        args.basePath = os.environ['Cudaq_dataVault']
    if args.inpPath is None:
        args.inpPath = os.path.join(args.basePath, 'meas')
    if args.outPath is None:
        args.outPath = os.path.join(args.basePath, 'post')
    for arg in vars(args):
        print('myArg:', arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    args.showPlots = ''.join(args.showPlots)

    return args


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def _make_canvas(self, figId,yIn=5.5):
        figId = self.smart_append(figId)
        fig, ax = self.plt.subplots(facecolor='white', figsize=(5, yIn))  # Increase the figure size
        fig.tight_layout(pad=3.0)  # Adjust layout
        return ax

    def _add_stateSize_axis(self, ax):
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ticks = [28, 32, 36, 40]
        tick_labels = [f'{2**n:.1e}' for n in ticks]
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(tick_labels)
        ax_top.set_xlabel('state-vector size', color='green')
        ax_top.tick_params(axis='x', colors='green')

    def compute_time_par_cpu(self, md, bigD, figId=1):
        ax = self._make_canvas(figId,8.5)
        nqV = bigD['num_qubit']
        runtV = bigD['run_time_1circ'] / 60.0
        dLab = md['run_label']
        dCnt = md['run_count']
        details = md['details']
        nT = len(dLab)
        # nqR = (nqV[0] - 0.5, nqV[-1] + 1.5)
        # max hit for x axis
        nqR = (nqV[0] - 0.5, 40)
        tit = 'cpuA'
        ax.set(xlabel='num qubits', ylabel='compute end-state (minutes)')
        ax.set_title(tit, pad=20)

        for j in range(nT):
            for i in range(len(details)):
                k = dCnt[j]
                valid_indices = ~np.isnan(runtV[:k, j, i])
                if valid_indices.any():
                    ax.plot(nqV[:k][valid_indices], runtV[:k, j, i][valid_indices], label=dLab[j] + ' ,cx:' + str(details[i]['num_cx']), marker='*', linestyle='-',  markersize=10)
                    # Linear extension based on the last 4 valid points
                    if sum(valid_indices) > 1:
                        x1, x2 = nqV[:k][valid_indices][-2:]
                        y1, y2 = np.log(runtV[:k, j, i][valid_indices][-2:])
                        log_slope = (y2 - y1) / (x2 - x1)
                        x_extend = np.arange(x2, 40 + 1)
                        log_y_extend = y2 + log_slope * (x_extend - x2)
                        y_extend = np.exp(log_y_extend)
                        
                        # Limit y_extend to 24 hours (1440 minutes)
                        below_timeout = y_extend <= 1440
                        x_extend = x_extend[below_timeout]
                        y_extend = y_extend[below_timeout]
                        
                        ax.plot(x_extend, y_extend, linestyle='-.', color=ax.get_lines()[-1].get_color())
        # Adding legend for dashed lines
        lines = [plt.Line2D([0], [0], color='black', linestyle='-'),
                 plt.Line2D([0], [0], color='black', linestyle='-.')]
        labels = ['data', 'predict']
        legend = plt.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.4, 0.3))
        
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
        #ax.set_xlim(nqR)
        ax.set_ylim(md['tLH'])
        ax.set_xlim(md['nqLH'])
        ax.grid()
        plt.gca().add_artist(legend)
        ax.legend(loc='upper right', bbox_to_anchor=(0.47, 0.15), fontsize=7.6)
        ax.axhline(1440, ls='--', lw=2, c='m')
        ax.text(34.1, 1000, '24h time-out', c='m')

        # ax.axvline(32, ls='--', lw=2, c='firebrick')
        # ax.text(32.1, 1, 'One A100 RAM limit', c='firebrick', rotation=90)

        # ax.axvline(34, ls='--', lw=2, c='firebrick')
        # ax.text(34.1, 1, 'Four A100s RAM limit', c='firebrick', rotation=90)
        #self._add_stateSize_axis(ax)

        return ax
    def compute_time_cpu(self, ax, md, bigD, figId=1):
        nqV = bigD['num_qubit']
        runtV = bigD['run_time_1circ'] / 60.0
        dLab = md['run_label']
        dCnt = md['run_count']
        details = md['details']
        nT = len(dLab)
        # nqR = (nqV[0] - 0.5, nqV[-1] + 1.5)
        # max hit for x axis
        nqR = (nqV[0] - 0.5, 40)
        tit = 'CPU results'
        ax.set(xlabel='num qubits', ylabel='compute end-state (minutes)')
        ax.set_title(tit, pad=20)

        for j in range(nT):
            for i in range(len(details)):
                k = dCnt[j]
                valid_indices = ~np.isnan(runtV[:k, j, i])
                if valid_indices.any():
                    ax.plot(nqV[:k][valid_indices], runtV[:k, j, i][valid_indices], label=dLab[j] + ' ,cx:' + str(details[i]['num_cx']), marker='*', linestyle='-',  markersize=10)
                    # Linear extension based on the last 4 valid points
                    if sum(valid_indices) > 1:
                        x1, x2 = nqV[:k][valid_indices][-2:]
                        y1, y2 = np.log(runtV[:k, j, i][valid_indices][-2:])
                        log_slope = (y2 - y1) / (x2 - x1)
                        x_extend = np.arange(x2, 40 + 1)
                        log_y_extend = y2 + log_slope * (x_extend - x2)
                        y_extend = np.exp(log_y_extend)
                        
                        # Limit y_extend to 24 hours (1440 minutes)
                        below_timeout = y_extend <= 1440
                        x_extend = x_extend[below_timeout]
                        y_extend = y_extend[below_timeout]
                        
                        ax.plot(x_extend, y_extend, linestyle='-.', color=ax.get_lines()[-1].get_color())
    def compute_time_par_gpu(self, md, bigD, figId=1):
            ax = self._make_canvas(figId,8.5)
            nqV = bigD['num_qubit']
            runtV = bigD['run_time_1circ'] / 60.0
            dLab = md['run_label']
            dCnt = md['run_count']
            details = md['details']
            nT = len(dLab)
            # nqR = (nqV[0] - 0.5, nqV[-1] + 1.5)
            nqR = (nqV[0] - 0.5, 38)
            tit = 'gpuA'
            ax.set(xlabel='num qubits', ylabel='compute end-state (minutes)')
            ax.set_title(tit, pad=20)

            for j in range(nT):
                for i in range(len(details)):
                    k = dCnt[j]
                    valid_indices = ~np.isnan(runtV[:k, j, i])
                    if valid_indices.any():
                        ax.plot(nqV[:k][valid_indices], runtV[:k, j, i][valid_indices], label=dLab[j] + ' ,cx:' + str(details[i]['num_cx']), marker='*', linestyle='-',  markersize=10)
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
            ax.set_ylim(md['tLH'])
            ax.set_xlim(md['nqLH'])
            ax.grid()
            ax.legend(loc='upper right', bbox_to_anchor=(1, 0.3))
            
            #ax.axhline(6, ls='--', lw=2, c='m')
            #ax.text(36.5, 980, '24h time-out', c='m')

            ax.axvline(32, ls='--', lw=2, c='firebrick')
            #ax.text(32.1, 1, 'One A100 RAM limit', c='firebrick', rotation=90)

            #ax.axvline(34, ls='--', lw=2, c='firebrick')
            #ax.text(34.1, 1, 'Four A100s RAM limit', c='firebrick', rotation=90)
            #self._add_stateSize_axis(ax)

            return

    def compute_time_adj_gpu(self, md, bigD, figId=1):
            ax = self._make_canvas(figId,8.5)
            nqV = bigD['num_qubit']
            runtV = bigD['run_time_1circ'] / 60.0
            dLab = md['run_label']
            dCnt = md['run_count']
            details = md['details']
            nT = len(dLab)
            # nqR = (nqV[0] - 0.5, nqV[-1] + 1.5)
            nqR = (nqV[0] - 0.5, 40)
            tit = 'gpuD'
            ax.set(xlabel='num qubits', ylabel='compute end-state (minutes)')
            ax.set_title(tit, pad=20)

            for j in range(nT):
                for i in range(len(details)):
                    k = dCnt[j]
                    valid_indices = ~np.isnan(runtV[:k, j, i]) & (runtV[:k, j, i] > 0)
                    if valid_indices.any():
                        ax.plot(nqV[:k][valid_indices], runtV[:k, j, i][valid_indices], label=dLab[j] + ' ,cx:' + str(details[i]['num_cx']), marker='*', linestyle='-', markersize=10)

            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
            #ax.set_ylim(2e-3,8)
            ax.set_ylim(md['tLH'])
            ax.set_xlim(md['nqLH'])
            ax.grid()
            ax.legend(bbox_to_anchor=(0.5, 0.7))
            # ax.axhline(5, ls='--', lw=2, c='m')
            #ax.text(36.5, 900, '24h time-out', c='m')

            # ax.axvline(32, ls='--', lw=2, c='firebrick')
           # ax.text(32.1, 1, 'One A100 RAM limit', c='firebrick', rotation=90)

            ax.axvline(34, ls='--', lw=2, c='firebrick')
           # ax.text(34.1, 1, 'Four A100s RAM limit', c='firebrick', rotation=90)
            #self._add_stateSize_axis(ax)

            return

def readOne(expN, path, verb):
    if expN.split('_')[-2] == 'tp4':
        path = path.replace('July12','July14')
    assert os.path.exists(path)
    # agnostic fix
    inpF = os.path.join(path, expN + '.yaml')
    print(inpF)
    if not os.path.exists(inpF):
        return 0, 0, {}
    xMD = read_yaml(inpF, verb)
    nq = float(xMD['num_qubit'])
    runt = float(xMD['elapsed_time']) / float(xMD['num_circ'])
    return nq, runt, xMD

def readMany(expN, path, verb):
    # agnostic fix
    path=path.replace('July12','July14')
    assert os.path.exists(path)
    inpFL = [0, 0, 0]
    runt = nq = xMD = inpFL
    for i in range(3):
        if os.path.exists(os.path.join(path, expN[i] + '.yaml')):
            inpFL[i] = os.path.join(path, expN[i] + '.yaml')
            xMD[i] = read_yaml(inpFL[i], verb)
            nq[i] = float(xMD[i]['num_qubit'])
            runt[i] = float(xMD[i]['elapsed_time']) / float(xMD[i]['num_circ'])
    return nq, runt, xMD


def post_process(md, bigD):
    nC = md['num_cpu_runs']
    runtV = bigD['run_time_1circ']
    gainV = runtV[:nC, 0] / runtV[:nC, 1]
    bigD['runt_spedup'] = gainV


if __name__ == '__main__':
    args = get_parser()
    nqL = [i for i in range(28, 35)]
    nqV = np.array(nqL)
    N = nqV.shape[0]
    t = args.drawType
    c = args.cpu
    runLabs = []
    if t == 'par-cpu':
        runLabs.append('par-cpu')
    if t == 'par-gpu':
        runLabs.append('par-gpu')
    if t == 'adj-gpu':
        runLabs.append('adj-gpu')
    # remove 20000 for now
    cxL = (100, 10000)
    nT = len(runLabs)
    nCx = len(cxL)
    cntT = [0] * nT
    mdT = [[None for i in range(nCx)] for j in range(nT)]
    runtV = np.zeros(shape=(N, nT, nCx))
    runtV[:] = np.nan
    shotsV = np.zeros(shape=(N, nT, nCx))
    if c == True:
        # prime types
        expNc = [0 for i in range(3)]
        cruntV = np.zeros(shape=(N, nT, nCx))
    prefix = "mar"
    for i in range(N):
        for j in range(nT):
            for k in range(nCx):
                expN = prefix + '%dq%dcx_%s' % (nqV[i], cxL[k], runLabs[j])
                # if c is True:
                #     # three types
                #     # {0: 64_4 1: 64_1 2: 32_4}
                #     expNc[0] = expN+'_c64_tp4_r0.4'
                #     expNc[1] = expN+'_c64_tp1_r0.4'
                #     expNc[2] = expN+'_c32_tp4_r0.4'
                if t != 'par-gpu':
                    expN += '_r0.4'
                nq, runt, xMD = readOne(expN, args.inpPath, i + j == 0)
                if nq == 0:
                    continue
                if mdT[j][k] is None:
                    mdT[j][k] = xMD
                if j == 2:
                    nqV[i] = nq
                runtV[i, j, k] = runt
                shotsV[i, j, k] = xMD['num_shots']
            cntT[j] += 1
    print('M: got jobs:', cntT)
    expD = {'num_qubit': nqV, 'run_time_1circ': runtV, 'shots': shotsV}
    expMD = {'run_label': runLabs, 'run_count': cntT}
    xMD = mdT[0]
    expMD['details'] = xMD
    for entry in expMD['details']:
        entry['run_day'] = entry['date'].split('_')[0]
    expMD['short_name'] = t + expMD['details'][0]['run_day']
    expMD['run_label'] = runLabs
    pprint(expMD)
    expMD['num_cpu_runs'] = cntT[0]

    args.prjName = expMD['short_name']
    expMD['nqLH']=(27.5,36.5)
    expMD['tLH']=(1e-3,2e3)

    
    plot = Plotter(args)

    if 'a' in args.showPlots and t == 'par-cpu':
        ax = plot.compute_time_par_cpu(expMD, expD, figId=1)
        args = get_parser()
        nqL = [i for i in range(28, 35)]
        nqV = np.array(nqL)
        N = nqV.shape[0]
        t = args.drawType
        c = args.cpu
        runLabs = []
        if t == 'par-cpu':
            runLabs.append('par-cpu')
        if t == 'par-gpu':
            runLabs.append('par-gpu')
        if t == 'adj-gpu':
            runLabs.append('adj-gpu')
        # remove 20000 for now
        cxL = (100, 10000)
        nT = len(runLabs)
        nCx = len(cxL)
        cntT = [0] * nT
        mdT = [[None for i in range(nCx)] for j in range(nT)]
        runtV = np.zeros(shape=(N, nT, nCx))
        runtV[:] = np.nan
        shotsV = np.zeros(shape=(N, nT, nCx))
        if c == True:
            # prime types
            expNc = [0 for i in range(3)]
            cruntV = np.zeros(shape=(N, nT, nCx))
        prefix = "mar"
        for i in range(N):
            for j in range(nT):
                for k in range(nCx):
                    expN = prefix + '%dq%dcx_%s' % (nqV[i], cxL[k], runLabs[j])
                    # if c is True:
                    #     # three types
                    #     # {0: 64_4 1: 64_1 2: 32_4}
                    #     expNc[0] = expN+'_c64_tp4_r0.4'
                    #     expNc[1] = expN+'_c64_tp1_r0.4'
                    #     expNc[2] = expN+'_c32_tp4_r0.4'
                    if t != 'par-gpu' and c == 1:
                        expN += '_c32_tp4_r0.4'
                    nq, runt, xMD = readOne(expN, args.inpPath, i + j == 0)
                    if nq == 0:
                        continue
                    if mdT[j][k] is None:
                        mdT[j][k] = xMD
                    if j == 2:
                        nqV[i] = nq
                    runtV[i, j, k] = runt
                    shotsV[i, j, k] = xMD['num_shots']
                cntT[j] += 1
        print('M: got jobs:', cntT)
        expD = {'num_qubit': nqV, 'run_time_1circ': runtV, 'shots': shotsV}
        expMD = {'run_label': runLabs, 'run_count': cntT}
        xMD = mdT[0]
        expMD['details'] = xMD
        for entry in expMD['details']:
            entry['run_day'] = entry['date'].split('_')[0]
        expMD['short_name'] = t + expMD['details'][0]['run_day']
        expMD['run_label'] = runLabs
        pprint(expMD)
        expMD['num_cpu_runs'] = cntT[0]

        args.prjName = expMD['short_name']
        expMD['nqLH']=(27.5,36.5)
        expMD['tLH']=(1e-3,2e3)

        # plot.compute_time_cpu(ax, expMD, expD, figId=1 )
    if 'a' in args.showPlots and t == 'par-gpu':
        plot.compute_time_par_gpu(expMD, expD, figId=1)
    if 'a' in args.showPlots and t == 'adj-gpu':
        plot.compute_time_adj_gpu(expMD, expD, figId=1)
    if 'b' in args.showPlots:
        plot.sample_plot(expMD, expD, figId=2)
        
    plot.display_all(png=1)