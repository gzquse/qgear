#!/usr/bin/env python3
import argparse
import cudaq
from cudaq import spin
import numpy as np
from typing import List, Tuple
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import PlotterBackbone
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-s", "--shift", type=bool, default=True, help="whether shift the dots")

    parser.add_argument("--outPath",default='out',help="all outputs from experiment")
    parser.add_argument("--circName",  default='canImg_b2_32_32', help='gate-list file name')
    parser.add_argument("--prjName", default='noisy_sim', help='project name for output files')
    parser.add_argument("--expName",  default=None,help='(optional) replaces jobID assigned during submission by users choice')

    args = parser.parse_args()
    # make arguments  more flexible
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    args.showPlots=''.join(args.showPlots)

    return args

#............................
#............................
#............................

def run_noise_analysis(
    error_probabilities: np.ndarray,
    num_shots: int = 1000,
    qubit_count: int = 2
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Run noise analysis for different error probabilities and noise types
    Returns expectations for different noise channels
    """
    # Results storage
    depolar_expectations = []
    bitflip_expectations = []
    phaseflip_expectations = []
    amp_damp_expectations = []
    
    # Define measurement operator
    hamiltonian = spin.z(0)
    
    # Basic kernel
    @cudaq.kernel
    def kernel(qubit_count: int):
        qvector = cudaq.qvector(qubit_count)
        x(qvector)
    
    # Set density matrix target
    cudaq.set_target("density-matrix-cpu")
    
    # Iterate through error probabilities
    for p in error_probabilities:
        # Create noise channels
        depolar_channel = cudaq.DepolarizationChannel(p)
        bitflip_channel = cudaq.BitFlipChannel(p)
        phaseflip_channel = cudaq.PhaseFlipChannel(p)
        # amp_damp_channel = cudaq.AmplitudeDampingChannel(p)
        
        # Test each noise type separately
        for channel, result_list in [
            (depolar_channel, depolar_expectations),
            (bitflip_channel, bitflip_expectations),
            (phaseflip_channel, phaseflip_expectations),
            # (amp_damp_channel, amp_damp_expectations)
        ]:
            noise_model = cudaq.NoiseModel()
            noise_model.add_channel("x", [0], channel)
            
            result = cudaq.observe(kernel, hamiltonian, qubit_count, 
                                 noise_model=noise_model)
            result_list.append(result.expectation())
    
    return (depolar_expectations, bitflip_expectations, 
            phaseflip_expectations)

class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    def plot_noise_analysis(self, error_probabilities: np.ndarray, results: Tuple[List[float], ...], figId=1):
        """Plot the noise analysis results with shaded error regions"""
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(10, 6))
        ax = self.plt.subplot(1, 1, 1)
        
        labels = ['Depolarization', 'Bit Flip', 'Phase Flip',]
        colors = ['blue', 'red', 'green', 'purple']
        
        # Plot ideal case
        ax.axhline(y=-1.0, color='black', linestyle='--', label='Ideal')
        
        # Plot each noise type
        for result, label, color in zip(results, labels, colors):
            result_array = np.array(result)
            error = 0.1 * np.abs(result_array)
            
            ax.plot(error_probabilities, result_array, label=label, color=color)
            ax.fill_between(error_probabilities, 
                          result_array - error, 
                          result_array + error, 
                          alpha=0.2, 
                          color=color)
        
        ax.set(xlabel='Error Probability',
               ylabel='Expectation Value',
               title='Noise Channel Analysis')
        ax.legend()
        ax.grid(True)

# Run the benchmark
if __name__ == "__main__":
    args = get_parser()
    error_probabilities = np.linspace(0, 0.5, 20)
    results = run_noise_analysis(error_probabilities)
    plot = Plotter(args)
    plot.plot_noise_analysis(error_probabilities, results)
    plot.display_all(png=1)