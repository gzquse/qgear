#!/usr/bin/env python3
"""
Perlmutter Circuit Timing Plot

This script generates a circuit timing plot for Perlmutter data.

Data includes circuit execution times for various numbers of GPUs and qubits.

Saves the resulting plot as a PDF.

"""

# data stored in a dictionary instead of yaml files

__author__ = "Your Name"
__email__ = "your.email@example.com"

import matplotlib.pyplot as plt
from neuron_inverter_benchmark.torch.toolbox.Plotter_Backbone import roys_fontset
import matplotlib.patches as patches

roys_fontset(plt)
def plot_perlmutter_circuit_timing():
    """
    Generates and saves the Perlmutter circuit timing plot.
    """
    # Define the number of qubits for restricted GPUs and all others
    num_qubits_G4 = [31, 32, 33, 34]  # G4 only up to 34
    num_qubits_G8 = [32, 33, 34, 35]  # G8 only up to 35
    num_qubits_G16 = [33, 34, 35, 36]
    num_qubits_G32 = [34, 35, 36, 37]
    num_qubits_G64 = [35, 36, 37, 38] 
    num_qubits_G128 = [36, 37, 38, 39]
    num_qubits_G256 = [37, 38, 39, 40]
    num_qubits_G512 = [38, 39, 40, 41]
    num_qubits_G1024 = [39, 40, 41, 42]

    # Corrected times in seconds
    times_in_sec_corrected = {
        "4": [21, 25, 60, 195],  # Corresponds to num_qubits_G4
        "8": [10, 24, 110, 106],  # Corresponds to num_qubits_G8
        "16": [28, 110, 70, 180],
        "32": [75, 40, 112, 518],
        "64": [25, 65, 290, 575],
        "128": [37, 175, 320, 205],
        "256": [113, 200, 120, 170],
        "512": [125, 75, 110, 740],
        "1024": [55, 255, 450, 245]
    }

    # Convert times to minutes
    times_in_min_corrected = {gpu: [t / 60 for t in times] for gpu, times in times_in_sec_corrected.items()}

    # Mapping the qubits for each GPU to match their specific ranges
    qubit_ranges = {
        "4": num_qubits_G4,
        "8": num_qubits_G8,
        "16": num_qubits_G16,
        "32": num_qubits_G32,
        "64": num_qubits_G64,
        "128": num_qubits_G128,
        "256": num_qubits_G256,
        "512": num_qubits_G512,
        "1024": num_qubits_G1024
    }

    # Define markers for each GPU line
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', 'h']

    # Create the plot
    plt.figure(facecolor='white', figsize=(5.5, 7))

    # Plot each GPU with its specific qubit range
    for (gpu, times), marker in zip(times_in_min_corrected.items(), markers):
        plt.plot(qubit_ranges[gpu], times, marker=marker, label=gpu)

    # Add title, labels, and legend
    plt.title("b)", pad=50)
    plt.xlabel("num qubits")
    plt.ylabel("one circuit simu time (minutes)")
    plt.legend(title="Num GPUs")

    # Highlight the data for qubit 40 with a rectangle until the maximum running time data at 40 qubits
    max_time_at_40 = max(times_in_min_corrected[gpu][qubit_ranges[gpu].index(40)] for gpu in qubit_ranges if 40 in qubit_ranges[gpu])
    rect = patches.Rectangle((39.5, 0), 1, max_time_at_40+0.5, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)

    # Disable grid
    plt.grid(False)

    # Save the plot as a PDF
    plt.savefig("perlmutter_circuit_timing.pdf", format="pdf")
    print("Plot saved as 'perlmutter_circuit_timing.pdf'.")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_perlmutter_circuit_timing()
