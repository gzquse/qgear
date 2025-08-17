#!/usr/bin/env python3
'''
 I'm running a long python program which gradually uses more memory, I want to know how much memory it used in the last moment before it stopped
'''

import psutil
import signal
import time
import numpy as np
import sys
from multiprocessing import Process, Value, Array

# Function to get the current process memory usage in GB
def get_process_memory_usage(pid):
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024 ** 3)  # Convert bytes to GB
    return memory_usage_gb

# Function to get free memory in GB
def get_free_memory():
    memory_info = psutil.virtual_memory()
    free_memory_gb = memory_info.available / (1024 ** 3)  # Convert bytes to GB
    return free_memory_gb

# Memory monitoring function to run in a separate process
def monitor_memory(pid, last_memory_usage, free_memory):
    while True:
        last_memory_usage.value = get_process_memory_usage(pid)
        free_memory.value = get_free_memory()
        time.sleep(1)  # Sample memory usage every second

# Function to handle termination signal
def handle_exit(signum, frame, last_memory_usage, free_memory):
    print(f"\nMemory usage just before stopping: {last_memory_usage.value:.2f} GB")
    print(f"Free memory just before stopping: {free_memory.value:.2f} GB")
    sys.exit(0)  # Use sys.exit to ensure a clean exit

def myMemoryEater():
    data = []
    for i in range(5):  # Adjust this range as needed for your long-running task
        # Simulate some work and memory usage        
        new_data = np.random.rand(400000000)  # Allocate memory
        data.append(new_data)
        
        # Print current process memory usage and free memory
        print(f"advance i={i} process memory usage={get_process_memory_usage(os.getpid()):.2f} GB, free memory={get_free_memory():.2f} GB")
        time.sleep(1)  # Simulate time delay between operations

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":
    import os

    last_memory_usage = Value('d', 0.0)  # Shared memory value to store last memory usage
    free_memory = Value('d', 0.0)  # Shared memory value to store free memory

    # Register the signal handler
    signal.signal(signal.SIGTERM, lambda signum, frame: handle_exit(signum, frame, last_memory_usage, free_memory))
    signal.signal(signal.SIGINT, lambda signum, frame: handle_exit(signum, frame, last_memory_usage, free_memory))

    # Start the memory monitoring process
    monitor_process = Process(target=monitor_memory, args=(os.getpid(), last_memory_usage, free_memory))
    monitor_process.start()

    # Simulating a long-running process
    try:
        myMemoryEater()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        monitor_process.terminate()
        monitor_process.join()

    print('M: end')
    print("Memory usaged=%.2f GB,  free=%.2f GB"%(last_memory_usage.value,free_memory.value))
  
