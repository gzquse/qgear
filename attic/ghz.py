import cudaq

def ghz_state(qubit_count, target):
    """A function that will generate a variable sized GHZ state (`qubit_count`)."""
    cudaq.set_target(target)

    kernel = cudaq.make_kernel()

    qubits = kernel.qalloc(qubit_count)

    kernel.h(qubits[0])

    for i in range(1, qubit_count):
        kernel.cx(qubits[0], qubits[i])

    kernel.mz(qubits)

    result = cudaq.sample(kernel, shots_count=1000)

    return result

if __name__ == '__main__':
    #cpu_result = ghz_state(qubit_count=32, target="qpp-cpu")
    
    #cpu_result = ghz_state(qubit_count=32, target="nvidia")
    
    #cpu_result = ghz_state(qubit_count=32, target="nvidia-mgpu")

    cpu_result = ghz_state(qubit_count=30, target="tensornet")
    cpu_result.dump()