import cudaq
from cudaq import spin

cudaq.set_target("nvidia-mqpu")


qubit_count = 30
term_count = 1000

@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)


# We create a random hamiltonian
# hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

# The observe calls allows us to calculate the expectation value of the Hamiltonian with respect to a specified kernel.

# Single node, single GPU.
# result = cudaq.observe(kernel, hamiltonian)
# result.expectation()

# If we have multiple GPUs/ QPUs available, we can parallelize the workflow with the addition of an argument in the observe call.

# Single node, multi-GPU.
# result = cudaq.observe(kernel, hamiltonian, execution=cudaq.parallel.thread)
# result.expectation()
result = cudaq.sample(kernel, qubit_count, shots_count=term_count)

# Multi-node, multi-GPU.
# result = cudaq.observe(kernel, hamiltonian, execution=cudaq.parallel.mpi)
# result.expectation()
