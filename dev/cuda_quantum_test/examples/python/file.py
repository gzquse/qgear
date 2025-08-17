import cudaq

cudaq.mpi.initialize()
cudaq.set_target("tensornet")
# kernel = cudaq.make_kernel()
# qubits = kernel.qalloc(100)
# kernel.h(qubits[0])
# for i in range(99):
#     kernel.cx(qubits[i], qubits[i + 1])
# kernel.mz(qubits)
# result = cudaq.sample(kernel, shots_count=100)
# if cudaq.mpi.rank() == 0:
#     print(result)
# cudaq.mpi.finalize()