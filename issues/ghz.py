# To run this with the mgpu target follow these instructions:
# https://github.com/poojarao8/nersc-quantum-day/blob/master/PerlmutterInstructions.md

import cudaq

def ghz_state(N):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(N)
    kernel.h(q[0])
    for i in range(N - 1):
      kernel.cx(q[i], q[i + 1])
 
    kernel.mz(q)
    return kernel

n = 2 
print("Preparing GHZ state for", n, "qubits.")
kernel = ghz_state(n)
cudaq.set_target("nvidia")
shots=1024*1000
print('got GPU, run %d shots'%shots)

counts = cudaq.sample(kernel, shots_count=shots)
counts.dump()
