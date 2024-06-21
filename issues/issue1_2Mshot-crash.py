#!/usr/bin/env python3

# demonstration a 2M shot simu of bell state crashes on A100
import cudaq
print(cudaq.__version__)
qubit_count = 2

@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])
    mz(qvector)

print(cudaq.draw(kernel, qubit_count))


cudaq.set_target("nvidia")
shots=1024*1024 
print('got GPU, run %d shots'%shots)
result = cudaq.sample(kernel, qubit_count, shots_count=shots)
print(result)
