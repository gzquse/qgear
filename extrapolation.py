import numpy as np
import matplotlib.pyplot as plt

# ...existing code...

def extrapolate_cpu_long_unitary(n):
    return 2 ** n

# Example usage
n_values = np.arange(0, 10)  # Example range for n
cpu_long_unitary_values = extrapolate_cpu_long_unitary(n_values)

plt.plot(n_values, cpu_long_unitary_values, label='2^n Extrapolation')
plt.xlabel('n')
plt.ylabel('2^n')
plt.legend()
plt.show()

# ...existing code...
