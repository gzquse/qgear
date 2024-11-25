import genQC
from genQC.imports import *
from genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from genQC.inference.export_cudaq import genqc_to_cudaq
import genQC.inference.infer_compilation as infer_comp
import genQC.util as util
import cudaq

# cudaq.set_target('qpp-cpu')  # Note that cpu is faster for 3 qubit kernels

cudaq.set_target('nvidia') # Set to GPU for larger circuits

kernel_list = []
valid_tensors = []

invalid_tensors = 0
for out_tensors_i in tqdm(out_tensors):

    # Use a try-except to catch invalid tensors (if any)
    try:
        kernel = genqc_to_cudaq(out_tensors_i,
                                vocab)  # Convert out_tensors to CUDA-Q kernels
    except:
        kernel = None

    if kernel:
        kernel_list.append(kernel)
        valid_tensors.append(out_tensors_i)
    else:
        invalid_tensors += 1

print(
    f"The model generated {invalid_tensors} invalid tensors that does not correspond to circuits."
)

print(valid_tensors[0])

# Arbitrary input state to the circuit for plotting

input_state = [0] * (2**num_of_qubits)

print(cudaq.draw(kernel_list[0], input_state))

N = 2**num_of_qubits

got_unitaries = np.zeros((len(kernel_list), N, N), dtype=np.complex128)

for i, kernel in tqdm(enumerate(kernel_list), total=got_unitaries.shape[0]):
    for j in range(N):
        basis_state_j = np.zeros((N), dtype=np.complex128)
        basis_state_j[j] = 1

        got_unitaries[i, :,
                      j] = np.array(cudaq.get_state(kernel, basis_state_j),
                                    copy=False)
np.set_printoptions(linewidth=1000)
print(np.round(got_unitaries[0], 4))

def infidelity(want_unitary, got_unitary):
    return 1 - np.abs(
        np.trace(np.conj(want_unitary).T @ got_unitary) / 2**num_of_qubits)**2


infidelities = np.array(
    [infidelity(U, got_unitary) for got_unitary in got_unitaries])

plt.figure(figsize=(7, 4))
plt.title(
    f"Distribution of infidelities for {len(got_unitaries)} generated circuits",
    fontsize=12)
plt.ylabel("Number of circuits", fontsize=14)
plt.xlabel("Unitary infidelity", fontsize=14)
plt.hist(infidelities, bins=30)
plt.show()
plt.savefig('infidelity_histogram.png')

min_index = np.argmin(infidelities)

print(f"The best kernel has an infidelity of {infidelities[min_index]:0.2},")

input_state = [0] * (2**num_of_qubits)
input_state[0] = 1
print(cudaq.draw(kernel_list[min_index], input_state))

print(f"with the unitary:")
print(np.round(got_unitaries[min_index], 4))

print(np.round(U, 4))

# First, we remove possible duplicates and only pick distinct circuits
_, idx_unique = np.unique(np.array(valid_tensors), axis=0, return_index=True)
unique_tensors = torch.stack(valid_tensors)[idx_unique]
unique_infidelities = infidelities[idx_unique]
unique_kernels = [kernel_list[idx] for idx in idx_unique]

# Then, find the correct circuits
idx_correct = torch.argwhere(torch.tensor(unique_infidelities) < 0.01).flatten()
correct_tensors = unique_tensors[idx_correct]
print(
    f"The model generated {correct_tensors.shape[0]} distinct correct circuits."
)

# Now let's flatten the last two dimensions (related to the actual circuit) and find out how many 5's (i.e. ccx) gates each circuit has:
num_ccx = (correct_tensors.flatten(1, 2) == 5).sum(1)
print("These circuits have this number of ccx gates:", num_ccx)

# Get the correct kernels
correct_kernels = [unique_kernels[idx] for idx in idx_correct]

# Get the ones with only one ccx
correct_kernels_ccx1 = [
    correct_kernels[idx] for idx in torch.argwhere(num_ccx == 1).flatten()
]

# Draw a few of this circuits
for kernel in correct_kernels_ccx1[:3]:
    print(cudaq.draw(kernel, input_state))