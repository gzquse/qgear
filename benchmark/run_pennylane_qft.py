import pennylane as qml
import random
from time import time

wires = 33
dev = qml.device("lightning.gpu", wires=wires, mpi=True)

@qml.qnode(dev)
def circuit_qft(basis_state):
    qml.BasisState(basis_state, wires=range(wires))
    qml.QFT(wires=range(wires))
    return qml.state()

input_state = [random.choice([0, 1]) for i in range(wires)]
T0=time()
circuit_qft(input_state)
print(time()-T0)

# from mpi4py import MPI
# import pennylane as qml
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# dev = qml.device('lightning.gpu', wires=8)
# @qml.qnode(dev)
# def circuit_mpi():
#     qml.PauliX(wires=[0])
#     return qml.state()
# local_state_vector = circuit_mpi()
# #rank 0 will collect the local state vector
# state_vector = comm.gather(local_state_vector, root=0)
# if rank == 0:
#     print(state_vector)