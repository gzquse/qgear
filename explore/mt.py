
# mpiexec -np 2 python3 mt.py
import random
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def even():
    data = {'rank': rank, 'random': random.randint(1, 100)}
    print(f"Send {data} to {rank + 1}")
    comm.send(data, dest=rank+1)

def odd():
    data = comm.recv(source=rank-1)
    print(f"Got {data} from {rank - 1}")

def main():
    if size % 2 != 0:
        raise RuntimeError("Must run even number of processes")
    if rank%2 == 0:
        even()
    else:
        odd()
    for i in range(size):
        comm.barrier()
        if (i == rank):
            print(f"{rank: 2}")
main()