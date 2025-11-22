from mpi4py import MPI
import numpy as np

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

if rank == 0:
    num_elements = int(1e5)
    vector_1, vector_2 = np.ones(num_elements), 2 * np.ones(num_elements)
    fragment_1, fragment_2 = np.array_split(vector_1, size-1), np.array_split(vector_2, size-1)    

    for worker_rank in range(1, size):
        MPI.COMM_WORLD.send(fragment_1[worker_rank - 1], dest=worker_rank, tag=worker_rank)
        MPI.COMM_WORLD.send(fragment_2[worker_rank - 1], dest=worker_rank, tag=worker_rank + 1000)

    dot_product = 0

    for worker_rank in range(1, size):
        partial_result = MPI.COMM_WORLD.recv(source=worker_rank, tag=worker_rank + 2000)
        dot_product += partial_result  

    print(f"Total dot product is {dot_product:.3f}")

else:
    fragment_1 = MPI.COMM_WORLD.recv(source=0, tag=rank)
    fragment_2 = MPI.COMM_WORLD.recv(source=0, tag=rank + 1000)
    partial_result = np.dot(fragment_1, fragment_2)   
    print(f"Worker {rank} partial result is {partial_result:.4f}")

    MPI.COMM_WORLD.send(partial_result, dest=0, tag=rank + 2000)
