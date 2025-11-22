from mpi4py import MPI
import numpy as np

rank = MPI.COMM_WORLD.Get_rank()

object1 = [1, 2, 3] 
class Object2:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Object2({self.value})"

object2 = Object2("Class Object") 
object3 = np.array([4, 5, 6])

list_of_objects = [object1, object2, object3]

if rank == 0:
    for worker_rank in range(1, MPI.COMM_WORLD.Get_size()):
        MPI.COMM_WORLD.send(list_of_objects[worker_rank - 1], dest=worker_rank, tag=worker_rank)

elif rank in {1, 2, 3}:
    received_object = MPI.COMM_WORLD.recv(source=0, tag=rank)
    print(f"Worker {rank} received: {received_object}")