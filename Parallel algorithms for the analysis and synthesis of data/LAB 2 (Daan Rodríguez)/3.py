from mpi4py import MPI
import numpy as np
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create objects
object1 = [1, 2, 3]
class Object2:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Object2({self.value})"

object2 = Object2("Class Object")
object3 = np.array([4, 5, 6])

list_of_objects = [object1, object2, object3]

# Sending time
send_times = []

if rank == 0:
    # Host Process
    for worker_rank in range(1, size):
        start_time = time.time()

        # Send the object to the worker
        comm.send(list_of_objects[worker_rank - 1], dest=worker_rank, tag=worker_rank)

        # Calculate the sending time
        send_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        send_times.append(send_time)

    # Calculate the average sending time
    avg_send_time = sum(send_times) / len(send_times)

    # Print the average sending time
    print(f"Average time taken to send messages: {avg_send_time} milliseconds")

elif rank in range(1, size):
    # Worker Processes
    # Receive the object from the host
    received_object = comm.recv(source=0, tag=rank)

    # Print the received object
    print(f"Worker {rank} received: {received_object}")

