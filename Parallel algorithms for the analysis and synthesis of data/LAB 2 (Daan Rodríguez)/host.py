from mpi4py import MPI
import sys

number_of_workers = 2

comm = MPI.COMM_WORLD.Spawn(sys.executable, args=['worker.py'], maxprocs=number_of_workers)

for i in range(number_of_workers):
    message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    print(f"Host received a message: '{message}' from a worker with rank {message}")

comm.Disconnect()
