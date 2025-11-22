from mpi4py import MPI

comm = MPI.Comm.Get_parent()
worker_rank = comm.Get_rank()

comm.send(worker_rank, dest=0, tag=6)
print(f"Worker with rank {worker_rank} has been created.")

comm.Disconnect()
