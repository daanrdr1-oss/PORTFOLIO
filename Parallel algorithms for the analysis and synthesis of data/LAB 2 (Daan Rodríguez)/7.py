from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

N = 10

assert size == N, f"Number of workers ({size}) != {N}"

circular = list(range(N)) + [0]
original_mensaje = "Hola"
mensaje = original_mensaje

for i in range(N+1):
    if i == rank or i == rank+N:
        if i != 0:
            req = MPI.COMM_WORLD.irecv(source=circular[i-1], tag=i-1)
            received_mensaje = req.wait()
            print(f"Rank {rank} received message: '{received_mensaje}' from Rank {circular[i-1]}.")
            mensaje = received_mensaje
        if i != N:
            MPI.COMM_WORLD.send(mensaje, dest=circular[i+1], tag=i)
            print(f"Rank {rank} sent message to Rank {circular[i+1]}.")

if rank == 0:
    print("DONE")
