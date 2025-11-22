from mpi4py import MPI
from sys import getsizeof
import time

comm = MPI.COMM_WORLD
rango = comm.Get_rank()
tamaño = comm.Get_size()

N = 10

if rango == 0:
    for i in range(1, 51):
        obj = [10] * (1000 * i)  
        L = getsizeof(obj)  
        T = time.time()

        for j in range(N):
            comm.send(obj, dest=1, tag=i + j)
            obj = comm.recv(source=1, tag=i + j + 1)
            
        T = time.time() - T
        R = (2 * N * L) / T / (1024 * 1024)
        print(f"Iter {i}: Object {L} bytes: {R:.2f} MB/s")

elif rango == 1:
    for i in range(1, 51):
        for j in range(N):
            mensaje = comm.recv(source=0, tag=i + j)
            comm.send(mensaje, dest=0, tag=i + j + 1)
