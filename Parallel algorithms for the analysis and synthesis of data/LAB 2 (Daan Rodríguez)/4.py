from mpi4py import MPI
import time

def custom_sleep(duration=10):
    time.sleep(duration)

comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()

if current_rank == 0:
    initial_message = "Que hay de nuevo viejo?"
    start_time_send = time.time()
    comm.send(initial_message, dest=1, tag=0)
    end_time_send = time.time()
    transfer_time_send = end_time_send - start_time_send
    print("Host is engaged in other tasks while waiting for the message processing.")
    custom_sleep()
    print(f"Time taken for message transfer: {transfer_time_send:.2f} seconds")

elif current_rank == 1:
    start_time_receive = time.time()
    received_message = comm.recv(source=0, tag=0)
    end_time_receive = time.time()
    reception_time_receive = end_time_receive - start_time_receive
    print(f"Worker received a message: {received_message}")
    print(f"Time taken for message reception: {reception_time_receive:.2f} seconds")