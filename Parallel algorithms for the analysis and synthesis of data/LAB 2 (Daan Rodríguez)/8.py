from mpi4py import MPI
import time

def custom_sleep(duration=25):
    full_cycles = int(duration / 5)
    for _ in range(full_cycles):
        time.sleep(5)
        print("WAITING")
    remaining_time = duration % 5
    time.sleep(remaining_time)

my_rank = MPI.COMM_WORLD.Get_rank()

if my_rank == 0:
    secret_message = "CONFIDENTIAL"
    send_request = MPI.COMM_WORLD.isend(secret_message, dest=1, tag=0)
    request_completed = False

    while not request_completed:
        custom_sleep(5)
        print("WAITING")
        if send_request.Test():
            request_completed = True
    print("Host 0: Message sent successfully")
    
else:
    receive_request = MPI.COMM_WORLD.irecv(source=0, tag=0)
    custom_sleep()
    received_message = receive_request.wait()
    print(f"Worker {my_rank}: Received message from Host: '{received_message}'")

    response_message = received_message + " back"
    send_response_request = MPI.COMM_WORLD.isend(response_message, dest=0, tag=1)
    print(f"Worker {my_rank}: Sent response message: '{response_message}' to Host")

