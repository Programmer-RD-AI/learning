import os
import pickle
import random
import socket
import sys
import time
from _thread import start_new_thread

ip = socket.gethostbyname(socket.gethostname())
port = 65535
server_marks = 0
client_marks = 0
client_address = {}
client_answers = {}
total_clients = 0
st_marks = 0
nd_marks = 0
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    server.bind((ip, port))
    print(" > Waiting for client(s)....")
    server.listen()
except socket.error as e:
    print(f" > Error : {e} ")
    quit()


def answer_check(com_choice, cli_choice, server_marks, client_marks):
    if cli_choice == "R":
        cli_choice = "Rock"
    elif cli_choice == "P":
        cli_choice = "Paper"
    elif cli_choice == "S":
        cli_choice = "Scissors"
    if com_choice == "R":
        com_choice = "Rock"
    elif com_choice == "P":
        com_choice = "Paper"
    elif com_choice == "S":
        com_choice = "Scissors"
    if com_choice == cli_choice:
        connection.send("\n > its a tie !!!....\n".encode("utf-8"))
    elif cli_choice == 'Rock' and com_choice == 'Scissors' or cli_choice == 'Scissors' and com_choice == 'Paper' or cli_choice == 'Paper' and com_choice == 'Rock':
        connection.send("\n > You lost !!!....\n".encode("utf-8"))
        server_marks += 1
    else:
        connection.send("\n > You won !!!....\n".encode("utf-8"))
        client_marks += 1
    connection.send(f"\n > Your Choice : {cli_choice} \n".encode("utf-8"))
    connection.send(f"\n > Servers Choice : {com_choice} \n".encode("utf-8"))


def rpsws(connection, address, total_clients):

    name = connection.recv(1024).decode("utf-8")
    client_address[name] = [connection, address]
    print(f" > Received : {name}")
    while True:
        options = ["R", "P", 'S']
        server_option = random.choice(options)
        print(server_option)
        vailed_choices = ["R", 'S', 'P']
        client_option = connection.recv(1024).decode("utf-8").upper()
        print(client_option)

        def x():
            if client_option in vailed_choices:
                connection.send(" > Please wait for the results...\n".encode("utf-8"))
                connection.send(" > Waiting for results...\n".encode("utf-8"))
                time.sleep(5)
                answer_check(com_choice=server_option, cli_choice=client_option, server_marks=server_marks,
                             client_marks=client_marks)

        if client_option not in vailed_choices:
            pass
        elif client_option == "Q":
            connection.send(" > Disconnecting.....".encode("utf-8"))
            if os.path.exists(address):
                with open(str(address), "rb") as addr:
                    t = pickle.load(addr)
                    with open(str(address), "wb") as ddr:
                        pickle.dump(t + client_marks, ddr)
            else:
                with open(str(address), "wb") as addr:
                    pickle.dump(client_marks, addr)
            total_clients -= 1
            sys.exit(connection)
        elif client_option == "":
            connection.send(" > There is nothing in your choice ! \n")
        else:
            x()


def history_server(address):
    if os.path.exists(address):
        with open(str(address), "rb")as lol:
            olo = pickle.load(lol)
            connection.send(olo)
            file_data = lol.read(1024)
            connection.send(file_data)
            print(f" > Data file has been sent to the client(Address : {address})....")

    else:
        connection.send("\n > You don't have a history...\n".encode("utf-8"))


def current_marks_server(client_marks, server_marks):
    if client_marks > server_marks:
        connection.send(" > You are ahead of the server... \n > Keep up the good gaming....".encode("utf-8"))
    elif client_marks < server_marks:
        connection.send(" > Come on you can do this a little bit more win....".encode("utf-8"))
    elif client_marks == server_marks:
        connection.send(" > You and the server is tied....".encode("utf-8"))
    return f" > Your marks : {client_marks} \n > Server marks : {server_marks} "


def check_answer(st_client, nd_client, st_marks, nd_marks):
    if st_client == "R":
        st_client = "Rock"
    elif st_client == "P":
        st_client = "Paper"
    elif st_client == "S":
        st_client = "Scissors"

    if nd_client == "R":
        nd_client = "Rock"
    elif nd_client == "P":
        com_choice = "Paper"
    elif nd_client == "S":
        com_choice = "Scissors"
    if st_client == nd_client:
        connection.send("\n > its a tie !!!....\n".encode("utf-8"))
        connection_2.send("\n > its a tie !!!....\n".encode("utf-8"))
    # connection1 wins
    elif st_client == 'Rock' and nd_client == 'Scissors' or st_client == 'Scissors' and nd_client == 'Paper' or st_client == 'Paper' and nd_client == 'Rock':
        connection.send("\n > You lost !!!....\n".encode("utf-8"))
        st_marks += 1
    else:
        connection_2.send("\n > You won !!!....\n".encode("utf-8"))
        nd_marks += 1
    connection.send(f"\n > Your Choice : {st_client} \n".encode("utf-8"))
    connection.send(f"\n > Opponent Choice : {nd_client} \n".encode("utf-8"))
    connection.send(f"\n > Your Choice : {nd_client} \n".encode("utf-8"))
    connection.send(f"\n > Opponent Choice : {st_client} \n".encode("utf-8"))


def rpswc(connection_1, connection_2):
    name_1 = connection_1.recv(1024).decode("utf-8")
    name_2 = connection_2.recv(1024).decode("utf-8")
    connection_1.send(f" > Your Opponents username : {name_2} ".encode("utf-8"))
    connection_2.send(f" > Your Opponents username : {name_1} ".encode("utf-8"))
    time.sleep(5)
    choice = connection_1.recv(1024).decode("utf-8")
    choice_2 = connection_2.recv(1024).decode("utf-8")
    check_answer(st_client=choice, nd_client=choice_2, st_marks=st_marks, nd_marks=nd_marks)


def server_history(address_1, address_2):
    if os.path.exists(address_1):
        with open(str(address), "rb")as lol:
            olo = pickle.load(lol)
            connection.send(olo)
            file_data = lol.read(1024)
            connection.send(file_data)
            print(f" > Data file has been sent to the client(Address : {address})....")
    else:
        connection.send("\n > You don't have a history...\n".encode("utf-8"))
    if os.path.exists(address_2):
        with open(str(address), "rb")as lol:
            olo = pickle.load(lol)
            connection.send(olo)
            file_data = lol.read(1024)
            connection.send(file_data)
            print(f" > Data file has been sent to the client(Address : {address})....")
    else:
        connection.send("\n > You don't have a history...\n".encode("utf-8"))



while True:
    connection, address = server.accept()
    total_clients += 1
    print(" > New Client....")
    print(f" > Connection : {connection} ")
    print(f" > Connection address : {address} ")
    print(f" > Total Clients : {total_clients} ")
    option = connection.recv(1024).decode("utf-8").upper()
    start_new_thread(rpsws, (connection, address, total_clients,))
