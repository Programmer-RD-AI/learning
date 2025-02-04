import os
import pickle
import socket
import sys
from _thread import *
import random
import time

ip = socket.gethostbyname(socket.gethostname())
port = 5698
conection = True
client_answers = {}
client_info = {}
server_marks = 0
client_marks = 0
totatl_client = 0
client_1 = 0
client_2 = 0
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    server.bind((ip, port))
    print(" > Waiting for client(s)....")
    server.listen()
except socket.error as e:
    print(f" > Error : {e} ")
    quit()


def rps_s(connection, server_marks, client_marks):
    connection.send("! | > WELCOME TO ROCK PAPER SCISSORS SERVER < | ! \n".encode("utf-8"))
    while True:
        name = connection.recv(1024).decode("utf-8")
        connection.send(f"\n > Hi {name} \n".encode("utf-8"))
        time.sleep(5)
        choice = connection.recv(1024).decode("utf-8")
        choices = ['R', 'P', 'S', "s", "p", "r"]
        com_choice = random.choice(choices)
        if choice in choices:
            if choice == com_choice:
                connection.send('\n > It"s a tie...\n'.encode("utf-8"))
            elif choice == 'R':
                if com_choice == 'P':
                    connection.send('\n > You lose, sorry :(\n'.encode("utf-8"))
                    server_marks += 1
                elif com_choice == 'S':
                    connection.send('\n > You won!!!!!\n'.encode("utf-8"))
                    client_marks += 1
            elif choice == 'P':
                if com_choice == 'S':
                    connection.send('\n > You lose, sorry :(\n'.encode("utf-8"))
                    server_marks += 1
                elif com_choice == 'R':
                    connection.send('\n > You won !!!\n'.encode("utf-8"))
                    client_marks += 1
            elif choice == "S":
                if com_choice == 'R':
                    connection.send('\n > You lose, sorry :(\n'.encode("utf-8"))
                    server_marks += 1
                elif com_choice == 'P':
                    connection.send('\n > You won!!!!\n'.encode("utf-8"))
                    client_marks += 1
            if choice == "R" or choice == "r":
                choice = "Rock"
            elif choice == "P" or choice == "p":
                choice = "Paper"
            elif choice == "S" or choice == "s":
                choice = "Scissors"

            if com_choice == "R" or com_choice == "r":
                com_choice = "Rock"
            elif com_choice == "P" or com_choice == "p":
                com_choice = "Paper"
            elif com_choice == "S" or com_choice == "s":
                com_choice = "Scissors"
            connection.send(f"\n > Your Choice : {choice} \n > Servers Choice : {com_choice} ".encode("utf-8"))
        elif choice == "Q":
            connection.send(" > Disconnecting server....".encode("utf-8"))
            connection.send(" > Bye Bye ".encode("utf-8"))
            sys.exit(connection)
        else:
            connection.send('\n > Invalid choice \n > try again..\n'.encode("utf-8"))
        print(f"\n > Your Choice : {choice} \n > Servers Choice : {com_choice} ")
        if choice < com_choice:
            connection.send("\n > You are ahead of the server... \n > Keep up the good gaming....\n".encode("utf-8"))
        elif choice > com_choice:
            connection.send("\n > Come on you can do this a little bit more win....\n".encode("utf-8"))
        elif choice == com_choice:
            connection.send("\n > You and the server is tied....\n".encode("utf-8"))
        return connection.send(
            f"\n > Your marks : {client_marks} \n > Server marks : {server_marks} \n".encode("utf-8"))


while True:
    connection, address = server.accept()
    totatl_client += 1
    print(" > New Connection !!!")
    print(f" > Connection : {connection} ")
    print(f" > Address :  {address} ")
    start_new_thread(rps_s, (connection, server_marks, client_marks))
