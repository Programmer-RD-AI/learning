import socket
import time

ip = socket.gethostbyname(socket.gethostname())
address = 65535
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, address))
while True:
    # print(" > Choices : ")
    # print("""
    # > C = Play with client(client)
    # > S = Play with the server(server)
    # > HS = Playing with server History(Server)
    # > HC = PLaying with client History(Client)
    # > NS = Current marks(Server)
    # > NC = Current marks(Client)
    # """)
    choice = input(" > What is your choice ? \n > ")
    client.send(choice.encode("utf-8"))
    time.sleep(5)
    start = client.recv(1024).decode("utf-8")
    print(start)
    name = input(" > What is your name ? \n > ")
    client.send(name.encode("utf-8"))
    time.sleep(5)
    print("""
         > R = Rock
         > S = Scissors
         > P = Paper
        """)
    choice = input(" > What is your choice ? \n > ").upper()
    client.send(choice.encode("utf-8"))
    time.sleep(5)
    end = client.recv(2500).decode("utf-8")
    print(end)
