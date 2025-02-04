# THe start of the client
import socket
import time

ip = socket.gethostbyname(socket.gethostname())
port = 5698
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, port))

while True:
    start = client.recv(512).decode("utf-8")
    print(start)
    time.sleep(5)
    name = input(" > What is your Name ? \n > ")
    client.send(name.encode("utf8"))
    time.sleep(5)
    print(" > Choices : ")
    print(" > R = Rock. ")
    print(" > S = Scissors. ")
    print(" > P = Paper. ")
    choice = input(" > What is your choice  ? \n > ")
    client.send(choice.encode("utf-8"))
    end = client.recv(5500).decode("utf-8")
    print(end)
