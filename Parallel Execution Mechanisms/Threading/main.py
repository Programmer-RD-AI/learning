import threading
import time

def walk_dog(first, last):
    time.sleep(1)
    print(f"Finish Walking: {first} {last}")

def take_out_trash():
    time.sleep(5)
    print("You take out the trash")

def get_mail():
    time.sleep(3)
    print(f"You get the mail")

start_time = time.time()

chore1 = threading.Thread(target=walk_dog, args=("Scooby", "Dobby"))
chore1.start()

chore2 = threading.Thread(target=take_out_trash, daemon=True)
chore2.start()

chore3 = threading.Thread(target=get_mail)
chore3.start()

chore1.join()
chore2.join()
chore3.join()

end_time = time.time()

print(end_time - start_time)
