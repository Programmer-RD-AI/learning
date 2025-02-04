import time

print(" < > Welcome to your password Database < > ")
Password = 'Ranuga Disansa'
Command = input(" > ")
while "Hi" != Command:
    Command = input(" > ")
Password_input = input(" > What is the password...\n > ")
while Password != Password_input:
    Password_input = input(" > What is the Password...\n > ")

Name = input(" > what is your name...\n > ")
Age = int(input(" > How old are you...\n > "))
# Only lets children who are Aged more than 18
if Age > 18:
    pass

elif Age < 18:
    quit()

def grret(Name):
    # Greet the User
    print(f" > Hi {Name} ! I am Ranuga Disansa.")


print(grret(" > " + Name))

password_Database = {
    1: {"credit card": ["Ranuga : "  " 62345962312"]},
    2: {"Debit Card": ["Ranuga : "  " 54965621"]},
    3: {"Email Password": ["go2ranuga@gmail.com : "  " Ranugadisansa"]},
    4: {"Indika": {"Credit card": ["Indika prasad : " "45464201"]},
        5: {"Debit Card": ["Indika gamage : 45421000411242"]},
        6: {"Email": ["go2indika@gmail.com : " "Indika gamagehdbfj"]}}
}

print(" > Databases : ")
print(" > Ranuga  ")

if Command.upper() == "HI":
    print("""
 > 1 = Ranuga Credit card.
 > 2 = Ranuga Debit Card.
 > 3 = Ranuga Email.
 > 4 = Indika credit card.
 > 5 = Indika Debit card.
 > 6 = Indika Email
	""")

time.sleep(5)

if Command.upper() == "HI":
    print("""
 > Q = To Quit the programm.
 > S = To Search a Password.
 > A = To Add a Password.
 > R = To Remove a Password.  
 > Se = See the Password.
	""")
Input = input(" > ")
while Input != "Q":
    if Input == "Se":
        for i, v in password_Database.items():
            print(password_Database)

    elif Input.upper() == "S":
        Search = int(input(" > What is the Name...\n > "))
        print("Result : " + str(password_Database[Search]))

    if Search in password_Database:
        print(" > " + str(Search) + " is in the password list... ")
        pass

    elif Search not in password_Database:
        print(" > " + str(Search) + " is Not in the passowrd list...")
        # break
        pass

    elif Input.upper() == "A":
        Name = input.upper()(" > what is the name...\n > ")
        Value = input.upper()(" > what is the number of the name....\n > ")

        if Name and Value in password_Database:
            print("> Item is already in the Database ")

        else:
            password_Database[Name] = [Value]
            pass

    elif Input.upper() == "R":
        What = input(" > What is the Name that you want to Remove... \n > ")
        del password_Database[What]
        print(">" + password_Database)
print(" > Bye Bye....")
print(" > See you later...")
