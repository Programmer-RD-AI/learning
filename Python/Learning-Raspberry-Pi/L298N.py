import random
import RPi.GPIO as GPIO          
from time import sleep
import time
in1 = 23
in2 = 24
in3 = 25
in4 = 8
en1 = 12
en2 = 7
TRIG = 4
ECHO = 18
OUT = 20


GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)
GPIO.setup(OUT,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
GPIO.output(OUT,GPIO.LOW)
p1=GPIO.PWM(en1,1000)
p2=GPIO.PWM(en2,1000)

p1.start(25)
p2.start(25)
print("\n")
print("The default speed & direction of motor is LOW & Forward.....")
print("r-run s-stop f-forward b-backward l-low m-medium h-high e-exit")
print("\n")    
a = 0
while(1):
    x = input('Enter : ')
    
    if x=='r':
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        while GPIO.input(ECHO) == False:
            start = time.time()
        while GPIO.input(ECHO) == True:
            end = time.time()
        sig_time = end-start
        distance = sig_time / 0.000058
        # distance = random.choice([1.0,10.0])
        print(f'distance : {distance}')
    
        if distance <= 2500:
            print('HIGH')
            GPIO.output(OUT,GPIO.HIGH)
        else:
            GPIO.output(OUT,GPIO.LOW)
        print("run")
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        x='z'
        a = 1

    elif x=='s':
        print("stop")
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        x='z'

    elif x=='f':
        print("forward")
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        x='z'

    elif x=='b':
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        x='z'

    elif x=='l':
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        while GPIO.input(ECHO) == False:
            start = time.time()
        while GPIO.input(ECHO) == True:
            end = time.time()
        sig_time = end-start
        distance = sig_time / 0.000058
        # distance = random.choice([1.0,10.0])
        print(f'distance : {distance}')
    
        if distance <= 2500:
            print('HIGH')
            GPIO.output(OUT,GPIO.HIGH)
        else:
            GPIO.output(OUT,GPIO.LOW)
        print("low")
        p1.ChangeDutyCycle(25)
        p2.ChangeDutyCycle(25)
        x='z'

    elif x=='m':
        print("medium")
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        while GPIO.input(ECHO) == False:
            start = time.time()
        while GPIO.input(ECHO) == True:
            end = time.time()
        sig_time = end-start
        distance = sig_time / 0.000058
        # distance = random.choice([1.0,10.0])
        print(f'distance : {distance}')
    
        if distance <= 2500:
            print('HIGH')
            GPIO.output(OUT,GPIO.HIGH)
        else:
            GPIO.output(OUT,GPIO.LOW)
        p1.ChangeDutyCycle(50)
        p2.ChangeDutyCycle(50)
        x='z'

    elif x=='h':
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        while GPIO.input(ECHO) == False:
            start = time.time()
        while GPIO.input(ECHO) == True:
            end = time.time()
        sig_time = end-start
        distance = sig_time / 0.000058
        # distance = random.choice([1.0,10.0])
        print(f'distance : {distance}')
    
        if distance <= 2500:
            print('HIGH')
            GPIO.output(OUT,GPIO.HIGH)
        else:
            GPIO.output(OUT,GPIO.LOW)
        print("high")
        p1.ChangeDutyCycle(100)
        p2.ChangeDutyCycle(100)
        x='z'
     
    
    
    elif x=='e':
        GPIO.cleanup()
        print("GPIO Clean up")
        break
    
    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")




