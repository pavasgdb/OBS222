#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:18:01 2019

@author: mohit
"""
import serial
import time
from collections import deque

q = deque([60,90,120,150])
i= 0
k0 = 2
while k0 == 2:
    ser = serial.Serial("/dev/tty.usbmodem14101",9600)
    print("Trying connection")
    connected = False
    while not connected:
        serin = ser.read()
        connected = True
    print(ord(serin))
    print("Success")
    z = str(i)
    ba = bytes(z, encoding="ascii")
    #time.sleep(1)
    q.append(ba)
    ba1 = q.popleft()
    ser.write(ba1)
    #print(type(ba))
    print("Value send to arduino:")
    print(ba1)
    print("waiting for second connection")
    i = i + 30
    print("Success")
    ## send back to starting of the loop so that when arduino starts a new connection it can start from the begining