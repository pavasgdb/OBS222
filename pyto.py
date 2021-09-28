#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:33:52 2019

@author: mohit
"""

import serial
import time

i = 0
k0 = 2
k2 = 1

ser = serial.Serial("/dev/tty.usbmodem14101",9600)
print("Trying connection")
connected = False
while not connected:
    serin = ser.read()
    connected = True
print(ord(serin))
print("Success")
k0 = ord(serin)

while k0 == 6:
    k2 = 1
    z = str(i)
    ba = bytes(z, encoding="ascii")
    ser.write(ba)
    #print(type(ba))
    print("Value send to arduino:")
    print(ba)
    #time.sleep(5)
    connected2 = False
    """
    while not connected2:
        serin = ord(ser.read())
        if serin == 4:
            connected2 = True"""
    i = i + 1
    if i == 10:
        i = 0
    print(serin)
    print("Next Value")