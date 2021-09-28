#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:03:16 2019

@author: mohit
"""

import serial
import time

ser = serial.Serial("/dev/tty.usbmodem14101",9600)
print("Trying to connect")
connected = False
while not connected:
    serin = ser.read()
    connected = True
# making sure that it is connected and changing the value of connected to true
print("Connected with Arduino")

ii = 30
tt = time.time()
z1 = str(ii)
ser.write(bytes(z1, encoding="ascii"))
k2 = int(ser.read())
lo = 1
print(lo,k2) 
while lo == 1:
    if (time.time()-tt) > 4:
        ii = ii + 30
        tt = time.time()
        print(tt)
        z = str(ii)
        ba = bytes(z, encoding="ascii")
        #print((X,Y))
        #print((X*width,Y*height))
        ser.write(ba)
        print(ba)
        k2 = int(ser.read())
        print(k2)        