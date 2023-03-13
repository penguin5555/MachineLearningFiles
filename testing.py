import serial
import time as t
board = serial.Serial('COM3', baudrate=115200, timeout=1)

board.close()