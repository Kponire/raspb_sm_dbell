"""Hardware abstraction for Raspberry Pi: relays, buzzer, LCD, button.
Provides safe fallbacks for non-Raspberry environments for testing on desktop.
"""
import os
try:
    import RPi.GPIO as GPIO
    from RPLCD.i2c import CharLCD
    REAL_GPIO = True
except Exception:
    REAL_GPIO = False


class Relay:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)

    def open(self):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.HIGH)
        else:
            print(f"[HARDWARE SIM] Relay {self.pin} OPEN")

    def close(self):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HARDWARE SIM] Relay {self.pin} CLOSE")

class YellowIndicator:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)

    def on(self):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.HIGH)
        else:
            print(f"[HARDWARE SIM] Yellow Indicator {self.pin} ON")

    def off(self):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HARDWARE SIM] Yellow Indicator {self.pin} OFF")

class RedIndicator:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)

    def on(self):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.HIGH)
        else:
            print(f"[HARDWARE SIM] Red Indicator {self.pin} ON")

    def off(self):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HARDWARE SIM] Red Indicator {self.pin} OFF")


class Buzzer:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.OUT)

    def beep(self, ms: int = 100):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.HIGH)
            GPIO.delay(ms)
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HARDWARE SIM] Buzzer beep {ms}ms")


class LCD:
    def __init__(self):
        self.ADDRESS = 0x27 
        self.PORT = 1 # Use 0 for older Raspberry Pi models
        self.COLS = 16 
        self.ROWS = 2 
        self.CHARMAP = 'A00'
        self.I2C_EXPANDER = 'PCF8574'
        self.lcd = CharLCD(self.I2C_EXPANDER, self.ADDRESS, port=self.PORT, charmap=self.CHARMAP, cols=self.COLS, rows=self.ROWS)

    def display(self, text: str):
        self.lcd.write_string(text)

class Button:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def is_pressed(self) -> bool:
        if REAL_GPIO:
            return GPIO.input(self.pin) == GPIO.HIGH
        return False
