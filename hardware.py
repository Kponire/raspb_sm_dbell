import os
import time

try:
    import RPi.GPIO as GPIO
    from RPLCD.i2c import CharLCD
    REAL_GPIO = True
except Exception:
    REAL_GPIO = False


if REAL_GPIO:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)


class Relay:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

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
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

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

    def blink(self, interval=0.5):
        if REAL_GPIO:
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HARDWARE SIM] Yellow Indicator {self.pin} BLINK")


class RedIndicator:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

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
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def beep(self, ms: int = 100, repeat: int = 1):
        if REAL_GPIO:
            for _ in range(repeat):
                GPIO.output(self.pin, GPIO.HIGH)
                time.sleep(ms / 1000)
                GPIO.output(self.pin, GPIO.LOW)
                time.sleep(0.05)
        else:
            print(f"[HARDWARE SIM] Buzzer beep {ms}ms x{repeat}")


class LCD:
    def __init__(self):
        if not REAL_GPIO:
            print("[HARDWARE SIM] LCD initialized")
            return

        self.lcd = CharLCD(
            i2c_expander='PCF8574',
            address=0x27,
            port=1,
            cols=16,
            rows=2,
            charmap='A00'
        )

    def display(self, line1: str, line2: str = ""):
        if REAL_GPIO:
            self.lcd.clear()
            self.lcd.write_string(line1[:16])
            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:16])
        else:
            print(f"[LCD] {line1} | {line2}")

    def clear(self):
        if REAL_GPIO:
            self.lcd.clear()


class Button:
    def __init__(self, pin: int):
        self.pin = pin
        if REAL_GPIO:
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def is_pressed(self) -> bool:
        if REAL_GPIO:
            return GPIO.input(self.pin) == GPIO.HIGH
        return False


def cleanup():
    """Call on shutdown"""
    if REAL_GPIO:
        GPIO.cleanup()
