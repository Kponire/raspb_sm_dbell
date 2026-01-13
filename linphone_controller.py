import subprocess
import threading
import time
import os
import signal

class LinphoneController:
    def __init__(self, sip_target, soundcard_id=5):
        self.sip_target = sip_target
        self.soundcard_id = soundcard_id
        self.process = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        if self.running:
            return

        self.process = subprocess.Popen(
            ["linphonec"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        self.running = True

        # Start output reader thread
        threading.Thread(target=self._read_output, daemon=True).start()

        # Allow linphonec to boot
        time.sleep(2)

        # Select soundcard
        self._send(f"soundcard use {self.soundcard_id}")
        time.sleep(1)

        print("[LINPHONE] Ready")

    def _read_output(self):
        for line in self.process.stdout:
            print("[LINPHONE]", line.strip())

    def _send(self, command):
        with self.lock:
            if self.process and self.process.stdin:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()

    def call(self):
        if not self.running:
            self.start()

        print(f"[LINPHONE] Calling {self.sip_target}")
        self._send(f"call {self.sip_target}")

    def hangup(self):
        print("[LINPHONE] Hanging up")
        self._send("terminate")

    def stop(self):
        if not self.running:
            return

        print("[LINPHONE] Stopping")
        self._send("quit")

        try:
            self.process.send_signal(signal.SIGTERM)
        except Exception:
            pass

        self.running = False
