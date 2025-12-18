import threading
import time
import cv2
from flask import Flask, Response, jsonify
from camera import Camera
from recognizer import Recognizer
from hardware import Relay, Buzzer, LCD, YellowIndicator, RedIndicator, Button
from api_client import api_client
import os
from datetime import datetime
from queue import Queue

app = Flask(__name__)

class DeviceServiceLocal:
    def __init__(self, device_id, base_url):
        self.device_id = device_id or os.getenv("DEVICE_ID")
        self.base_url = base_url or os.getenv("BACKEND_URL")

        # Camera (REAL-TIME ONLY)
        self.camera = Camera(resolution=(640, 480), framerate=15)

        # Recognition (WORKER ONLY)
        self.recognizer = Recognizer()
        self.face_detector = self.recognizer.face_detector

        # Hardware
        self.relay = Relay(4)
        self.buzzer = Buzzer(27)
        self.yellow = YellowIndicator(13)
        self.red = RedIndicator(21)
        self.button = Button(7)
        self.lcd = LCD()

        # Frame handling
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Queues
        self.recognition_queue = Queue(maxsize=2)

        # State
        self.processing = False
        self.last_button_press = 0
        self.local_door_state = "locked"

        self._init_api_client()
        self._start_gallery_builder()
        self._start_worker()

    # -------------------------
    def _init_api_client(self):
        from api_client import init_api_client
        global api_client
        api_client = init_api_client(self.base_url, self.device_id, "Smart Doorbell")
        self.lcd.display("Smart Doorbell", "Starting...")

    def _start_gallery_builder(self):
        def build():
            self.lcd.display("Loading Faces", "Please wait...")
            self.recognizer.build_gallery_from_device()
            self.lcd.display("Ready", "Door Locked")

        threading.Thread(target=build, daemon=True).start()

    # -------------------------
    # Camera loop (NON-BLOCKING)
    # -------------------------
    def start_camera_loop(self):
        self.camera.start_capture()

        def loop():
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue

                with self.frame_lock:
                    self.latest_frame = frame.copy()

                # Push frame for recognition (non-blocking)
                if not self.recognition_queue.full():
                    self.recognition_queue.put(frame.copy())

                # Button
                if self.button.is_pressed():
                    self.initiate_call_to_owner()

                time.sleep(0.01)

        threading.Thread(target=loop, daemon=True).start()

    # -------------------------
    # Worker thread (HEAVY WORK)
    # -------------------------
    def _start_worker(self):
        def worker():
            while True:
                frame = self.recognition_queue.get()
                self.process_frame(frame)
                self.recognition_queue.task_done()

        threading.Thread(target=worker, daemon=True).start()

    def process_frame(self, frame):
        faces = self.face_detector.detect(frame)
        if not faces:
            return

        recognized, info = self.recognizer.recognize(frame)

        if recognized:
            self.handle_recognized(info, frame)
        else:
            self.handle_unrecognized(frame)

    # -------------------------
    # Door + Notifications
    # -------------------------
    def handle_recognized(self, info, frame):
        name = info.get("name", "Unknown")
        conf = info.get("confidence", 0)

        print(f"[INFO] Recognized: {name} ({conf:.2f})")
        self.lcd.display("Welcome", name[:16])

        self.capture_and_upload(frame, name, "recognized")

        if self.local_door_state == "locked":
            self.red.on()
            self.buzzer.beep(200)
            self.lcd.display("Door Locked", "Access Denied")
            time.sleep(2)
            self.red.off()
            return

        self.relay.open()
        self.yellow.on()
        self.buzzer.beep(100)
        self.lcd.display("Access Granted", "Door Open")
        time.sleep(5)
        self.relay.close()
        self.yellow.off()
        self.lcd.display("Door Locked", "Ready")


    def handle_unrecognized(self, frame):
        self.lcd.display("Access Denied", "Unknown")
        self.red.on()
        self.buzzer.beep(300)
        self.capture_and_upload(frame, "Unknown", "unrecognized")
        time.sleep(2)
        self.red.off()

    # -------------------------
    def capture_and_upload(self, frame, person, status):
        try:
            _, buf = cv2.imencode(".jpg", frame)
            api_client.upload_captured_face(
                image_bytes=buf.tobytes(),
                filename=f"{person}_{status}_{int(time.time())}.jpg",
                person_name=person,
                status=status
            )
        except Exception as e:
            print("[UPLOAD ERROR]", e)

    # -------------------------
    def initiate_call_to_owner(self):
        now = time.time()
        if now - self.last_button_press < 2:
            return
        self.last_button_press = now
        self.lcd.display("Calling Owner", "Please wait")
        api_client.initiate_call()

    # -------------------------
    # MJPEG
    # -------------------------
    def mjpeg_stream(self):
        while True:
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                _, jpg = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
            time.sleep(0.03)


# -------------------------
service = DeviceServiceLocal(
    os.getenv("DEVICE_ID"),
    os.getenv("BACKEND_URL")
)
service.start_camera_loop()

@app.route("/video_feed")
def video_feed():
    return Response(service.mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def status():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    print("[INFO] MJPEG server running on :5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
