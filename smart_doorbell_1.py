import eventlet
eventlet.monkey_patch()

import threading
import time
import cv2
import os
from datetime import datetime

from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from camera import Camera
from recognizer import Recognizer
from hardware import Relay, Buzzer
from security import decrypt_request
from api_client import init_api_client

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-doorbell-secret'

CORS(app)

socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   ping_timeout=30,  # Increased ping timeout
                   ping_interval=10,  # Send pings every 10 seconds
                   async_mode='threading',
                   logger=True,  # Enable logging for debugging
                   engineio_logger=True)  # Enable engine.io logging

class DeviceServiceLocal:
    def __init__(self, device_id, base_url):
        self.device_id = device_id
        self.base_url = base_url

        print("[INFO] Initializing Camera")
        self.camera = Camera(resolution=(640, 480), framerate=10)

        print("[INFO] Initializing Face Recognition")
        self.recognizer = Recognizer(threshold=0.6, base_url=base_url)
        self.face_detector = self.recognizer.face_detector

        print("[INFO] Initializing Hardware")
        self.relay = Relay(4)
        self.buzzer = Buzzer(27)

        # State
        self.system_status = "booting"
        self.local_door_state = "locked"
        self.processing = False
        self.call_in_progress = False

        # Timing
        self.last_recognition_time = 0
        self.recognition_cooldown = 3

        # MJPEG
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        print("[INFO] Initializing API Client")
        self.api_client = init_api_client(
            self.base_url,
            self.device_id,
            "Smart Doorbell"
        )

    def emit_status(self, state, message, **extra):
        payload = {
            "state": state,
            "message": message,
            "door_locked": self.local_door_state == "locked",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            **extra
        }
        socketio.start_background_task(
            socketio.emit, "status_update", payload
        )

    def start_camera_loop(self):
        self.camera.start_capture()
        self.recognizer.load_embeddings_from_backend()

        self.system_status = "idle"

        self.emit_status("idle", "Monitoring for faces")

        def loop():
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.1)
                    continue

                faces = self.face_detector.detect(frame)
                processed = frame.copy()

                with self.frame_lock:
                    self.latest_frame = processed

                now = time.time()

                if faces and not self.processing and now - self.last_recognition_time > self.recognition_cooldown:
                    self.processing = True
                    self.last_recognition_time = now

                    self.emit_status("detecting", "Face detected")

                    face = faces[0]
                    x1, y1, x2, y2 = face["box"]
                    face_img = frame[y1:y2, x1:x2]

                    if face_img.size > 0:
                        face_img = cv2.resize(face_img, (160, 160))
                        recognized, info = self.recognizer.recognize_face(face_img)

                        if recognized:
                            self.handle_recognized(info, frame)
                        else:
                            self.handle_unrecognized(frame)

                    self.processing = False

                time.sleep(0.1)

        threading.Thread(target=loop, daemon=True).start()

    def handle_recognized(self, info, frame):
        name = info.get("name", "Authorized")
        confidence = float(info.get("confidence", 0))

        print(f"[INFO] Recognized {name} ({confidence:.2f})")

        self.emit_status("access_granted", f"Welcome {name}", person_name=name)

        self.relay.open()
        self.local_door_state = "unlocked"
        self.buzzer.beep(100)

        time.sleep(5)

        self.relay.close()
        self.local_door_state = "locked"

        self.emit_status("idle", "Door locked - Monitoring")

    def handle_unrecognized(self, frame):
        print("[WARN] Unknown person detected")

        self.emit_status("access_denied", "Access denied")

        self.buzzer.beep(300)

        time.sleep(3)

        self.emit_status("idle", "Monitoring for faces")

    def initiate_call(self):
        if self.call_in_progress:
            return

        self.call_in_progress = True
        self.emit_status("calling", "Calling owner")

        if self.api_client:
            self.api_client.initiate_call()

        self.buzzer.beep(200)
        time.sleep(3)

        self.call_in_progress = False
        self.emit_status("idle", "Monitoring for faces")

    def mjpeg_stream(self):
        try:
            while True:
                with self.frame_lock:
                    frame = self.latest_frame

                if frame is not None:
                    _, buffer = cv2.imencode(".jpg", frame)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        buffer.tobytes() +
                        b"\r\n"
                    )

                time.sleep(0.05)
        except GeneratorExit:
            print("[INFO] MJPEG client disconnected")


service = DeviceServiceLocal(
    os.getenv("DEVICE_ID"),
    os.getenv("BACKEND_URL")
)

threading.Thread(
    target=service.start_camera_loop,
    daemon=True
).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        service.mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "running",
        "door_state": service.local_door_state
    })

@app.route("/api/call", methods=["POST"])
def api_call():
    service.initiate_call()
    return jsonify({"status": "calling"})


@socketio.on("connect")
def on_connect():
    print("[INFO] Client connected")
    emit("status_update", {
        "state": "idle",
        "message": "System ready",
        "door_locked": service.local_door_state == "locked"
    })

@socketio.on("disconnect")
def on_disconnect():
    print("[INFO] Client disconnected")

@socketio.on("call_owner")
def on_call_owner():
    service.initiate_call()

# Add a thread to send periodic keep-alive messages
def keep_alive_thread():
    """Send periodic keep-alive messages to all clients"""
    while True:
        time.sleep(15)  # Send every 15 seconds
        socketio.emit('keep_alive', {'timestamp': datetime.now().isoformat()})

# Start the keep-alive thread when app starts
if __name__ == "__main__":
    print("[INFO] Starting Smart Doorbell System")
    print("[INFO] Web UI available at http://localhost:5000")
    
    # Start keep-alive thread
    keep_alive = Thread(target=keep_alive_thread, daemon=True)
    keep_alive.start()
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
