import threading
import time
import cv2
import os
from datetime import datetime

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

import tkinter as tk
from PIL import Image, ImageTk

from camera import Camera
from recognizer import Recognizer
from security import decrypt_request
from hardware import Relay, Buzzer
from api_client import api_client
from linphone_controller import LinphoneController


PRIMARY_COLOR = "#c2255c"
BG_COLOR = "#0f0f0f"
CARD_COLOR = "#1a1a1a"
TEXT_COLOR = "#ffffff"
MUTED_TEXT = "#aaaaaa"


# ----------------------------
# Flask app (Remote Monitoring)
# ----------------------------
app = Flask(__name__)
CORS(app)


class DeviceServiceLocal:
    """
    Core Smart Doorbell service.
    Shared by:
    - Tkinter UI (local touchscreen)
    - Flask API (remote PC monitoring/control)
    """

    def __init__(self, device_id, base_url):
        self.device_id = device_id if device_id else os.getenv("DEVICE_ID")
        self.base_url = base_url if base_url else os.getenv("BACKEND_URL")

        print("[INFO] DEVICE_ID =", self.device_id)
        print("[INFO] BACKEND_URL =", self.base_url)

        # State
        self.processing = False
        self.call_in_progress = False
        self.local_door_state = "locked"
        self.last_recognition_time = 0
        self.recognition_cooldown = 3

        self.system_status = "booting"
        self.status_message = "Starting system..."
        self.last_person = "None"
        self.last_confidence = 0.0

        # MJPEG Frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Init backend API client
        self._init_api_client()

        # Hardware
        print("[INFO] Initializing hardware...")
        self.relay = Relay(21)
        self.buzzer = Buzzer(13)

        # Camera
        print("[INFO] Initializing camera...")
        self.camera = Camera(resolution=(640, 480), framerate=15)

        # Face Recognition
        print("[INFO] Initializing face recognition...")
        self.recognizer = Recognizer(threshold=0.60, base_url=self.base_url)
        self.face_detector = self.recognizer.face_detector

        # SIP Calling
        print("[INFO] Initializing Linphone...")
        self.linphone = LinphoneController(
            sip_target="6001@10.228.154.143",
            soundcard_id=5,
            on_call_end=self.on_call_ended
        )
        self.linphone.start()

        print("[INFO] Service initialized successfully.")

    def _init_api_client(self):
        global api_client
        from api_client import init_api_client
        api_client = init_api_client(self.base_url, self.device_id, "Smart Doorbell")

    def set_status(self, state, message, person=None, conf=None):
        self.system_status = state
        self.status_message = message

        if person is not None:
            self.last_person = person

        if conf is not None:
            self.last_confidence = conf

    # ----------------------------
    # Camera + Recognition loop
    # ----------------------------
    def start_camera_loop(self):
        self.camera.start_capture()

        self.set_status("loading", "Loading face database...")
        self.recognizer.load_embeddings_from_backend()

        self.set_status("ready", "Monitoring for faces...", person="None", conf=0.0)

        def loop():
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                faces = self.face_detector.detect(frame)
                processed_frame = frame.copy()
                recognized_info = None

                # draw bounding boxes
                for face in faces:
                    startX, startY, endX, endY = face["box"]
                    confidence = face["confidence"]

                    cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(
                        processed_frame,
                        f"Face {confidence*100:.1f}%",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    # crop face region
                    margin = 0.25
                    h, w = frame.shape[:2]
                    dx = int((endX - startX) * margin)
                    dy = int((endY - startY) * margin)

                    x1 = max(0, startX - dx)
                    y1 = max(0, startY - dy)
                    x2 = min(w, endX + dx)
                    y2 = min(h, endY + dy)

                    face_region = frame[y1:y2, x1:x2]

                    if face_region is None or face_region.size == 0:
                        continue
                    if face_region.shape[0] < 30 or face_region.shape[1] < 30:
                        continue

                    face_region = cv2.resize(face_region, (160, 160), interpolation=cv2.INTER_AREA)
                    recognized, info = self.recognizer.recognize_face(face_region)

                    if recognized:
                        recognized_info = info
                        name = info.get("name", "Recognized")
                        rec_conf = float(info.get("confidence", 0.0))

                        cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 255), 3)
                        cv2.putText(
                            processed_frame,
                            f"{name} {rec_conf*100:.1f}%",
                            (startX, startY - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
                        break

                # store latest frame for Tkinter + Flask
                with self.frame_lock:
                    self.latest_frame = processed_frame

                # recognition cooldown
                now = time.time()
                if faces and not self.processing and (now - self.last_recognition_time > self.recognition_cooldown):
                    self.processing = True

                    if recognized_info:
                        self.handle_recognized_person(recognized_info, frame)
                    else:
                        self.handle_unrecognized_person(frame, len(faces))

                    self.last_recognition_time = now
                    self.processing = False

                if not faces and not self.processing:
                    if self.system_status != "ready":
                        self.set_status("ready", "Monitoring for faces...", person="None", conf=0.0)

                time.sleep(0.03)

        threading.Thread(target=loop, daemon=True).start()

    def capture_and_upload(self, frame, person_name="Unknown", status="unrecognized"):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            image_bytes = buffer.tobytes()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name}_{status}_{timestamp}.jpg"

            if api_client:
                return api_client.upload_captured_face(
                    image_bytes=image_bytes,
                    filename=filename,
                    person_name=person_name,
                    status=status
                )
        except Exception as e:
            print("[ERROR] Upload failed:", e)

        return None

    def handle_recognized_person(self, info, frame):
        name = info.get("name", "Unknown")
        conf = float(info.get("confidence", 0.0))

        print(f"[INFO] Recognized: {name} ({conf:.2f})")

        self.set_status("granted", f"Welcome {name}", person=name, conf=conf)
        self.buzzer.beep(100)

        image_url = self.capture_and_upload(frame, name, "recognized")
        if image_url and api_client:
            api_client.send_notification(
                status="recognized",
                image_url=image_url,
                confidence=conf,
                person_name=name
            )

        self.relay.open()
        self.local_door_state = "unlocked"

        time.sleep(5)

        self.relay.close()
        self.local_door_state = "locked"

        self.set_status("ready", "Door locked - Monitoring...", person="None", conf=0.0)

    def handle_unrecognized_person(self, frame, faces_count=1):
        print(f"[INFO] Unrecognized person detected ({faces_count} faces)")

        self.relay.close()
        self.local_door_state = "locked"

        self.set_status("denied", "Unknown person detected!", person="Unknown", conf=0.0)
        self.buzzer.beep(300)

        image_url = self.capture_and_upload(frame, "Unknown", "unrecognized")
        if image_url and api_client:
            api_client.send_notification(
                status="unrecognized",
                image_url=image_url,
                confidence=None,
                person_name="Unknown"
            )

        time.sleep(3)
        self.set_status("ready", "Monitoring for faces...", person="None", conf=0.0)

    # ----------------------------
    # Call Controls
    # ----------------------------
    def initiate_call_to_owner(self):
        if self.call_in_progress:
            return

        self.call_in_progress = True
        self.set_status("calling", "Calling owner...")

        self.buzzer.beep(100)

        try:
            self.linphone.call()
        except Exception as e:
            print("[ERROR] Call failed:", e)
            self.call_in_progress = False
            self.set_status("ready", "Call failed - Monitoring...")

    def hangup_call(self):
        if not self.call_in_progress:
            return
        self.linphone.hangup()

    def on_call_ended(self):
        print("[INFO] Call ended")
        self.call_in_progress = False
        self.set_status("ready", "Call ended - Monitoring...")


# ----------------------------
# Flask Routes (Remote PC)
# ----------------------------
service = None


@app.route("/api/status")
def status():
    if service is None:
        return jsonify({"status": "booting"}), 503

    return jsonify({
        "status": "running",
        "system_status": service.system_status,
        "message": service.status_message,
        "door_state": service.local_door_state,
        "call_in_progress": service.call_in_progress,
        "last_person": service.last_person,
        "confidence": service.last_confidence
    })


@app.route("/video_feed")
def video_feed():
    if service is None:
        return Response("Service not ready", status=503)

    def generator():
        while True:
            with service.frame_lock:
                frame = service.latest_frame

            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       buffer.tobytes() + b"\r\n")

            time.sleep(0.03)

    return Response(generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/door/control", methods=["POST"])
def door_control():
    if service is None:
        return jsonify({"error": "service not ready"}), 503

    data = request.json
    encrypted = data.get("data")

    if not encrypted:
        return jsonify({"error": "missing payload"}), 400

    payload = decrypt_request(encrypted)
    if not payload:
        return jsonify({"error": "invalid or expired request"}), 403

    action = payload["action"]

    if action == "unlock":
        service.local_door_state = "unlocked"
        service.relay.open()
        service.buzzer.beep(200)
        service.set_status("remote_unlock", "Door unlocked remotely", person=service.last_person)
        return jsonify({"status": "unlocked"})

    if action == "lock":
        service.local_door_state = "locked"
        service.relay.close()
        service.set_status("remote_lock", "Door locked remotely", person=service.last_person)
        return jsonify({"status": "locked"})

    return jsonify({"error": "invalid action"}), 400


@app.route("/api/call", methods=["POST"])
def trigger_call():
    if service is None:
        return jsonify({"error": "service not ready"}), 503

    service.initiate_call_to_owner()
    return jsonify({"status": "calling"})


# ----------------------------
# Tkinter UI
# ----------------------------
class TkinterUI:
    def __init__(self, root, service: DeviceServiceLocal):
        self.root = root
        self.service = service

        self.root.configure(bg=BG_COLOR)
        self.root.title("Smart Doorbell")
        self.root.attributes("-fullscreen", True)

        # Main layout
        self.main_frame = tk.Frame(root, bg=BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.right_frame = tk.Frame(self.main_frame, bg=BG_COLOR, width=250)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Video
        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # Status card
        self.status_card = tk.Frame(self.right_frame, bg=CARD_COLOR)
        self.status_card.pack(fill="x", pady=(0, 10))

        self.status_title = tk.Label(
            self.status_card,
            text="SMART DOORBELL",
            font=("Arial", 14, "bold"),
            fg=PRIMARY_COLOR,
            bg=CARD_COLOR
        )
        self.status_title.pack(anchor="w", padx=10, pady=(10, 0))

        self.status_message_label = tk.Label(
            self.status_card,
            text="Booting...",
            font=("Arial", 12),
            fg=TEXT_COLOR,
            bg=CARD_COLOR,
            wraplength=220,
            justify="left"
        )
        self.status_message_label.pack(anchor="w", padx=10, pady=(5, 10))

        self.person_label = tk.Label(
            self.status_card,
            text="Visitor: None",
            font=("Arial", 11),
            fg=MUTED_TEXT,
            bg=CARD_COLOR
        )
        self.person_label.pack(anchor="w", padx=10)

        self.conf_label = tk.Label(
            self.status_card,
            text="Confidence: 0%",
            font=("Arial", 11),
            fg=MUTED_TEXT,
            bg=CARD_COLOR
        )
        self.conf_label.pack(anchor="w", padx=10, pady=(0, 10))

        # Door label
        self.door_label = tk.Label(
            self.right_frame,
            text="DOOR: LOCKED",
            font=("Arial", 14, "bold"),
            fg="white",
            bg=PRIMARY_COLOR,
            padx=10,
            pady=10
        )
        self.door_label.pack(fill="x", pady=(0, 10))

        # Buttons
        self.call_btn = tk.Button(
            self.right_frame,
            text="📞 CALL OWNER",
            font=("Arial", 16, "bold"),
            bg=PRIMARY_COLOR,
            fg="white",
            relief="flat",
            height=2,
            command=self.service.initiate_call_to_owner
        )
        self.call_btn.pack(fill="x", pady=(0, 10))

        self.hangup_btn = tk.Button(
            self.right_frame,
            text="⛔ END CALL",
            font=("Arial", 16, "bold"),
            bg="#444444",
            fg="white",
            relief="flat",
            height=2,
            command=self.service.hangup_call
        )
        self.hangup_btn.pack(fill="x", pady=(0, 10))

        self.exit_btn = tk.Button(
            self.right_frame,
            text="EXIT",
            font=("Arial", 12),
            bg="#222222",
            fg="white",
            relief="flat",
            command=self.root.destroy
        )
        self.exit_btn.pack(fill="x", pady=(20, 0))

        self.update_ui()

    def update_ui(self):
        # Status update
        self.status_message_label.config(text=self.service.status_message)
        self.person_label.config(text=f"Visitor: {self.service.last_person}")
        self.conf_label.config(text=f"Confidence: {self.service.last_confidence*100:.1f}%")

        # Door indicator
        if self.service.local_door_state == "locked":
            self.door_label.config(text="DOOR: LOCKED", bg=PRIMARY_COLOR)
        else:
            self.door_label.config(text="DOOR: UNLOCKED", bg="#2f9e44")

        # Call button state
        if self.service.call_in_progress:
            self.call_btn.config(state="disabled", text="📞 CALLING...")
            self.hangup_btn.config(state="normal")
        else:
            self.call_btn.config(state="normal", text="📞 CALL OWNER")
            self.hangup_btn.config(state="disabled")

        # Video update
        with self.service.frame_lock:
            frame = self.service.latest_frame

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()

            if w > 10 and h > 10:
                img = img.resize((w, h))

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(33, self.update_ui)


# ----------------------------
# Start Flask in background
# ----------------------------
def start_flask():
    print("[INFO] Starting Flask server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    def start_service():
        global service
        service = DeviceServiceLocal(os.getenv("DEVICE_ID"), os.getenv("BACKEND_URL"))
        service.start_camera_loop()

    # Start backend service first
    threading.Thread(target=start_service, daemon=True).start()

    # Start Flask server in background thread
    threading.Thread(target=start_flask, daemon=True).start()

    # Start Tkinter UI (must run in main thread)
    root = tk.Tk()

    # Wait until service is ready
    while service is None:
        time.sleep(0.2)

    ui = TkinterUI(root, service)
    root.mainloop()
