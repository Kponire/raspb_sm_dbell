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

# Flask app for local MJPEG streaming
app = Flask(__name__)

# Hardware & device service
class DeviceServiceLocal:
    def __init__(self, device_id, base_url):
        self.device_id = device_id if device_id else os.getenv("DEVICE_ID")
        self.base_url = base_url if base_url else os.getenv("BACKEND_URL")

        # Camera & recognizer
        self.camera = Camera(resolution=(640, 480), framerate=15)
        self.recognizer = Recognizer(model_name="SFace")
        self.face_detector = self.recognizer.face_detector

        # Hardware
        self.relay = Relay(4)
        self.buzzer = Buzzer(27)
        self.yellow_indicator = YellowIndicator(13)
        self.red_indicator = RedIndicator(21)
        self.button = Button(7)
        self.lcd = LCD()

        # State
        self.processing = False
        self.call_in_progress = False
        self.last_button_press = 0
        self.button_debounce_time = 2
        self.local_door_state = "locked"

        # MJPEG frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # API client
        self._init_api_client()

    def _init_api_client(self):
        global api_client
        from api_client import init_api_client
        api_client = init_api_client(self.base_url, self.device_id, "Smart Doorbell")
        self.lcd.display("Smart Doorbell", "Waiting...")

    def start_camera_loop(self):
        """Background thread to capture and process frames"""
        self.camera.start_capture()
        self.lcd.display("Device Powered", "Initializing...")
        self.recognizer.build_gallery_from_device()

        self.lcd.display("Ready", "Door Locked")

        def loop():
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Face detection
                faces = self.face_detector.detect(frame)
                processed_frame = frame.copy()
                recognized_info = None

                for face in faces:
                    startX, startY, endX, endY = face["box"]
                    confidence = face["confidence"]
                    color = (0, 255, 0)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(processed_frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(processed_frame, f"Face {confidence*100:.1f}%", (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                    face_region = frame[startY:endY, startX:endX]
                    if face_region is None or face_region.size == 0:
                        continue

                    h, w = face_region.shape[:2]
                    if h < 30 or w < 30:
                        continue

                    face_region = cv2.resize(face_region, (160, 160), interpolation=cv2.INTER_AREA)
                    recognized, info = self.recognizer.recognize_face(face_region)

                    if recognized:
                        name = info.get("name", "Recognized")
                        rec_conf = info.get("confidence", 0)
                        color = (0, 255, 255)
                        cv2.rectangle(processed_frame, (startX, startY), (endX, endY), color, 3)
                        y_text = y-20 if y-20 > 10 else y+20
                        cv2.putText(processed_frame, f"{name} {rec_conf*100:.1f}%", (startX, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        recognized_info = info
                        break

                # Save frame for streaming
                with self.frame_lock:
                    self.latest_frame = processed_frame

                # Hardware / door handling
                if faces and not self.processing:
                    self.processing = True
                    if recognized_info:
                        self.handle_recognized_person(recognized_info, frame)
                    else:
                        self.handle_unrecognized_person(frame, len(faces))
                    self.processing = False

                # Button check
                if self.button.is_pressed():
                    self.initiate_call_to_owner()
                    time.sleep(0.5)

                time.sleep(0.05)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    # ----------------------------
    # Recognition / Door / Notification
    # ----------------------------
    def capture_and_upload(self, frame, person_name="Unknown", status="unrecognized"):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
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
        conf = info.get("confidence", 0)

        print(f"[INFO] Recognized: {name} ({conf:.2f})")
        self.lcd.display("Welcome", name[:16])
        image_url = self.capture_and_upload(frame, name, "recognized")

        if image_url and api_client:
            api_client.send_notification(
                status="recognized",
                image_url=image_url,
                confidence=conf,
                person_name=name
            )


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


    def handle_unrecognized_person(self, frame, faces_count=1):
        print(f"[INFO] Unrecognized person detected ({faces_count} faces)")
        self.lcd.display("Access Denied", "Unknown Person")
        self.red_indicator.on()
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
        self.red_indicator.off()

    def initiate_call_to_owner(self):
        current_time = time.time()
        if current_time - self.last_button_press < self.button_debounce_time:
            return
        self.last_button_press = current_time
        print("[INFO] Button pressed - initiating call")
        self.lcd.display("Calling Owner...", "Please wait")
        self.yellow_indicator.blink(0.5)
        if api_client and api_client.initiate_call():
            self.call_in_progress = True
            self.lcd.display("Call Initiated", "Talking to owner")
            self.buzzer.beep(100)

    # ----------------------------
    # Flask streaming routes
    # ----------------------------
    def mjpeg_frame_generator(self):
        while True:
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

# ----------------------------
# Flask routes
# ----------------------------
service = DeviceServiceLocal(os.getenv("DEVICE_ID"), os.getenv("BACKEND_URL"))
service.start_camera_loop()

@app.route('/')
def index():
    return "<h1>Raspberry Pi Smart Doorbell</h1><p><img src='/video_feed'></p>"

@app.route('/video_feed')
def video_feed():
    return Response(service.mjpeg_frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({
        "status": "running",
        "camera": "active" if service.camera else "inactive",
        "recognizer": "ready" if service.recognizer else "not_ready"
    })

# ----------------------------
if __name__ == "__main__":
    print("[INFO] Starting Flask server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
