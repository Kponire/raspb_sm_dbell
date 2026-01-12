import threading
import time
import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from camera import Camera
from security import decrypt_request
from recognizer import Recognizer
from hardware import Relay, Buzzer
from api_client import api_client
from ui_manager import UIManager
import os
from datetime import datetime

# Flask app for local MJPEG streaming and UI
app = Flask(__name__)
CORS(app)

class DeviceServiceLocal:
    """
    Smart Doorbell Device Service with integrated touchscreen UI
    Handles camera, face recognition, door control, and user interface
    """
    
    def __init__(self, device_id, base_url):
        self.device_id = device_id if device_id else os.getenv("DEVICE_ID")
        self.base_url = base_url if base_url else os.getenv("BACKEND_URL")

        # Initialize UI Manager first for startup feedback
        self.ui = UIManager(primary_color="#c2255c")
        self.ui.show_loading("Initializing System", "Starting Smart Doorbell...")

        # Camera & recognizer
        self.ui.update_loading("Initializing Camera", 20)
        self.camera = Camera(resolution=(640, 480), framerate=15)
        
        self.ui.update_loading("Loading Face Recognition", 40)
        self.recognizer = Recognizer(threshold=0.60, base_url=self.base_url)
        self.face_detector = self.recognizer.face_detector

        # Hardware (relay and buzzer only, no LEDs/Button)
        self.ui.update_loading("Initializing Hardware", 60)
        self.relay = Relay(4)
        self.buzzer = Buzzer(27)

        # State management
        self.processing = False
        self.call_in_progress = False
        self.local_door_state = "locked"
        self.last_recognition_time = 0
        self.recognition_cooldown = 3  # seconds

        # MJPEG frame for browser streaming
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Initialize API client
        self.ui.update_loading("Connecting to Server", 80)
        self._init_api_client()
        
        self.ui.update_loading("System Ready", 100)
        time.sleep(1)

    def _init_api_client(self):
        """Initialize connection to backend API"""
        global api_client
        from api_client import init_api_client
        api_client = init_api_client(self.base_url, self.device_id, "Smart Doorbell")

    def start_camera_loop(self):
        """
        Main background thread for camera capture and face processing
        Continuously monitors for faces and updates the UI
        """
        self.camera.start_capture()
        
        # Load face embeddings from backend
        self.ui.show_status("loading", "Loading Face Database...")
        self.recognizer.load_embeddings_from_backend()
        
        # Show ready state
        self.ui.show_idle(door_locked=True)

        def loop():
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Detect faces in frame
                faces = self.face_detector.detect(frame)
                processed_frame = frame.copy()
                recognized_info = None
                face_detected = len(faces) > 0

                # Update UI with detection status
                if face_detected and not self.processing:
                    self.ui.show_detecting()

                # Process each detected face
                for face in faces:
                    startX, startY, endX, endY = face["box"]
                    confidence = face["confidence"]
                    
                    # Draw face detection box
                    color = (0, 255, 0)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(processed_frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(processed_frame, f"Face {confidence*100:.1f}%", (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                    # Extract face region with margin for better recognition
                    margin = 0.25
                    h, w = frame.shape[:2]
                    dx = int((endX - startX) * margin)
                    dy = int((endY - startY) * margin)
                    x1 = max(0, startX - dx)
                    y1 = max(0, startY - dy)
                    x2 = min(w, endX + dx)
                    y2 = min(h, endY + dy)
                    face_region = frame[y1:y2, x1:x2]

                    # Validate face region
                    if face_region is None or face_region.size == 0:
                        continue
                    h, w = face_region.shape[:2]
                    if h < 30 or w < 30:
                        continue

                    # Prepare face for recognition
                    face_region = cv2.resize(face_region, (160, 160), interpolation=cv2.INTER_AREA)
                    recognized, info = self.recognizer.recognize_face(face_region)

                    if recognized:
                        # Draw recognition indicators on frame
                        name = info.get("name", "Recognized")
                        rec_conf = info.get("confidence", 0)
                        color = (0, 255, 255)
                        cv2.rectangle(processed_frame, (startX, startY), (endX, endY), color, 3)
                        y_text = y-20 if y-20 > 10 else y+20
                        cv2.putText(processed_frame, f"{name} {rec_conf*100:.1f}%", (startX, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        recognized_info = info
                        break

                # Save processed frame for MJPEG streaming
                with self.frame_lock:
                    self.latest_frame = processed_frame

                # Handle recognition logic with cooldown
                current_time = time.time()
                if faces and not self.processing and (current_time - self.last_recognition_time > self.recognition_cooldown):
                    self.processing = True
                    if recognized_info:
                        self.handle_recognized_person(recognized_info, frame)
                    else:
                        self.handle_unrecognized_person(frame, len(faces))
                    self.last_recognition_time = current_time
                    self.processing = False
                elif not faces and not self.processing:
                    # Return to idle when no faces detected
                    self.ui.show_idle(door_locked=(self.local_door_state == "locked"))

                time.sleep(0.05)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    # ----------------------------
    # Image Capture & Upload
    # ----------------------------
    def capture_and_upload(self, frame, person_name="Unknown", status="unrecognized"):
        """
        Capture current frame and upload to backend server
        Returns the uploaded image URL or None on failure
        """
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

    # ----------------------------
    # Recognition Handlers
    # ----------------------------
    def handle_recognized_person(self, info, frame):
        """
        Handle authorized person detection
        - Shows welcome message
        - Unlocks door
        - Sends notification
        - Plays success sound
        """
        name = info.get("name", "Unknown")
        conf = info.get("confidence", 0)
        conf = float(conf) if conf is not None else None
        
        print(f"[INFO] Recognized: {name} ({conf:.2f})")
        
        # Update UI - Access Granted
        self.ui.show_access_granted(name)
        
        # Upload image and send notification
        image_url = self.capture_and_upload(frame, name, "recognized")
        if image_url and api_client:
            api_client.send_notification(
                status="recognized",
                image_url=image_url,
                confidence=conf,
                person_name=name
            )

        # Unlock door
        self.relay.open()
        self.local_door_state = "unlocked"
        self.buzzer.beep(100)
        
        # Keep door open for 5 seconds
        time.sleep(5)
        
        # Lock door and return to ready state
        self.relay.close()
        self.local_door_state = "locked"
        self.ui.show_idle(door_locked=True)

    def handle_unrecognized_person(self, frame, faces_count=1):
        """
        Handle unauthorized person detection
        - Shows access denied message
        - Keeps door locked
        - Plays alert sound
        - Optional: send notification (currently commented out)
        """
        self.relay.close()
        self.local_door_state = "locked"
        
        print(f"[INFO] Unrecognized person detected ({faces_count} faces)")
        
        # Update UI - Access Denied
        self.ui.show_access_denied()
        self.buzzer.beep(300)
        
        # Optional: Upload and notify for unrecognized faces
        # Uncomment if you want notifications for strangers
        # image_url = self.capture_and_upload(frame, "Unknown", "unrecognized")
        # if image_url and api_client:
        #     api_client.send_notification(
        #         status="unrecognized",
        #         image_url=image_url,
        #         confidence=None,
        #         person_name="Unknown"
        #     )
        
        time.sleep(3)
        self.ui.show_idle(door_locked=True)

    def initiate_call_to_owner(self):
        """
        Initiate a call to the homeowner
        Triggered by the call button on the UI
        """
        if self.call_in_progress:
            return
            
        print("[INFO] Initiating call to owner")
        self.ui.show_calling()
        
        if api_client and api_client.initiate_call():
            self.call_in_progress = True
            self.buzzer.beep(100)
            time.sleep(3)
            self.call_in_progress = False
            self.ui.show_idle(door_locked=(self.local_door_state == "locked"))

    # ----------------------------
    # Flask streaming
    # ----------------------------
    def mjpeg_frame_generator(self):
        """
        Generator for MJPEG video stream
        Used for browser-based video feed viewing
        """
        while True:
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)


# ----------------------------
# Initialize Service
# ----------------------------
service = DeviceServiceLocal(os.getenv("DEVICE_ID"), os.getenv("BACKEND_URL"))
service.start_camera_loop()


# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    """Simple HTML page showing the video feed"""
    return "<h1>Raspberry Pi Smart Doorbell</h1><p><img src='/video_feed'></p>"


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint for browser viewing"""
    return Response(service.mjpeg_frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def status():
    """System status endpoint"""
    return jsonify({
        "status": "running",
        "camera": "active" if service.camera else "inactive",
        "recognizer": "ready" if service.recognizer else "not_ready",
        "door_state": service.local_door_state
    })


@app.route("/api/door/control", methods=["POST"])
def door_control():
    """
    Remote door control endpoint
    Accepts encrypted commands to lock/unlock the door
    """
    data = request.json
    encrypted = data.get("data")

    if not encrypted:
        return jsonify({"error": "missing payload"}), 400

    payload = decrypt_request(encrypted)
    print("Decrypted payload:", payload)
    
    if not payload:
        return jsonify({"error": "invalid or expired request"}), 403

    action = payload["action"]

    if action == "unlock":
        service.local_door_state = "unlocked"
        service.relay.open()
        service.ui.show_status("unlocked", "Door Unlocked Remotely")
        time.sleep(2)
        service.ui.show_idle(door_locked=False)
        return jsonify({"status": "unlocked"})

    if action == "lock":
        service.local_door_state = "locked"
        service.relay.close()
        service.ui.show_status("locked", "Door Locked Remotely")
        time.sleep(2)
        service.ui.show_idle(door_locked=True)
        return jsonify({"status": "locked"})

    return jsonify({"error": "invalid action"}), 400


@app.route("/api/call", methods=["POST"])
def trigger_call():
    """API endpoint to trigger call from UI button"""
    service.initiate_call_to_owner()
    return jsonify({"status": "call_initiated"})


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    print("[INFO] Starting Smart Doorbell System")
    print("[INFO] Flask server running on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)