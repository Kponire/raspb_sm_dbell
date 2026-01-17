import threading
import time
import cv2
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from camera import Camera
from security import decrypt_request
from recognizer import Recognizer
from hardware import Button, Relay, Buzzer
from api_client import api_client
import os
from datetime import datetime
from linphone_controller import LinphoneController
from launch_browser import start_chromium

# Flask app with SocketIO for real-time updates
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
# Update SocketIO initialization with better configuration
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   ping_timeout=30,  # Increased ping timeout
                   ping_interval=10,  # Send pings every 10 seconds
                   async_mode='threading',
                   logger=False,  # Enable logging for debugging
                   engineio_logger=False)  # Enable engine.io logging

class DeviceServiceLocal:
    """Smart Doorbell Device Service with web-based UI"""
    
    def __init__(self, device_id, base_url):
        self.device_id = device_id if device_id else os.getenv("DEVICE_ID")
        self.base_url = base_url if base_url else os.getenv("BACKEND_URL")

        print("[INFO] Initializing Camera...")
        self.camera = Camera(resolution=(640, 480), framerate=15)
        
        print("[INFO] Loading Face Recognition...")
        self.recognizer = Recognizer(threshold=0.60, base_url=self.base_url)
        self.face_detector = self.recognizer.face_detector

        print("[INFO] Initializing Hardware...")
        self.relay = Relay(21)
        self.buzzer = Buzzer(13)
        self.button = Button(5)

        # State
        self.processing = False
        self.call_in_progress = False
        self.last_button_press = 0
        self.button_debounce_time = 2
        self.local_door_state = "locked"
        self.last_recognition_time = 0
        self.recognition_cooldown = 3
        self.system_status = "initializing"

        # MJPEG frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        print("[INFO] Connecting to Server...")
        self._init_api_client()
        
        print("[INFO] System Ready!")
        self.system_status = "ready"
        self._emit_status("idle", "System Ready", door_locked=True)

        print("[INFO] Initializing Linphone...")
        self.linphone = LinphoneController(
            sip_target="6001@10.30.132.143",
            soundcard_id=5  # ALSA bcm2835 Headphones
        )
        self.linphone.start()


    def _init_api_client(self):
        """Initialize connection to backend API"""
        global api_client
        from api_client import init_api_client
        api_client = init_api_client(self.base_url, self.device_id, "Smart Doorbell")

    def _emit_status(self, state, message, **kwargs):
        """Emit status update to web UI via SocketIO"""
        data = {
            "state": state,
            "message": message,
            "door_locked": kwargs.get("door_locked", self.local_door_state == "locked"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            **kwargs
        }
        socketio.emit('status_update', data)

    def start_camera_loop(self):
        """Background thread for camera capture and face processing"""
        self.camera.start_capture()
        
        self._emit_status("loading", "Loading Face Database...")
        self.recognizer.load_embeddings_from_backend()
        
        self._emit_status("idle", "Monitoring for faces", door_locked=True)

        def loop():
            while True:
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                faces = self.face_detector.detect(frame)
                processed_frame = frame.copy()
                recognized_info = None
                face_detected = len(faces) > 0

                if face_detected and not self.processing:
                    self._emit_status("detecting", "Face detected - Analyzing...")

                for face in faces:
                    startX, startY, endX, endY = face["box"]
                    confidence = face["confidence"]
                    
                    color = (0, 255, 0)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(processed_frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(processed_frame, f"Face {confidence*100:.1f}%", (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

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

                with self.frame_lock:
                    self.latest_frame = processed_frame

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
                    if self.system_status != "idle":
                        self.system_status = "idle"
                        self._emit_status("idle", "Monitoring for faces", 
                                        door_locked=(self.local_door_state == "locked"))
                if self.button.is_pressed():
                    self.initiate_call_to_owner()
                    time.sleep(0.5)

                time.sleep(0.05)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    def capture_and_upload(self, frame, person_name="Unknown", status="unrecognized"):
        """Capture and upload frame to backend"""
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
        """Handle authorized person detection"""
        name = info.get("name", "Unknown")
        conf = info.get("confidence", 0)
        conf = float(conf) if conf is not None else None
        
        print(f"[INFO] Recognized: {name} ({conf:.2f})")
        
        self._emit_status("access_granted", f"Welcome {name}!", 
                         person_name=name, door_locked=False)
        
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
        self.buzzer.beep(100)
        
        time.sleep(5)
        
        self.relay.close()
        self.local_door_state = "locked"
        self.system_status = "idle"
        self._emit_status("idle", "Door locked - Monitoring", door_locked=True)

    def handle_unrecognized_person(self, frame, faces_count=1):
        """Handle unauthorized person detection"""
        self.relay.close()
        self.local_door_state = "locked"
        
        print(f"[INFO] Unrecognized person detected ({faces_count} faces)")
        
        self._emit_status("access_denied", "Unknown Person Detected", door_locked=True)
        self.buzzer.beep(300)
        
        time.sleep(3)
        self.system_status = "idle"
        self._emit_status("idle", "Monitoring for faces", door_locked=True)

    def initiate_call_to_owner(self):
        """Initiate SIP call using linphonec"""
        current_time = time.time()
        if current_time - self.last_button_press < self.button_debounce_time:
            return 
        if self.call_in_progress:
            return

        self.call_in_progress = True
        print("[INFO] Initiating SIP call to owner")

        self._emit_status("calling", "Calling owner...")
        self.buzzer.beep(100)

        try:
            self.linphone.call()

            # Optional: auto hangup after 30s
            # threading.Timer(30, self._end_call).start()

        except Exception as e:
            print("[ERROR] Call failed:", e)
            self.call_in_progress = False
            self._emit_status("idle", "Call failed",
                            door_locked=(self.local_door_state == "locked"))
    
    def _end_call(self):
        print("[INFO] Ending call")
        self.linphone.hangup()
        self.call_in_progress = False
        self._emit_status("idle", "Monitoring for faces",
                        door_locked=(self.local_door_state == "locked"))

    def mjpeg_frame_generator(self):
        """Generator for MJPEG video stream"""
        while True:
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)


# Initialize service
service = DeviceServiceLocal(os.getenv("DEVICE_ID"), os.getenv("BACKEND_URL"))
service.start_camera_loop()

# service = None

# def start_services():
#     global service
#     service = DeviceServiceLocal(os.getenv("DEVICE_ID"), os.getenv("BACKEND_URL"))
#     service.start_camera_loop()

# # Start heavy services in background
# threading.Thread(target=start_services, daemon=True).start()

# Flask routes
@app.route('/')
def index():
    """Main UI page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG stream for monitoring"""
    if service is None:
        return Response("Service not ready", status=503)
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
    """Remote door control endpoint"""
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
        service._emit_status("unlocked", "Door Unlocked Remotely", door_locked=False)
        threading.Timer(2, lambda: service._emit_status("idle", "Monitoring", 
                       door_locked=False)).start()
        return jsonify({"status": "unlocked"})

    if action == "lock":
        service.local_door_state = "locked"
        service.relay.close()
        service._emit_status("locked", "Door Locked Remotely", door_locked=True)
        threading.Timer(2, lambda: service._emit_status("idle", "Monitoring", 
                       door_locked=True)).start()
        return jsonify({"status": "locked"})

    return jsonify({"error": "invalid action"}), 400

@app.route("/api/call", methods=["POST"])
def trigger_call():
    """API endpoint to trigger call"""
    service.initiate_call_to_owner()
    return jsonify({"status": "call_initiated"})


# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('[INFO] Client connected')
    emit('status_update', {
        'state': service.system_status,
        'message': 'Connected to doorbell',
        'door_locked': service.local_door_state == "locked",
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

@socketio.on('request_status')
def handle_request_status():
    """Send current status to client"""
    emit('status_update', {
        'state': service.system_status,
        'message': 'System Ready',
        'door_locked': service.local_door_state == "locked",
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('[INFO] Client disconnected')

@socketio.on('call_owner')
def handle_call_owner():
    """Handle call button press from UI"""
    service.initiate_call_to_owner()


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
    
    start_chromium()
    # Start keep-alive thread
    keep_alive = threading.Thread(target=keep_alive_thread, daemon=True)
    keep_alive.start()
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)

import atexit

@atexit.register
def cleanup():
    try:
        service.linphone.stop()
    except Exception:
        pass
