from flask import Flask, Response, jsonify
import threading
import time
import cv2
import numpy as np
from camera import Camera
from recognizer import Recognizer

app = Flask(__name__)

# Global variables for local streaming
camera = None
recognizer = None
latest_frame = None
frame_lock = threading.Lock()

def init_camera():
    """Initialize camera in background thread"""
    global camera, recognizer
    camera = Camera()
    camera.start_capture()
    
    recognizer = Recognizer()
    recognizer.build_gallery_from_device()
    
    # Start frame capture thread
    def capture_thread():
        global latest_frame
        while True:
            frame = camera.read()
            if frame is not None:
                # Process for recognition
                faces = recognizer.face_detector.detect(frame)
                processed_frame = frame.copy()
                
                # Draw detections
                for face in faces:
                    startX, startY, endX, endY = face['box']
                    confidence = face['confidence']
                    
                    color = (0, 255, 0)
                    text = f"Face {confidence*100:.1f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    
                    cv2.rectangle(processed_frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(processed_frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                with frame_lock:
                    latest_frame = processed_frame
            
            time.sleep(0.033)  # ~30 FPS
    
    thread = threading.Thread(target=capture_thread, daemon=True)
    thread.start()
    print("[INFO] Local camera stream initialized")

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Raspberry Pi Camera Stream</title>
        </head>
        <body>
            <h1>Live Camera Stream</h1>
            <img src="/video_feed" width="640" height="480">
            <p><a href="/api/status">Status</a></p>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route for local LAN access"""
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', latest_frame)
                    frame_bytes = buffer.tobytes()
                else:
                    time.sleep(0.1)
                    continue
            
            # MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get device status"""
    return jsonify({
        'status': 'running',
        'camera': 'active' if camera else 'inactive',
        'recognizer': 'ready' if recognizer else 'not_ready',
        'streaming': True
    })

@app.route('/api/capture', methods=['POST'])
def capture_image():
    """Capture a single image"""
    with frame_lock:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    return jsonify({'error': 'No frame available'}), 404

if __name__ == '__main__':
    # Initialize camera
    init_camera()
    
    # Start Flask server for local LAN access
    print("[INFO] Starting local server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)