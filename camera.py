import cv2
import numpy as np
from picamera2 import Picamera2
import threading
import time

class Camera:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.resolution = resolution
        self.framerate = framerate
        self.picam2 = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.streaming = False
        self.thread = None
        
    def start_capture(self):
        """Start camera capture in a separate thread"""
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"},
            controls={"FrameRate": self.framerate}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.streaming = True
        
        def capture_loop():
            while self.streaming:
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        with self.frame_lock:
                            self.latest_frame = frame_bgr
                except Exception as e:
                    print(f"[ERROR] Camera capture error: {e}")
                time.sleep(1/self.framerate)
        
        self.thread = threading.Thread(target=capture_loop, daemon=True)
        self.thread.start()
        print(f"[INFO] Camera started at {self.resolution[0]}x{self.resolution[1]} @ {self.framerate}fps")
    
    def read(self):
        """Get the latest frame"""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()
    
    def stop(self):
        """Stop camera capture"""
        self.streaming = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        print("[INFO] Camera stopped")
    
    def get_frame_bytes(self, quality=85):
        """Get frame as JPEG bytes"""
        frame = self.read()
        if frame is None:
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
    
    def get_frame_with_detections(self, detections, recognized_faces=None):
        """Get frame with drawn detection boxes"""
        frame = self.read()
        if frame is None:
            return None
        
        frame_copy = frame.copy()
        
        # Draw detections
        for detection in detections:
            startX, startY, endX, endY = detection['box']
            confidence = detection['confidence']
            
            # Draw rectangle
            color = (0, 255, 0)  # Green for detected faces
            label = "Face"
            
            # Check if this face was recognized
            if recognized_faces:
                for i, (face_box, person_name) in enumerate(recognized_faces.items()):
                    if (abs(startX - face_box[0]) < 20 and abs(startY - face_box[1]) < 20 and
                        abs(endX - face_box[2]) < 20 and abs(endY - face_box[3]) < 20):
                        label = person_name
                        color = (0, 255, 255)  # Yellow for recognized faces
                        break
            
            text = f"{label} {confidence*100:.1f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame_copy, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        return frame_copy