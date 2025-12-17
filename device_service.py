import time
import threading
import cv2
import numpy as np
from camera import Camera
from recognizer import Recognizer, FaceDetector
from hardware import Relay, Buzzer, LCD, YellowIndicator, RedIndicator, Button
from api_client import api_client
import uuid
import os
from datetime import datetime
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class DeviceService:
    def __init__(self, device_id, base_url):
        self.device_id = device_id if device_id else os.getenv('DEVICE_ID')
        self.base_url = base_url if base_url else os.getenv('BACKEND_URL')
        
        # Initialize components
        self.camera = Camera(resolution=(640, 480), framerate=15)
        self.recognizer = Recognizer()
        self.face_detector = FaceDetector()
        
        # Hardware components
        self.relay = Relay(4)
        self.buzzer = Buzzer(27)
        self.yellow_indicator = YellowIndicator(13)
        self.red_indicator = RedIndicator(21)
        self.button = Button(7)
        self.lcd = LCD()
        
        # State management
        self.processing = False
        self.streaming = False
        self.door_locked = True
        
        # Call state
        self.call_in_progress = False
        self.last_button_press = 0
        self.button_debounce_time = 2  # seconds
        
        # Initialize API client
        self._init_api_client()
        
        print(f"[INFO] Device service initialized for device: {device_id}")
    
    def _init_api_client(self):
        """Initialize API client"""
        global api_client
        from api_client import init_api_client
        api_client = init_api_client(self.base_url, self.device_id, "Smart Doorbell")
    
    def capture_and_upload_to_supabase(self, frame, person_name="Unknown", status="unrecognized"):
        """Capture image and upload to Supabase captured-faces bucket"""
        try:
            # Convert frame to JPEG bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_person_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_person_name = safe_person_name.replace(' ', '_')
            filename = f"{safe_person_name}_{status}_{timestamp}.jpg"
            
            # Upload to backend (which will handle Supabase upload)
            if api_client:
                # Send to backend for Supabase upload
                api_client.upload_captured_face(
                    image_bytes=image_bytes,
                    filename=filename,
                    person_name=person_name,
                    status=status
                )
                
                # Return URL pattern (backend will provide actual URL)
                return f"captured-faces/{self.device_id}/{filename}"
            
        except Exception as e:
            print(f"[ERROR] Failed to capture/upload image: {e}")
        
        return None
    
    def initiate_call_to_owner(self):
        """Initiate a call to the owner when button is pressed"""
        if self.call_in_progress:
            print("[INFO] Call already in progress")
            return False
        
        current_time = time.time()
        if current_time - self.last_button_press < self.button_debounce_time:
            print("[INFO] Button debounce - ignoring press")
            return False
        
        self.last_button_press = current_time
        
        print("[INFO] Button pressed - initiating call to owner")
        
        # Visual feedback
        self.lcd.display("Calling Owner...", "Please wait")
        self.yellow_indicator.blink(0.5)  # Blink every 0.5 seconds
        
        try:
            # Call backend to initiate call through Africa's Talking
            if api_client:
                success = api_client.initiate_call()
                if success:
                    self.call_in_progress = True
                    self.lcd.display("Call Initiated", "Talking to owner")
                    self.buzzer.beep(100)
                    return True
                else:
                    self.lcd.display("Call Failed", "Try again")
                    self.red_indicator.on()
                    self.buzzer.beep(300, 3)
                    time.sleep(2)
                    self.red_indicator.off()
                    
        except Exception as e:
            print(f"[ERROR] Failed to initiate call: {e}")
            self.lcd.display("Call Error", "Check backend")
            self.red_indicator.on()
            time.sleep(2)
            self.red_indicator.off()
        
        return False
    
    def end_call(self):
        """End the current call"""
        if self.call_in_progress:
            print("[INFO] Ending call")
            self.call_in_progress = False
            self.yellow_indicator.off()
            self.lcd.display("Call Ended", "Ready")
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        if frame is None:
            return None, None
        
        # Get current door state
        door_state = api_client.get_door_state() if api_client else 'locked'
        
        # Detect faces using DNN
        faces = self.face_detector.detect(frame)
        
        # Prepare frame for streaming (with detection boxes)
        stream_frame = frame.copy()
        recognized_info = None
        
        if faces:
            # Try recognition on detected faces
            for face in faces:
                startX, startY, endX, endY = face['box']
                confidence = face['confidence']
                
                # Draw detection box
                color = (0, 255, 0)  # Green for detection
                text = f"Face {confidence*100:.1f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                
                cv2.rectangle(stream_frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(stream_frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                # Try recognition
                recognized, info = self.recognizer.recognize(frame)
                if recognized:
                    # Draw recognition box
                    name = info.get('name', 'Recognized')
                    rec_confidence = info.get('confidence', 0)
                    color = (0, 255, 255)  # Yellow for recognized
                    text = f"{name} {rec_confidence*100:.1f}%"
                    
                    cv2.rectangle(stream_frame, (startX, startY), (endX, endY), color, 3)
                    cv2.putText(stream_frame, text, (startX, y-20 if y-20 > 10 else y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    recognized_info = info
                    break
        
        # Queue frame for streaming
        if api_client and self.streaming:
            api_client.queue_frame(stream_frame)
        
        return faces, recognized_info
    
    def handle_recognized_person(self, info, frame):
        """Handle actions when a person is recognized"""
        name = info.get('name', 'Unknown')
        confidence = info.get('confidence', 0)
        
        print(f"[INFO] Recognized: {name} (confidence: {confidence:.2f})")
        
        # Update LCD
        self.lcd.display("Welcome", name[:16])
        
        # Capture and upload image
        image_url = self.capture_and_upload_to_supabase(frame, name, "recognized")
        
        # Check door state
        door_state = api_client.get_door_state() if api_client else 'locked'
        
        if door_state == 'locked':
            # Door is locked, don't open
            self.lcd.display("Door Locked", "Access Denied")
            self.red_indicator.on()
            self.buzzer.beep(200)
            
            # Send notification
            if api_client:
                api_client.send_notification(
                    status='recognized_denied',
                    image_data=frame,
                    confidence=confidence,
                    person_name=name,
                    image_url=image_url
                )
            
            time.sleep(3)
            self.red_indicator.off()
            
        else:
            # Door is unlocked, grant access
            self.relay.open()
            self.yellow_indicator.on()
            self.buzzer.beep(100)
            
            # Send notification
            if api_client:
                api_client.send_notification(
                    status='recognized_granted',
                    image_data=frame,
                    confidence=confidence,
                    person_name=name,
                    image_url=image_url
                )
            
            time.sleep(5)
            self.relay.close()
            self.yellow_indicator.off()
    
    def handle_unrecognized_person(self, frame, faces_count=1):
        """Handle actions when no person is recognized"""
        print(f"[INFO] Unrecognized person detected ({faces_count} faces)")
        
        self.lcd.display("Access Denied", "Unknown Person")
        self.red_indicator.on()
        self.buzzer.beep(300)
        
        # Capture and upload image
        image_url = self.capture_and_upload_to_supabase(frame, "Unknown", "unrecognized")
        
        # Send notification
        if api_client:
            api_client.send_notification(
                status='unrecognized',
                image_data=frame,
                image_url=image_url
            )
        
        time.sleep(3)
        self.red_indicator.off()
    
    def start_streaming(self):
        """Start streaming to backend"""
        self.streaming = True
        if api_client:
            api_client.start_streaming()
        print("[INFO] Started video streaming")
    
    def stop_streaming(self):
        """Stop streaming"""
        self.streaming = False
        if api_client:
            api_client.stop_streaming()
        print("[INFO] Stopped video streaming")
    
    def start(self):
        """Main device service loop"""
        # Initialize
        self.camera.start_capture()
        self.lcd.display("Device Powered", "Initializing...")
        
        # Build recognition gallery
        print("[INFO] Building face recognition gallery...")
        self.recognizer.build_gallery_from_device()
        
        # Start streaming
        self.start_streaming()
        
        self.lcd.display("Ready", "Door Locked")
        print("[INFO] Device service started")
        
        # Main processing loop
        try:
            while True:
                # Read frame from camera
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame (detection, recognition, streaming)
                faces, recognized_info = self.process_frame(frame)
                
                # Check for button press (manual call to owner)
                if self.button.is_pressed():
                    self.initiate_call_to_owner()
                    # Debounce - wait a bit after button press
                    time.sleep(0.5)
                
                # If faces detected and not currently processing
                if faces and not self.processing:
                    self.processing = True
                    
                    if recognized_info:
                        self.handle_recognized_person(recognized_info, frame)
                    else:
                        self.handle_unrecognized_person(frame, len(faces))
                    
                    self.processing = False
                    time.sleep(2)  # Cooldown after processing
                
                # Small delay to control processing rate
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("[INFO] Shutting down...")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_streaming()
            self.camera.stop()
            self.lcd.clear()
            print("[INFO] Device service stopped")


if __name__ == "__main__":
    DEVICE_ID = os.getenv("DEVICE_ID", "raspberrypi-doorbell-001")
    BASE_URL = os.getenv("BACKEND_URL", "https://rasp-sm-dbell-web-vrs.onrender.com")

    service = DeviceService(
        device_id=DEVICE_ID,
        base_url=BASE_URL
    )

    service.start()