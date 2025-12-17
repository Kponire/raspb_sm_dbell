import requests
import json
import base64
from datetime import datetime
import threading
import time
import queue
import cv2
import numpy as np

class APIClient:
    def __init__(self, base_url, device_id, device_name="Raspberry Pi"):
        self.base_url = base_url
        self.device_id = device_id
        self.device_name = device_name
        self.headers = {'Content-Type': 'application/json'}
        self.stream_active = False
        self.door_state = 'locked'
        self.door_state_lock = threading.Lock()
        self.last_door_check = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Start background threads
        self.door_monitor_thread = threading.Thread(target=self._monitor_door_state, daemon=True)
        self.stream_thread = threading.Thread(target=self._stream_frames, daemon=True)
        
        # Register device
        self._register_device()
    
    def initiate_call(self):
        """Initiate a call to the owner through backend"""
        try:
            payload = {
                'deviceId': self.device_id,
                'deviceName': self.device_name,
                'timestamp': datetime.utcnow().isoformat(),
                'callType': 'doorbell_button'
            }
            
            response = requests.post(
                f"{self.base_url}/api/notifications/call",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[INFO] Call initiated: {data.get('message')}")
                return True
            else:
                print(f"[ERROR] Call initiation failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to initiate call: {e}")
            return False
    
    def upload_captured_face(self, image_bytes, filename, person_name="Unknown", status="unrecognized"):
        """Upload captured face to backend for Supabase storage"""
        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            payload = {
                'deviceId': self.device_id,
                'imageData': image_b64,
                'filename': filename,
                'personName': person_name,
                'status': status,
                'bucket': 'captured-faces',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                f"{self.base_url}/api/images/upload-captured",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[INFO] Image uploaded: {data.get('url')}")
                return data.get('url')
            else:
                print(f"[WARN] Image upload failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to upload image: {e}")
            return None
    
    def _register_device(self):
        """Register device with backend"""
        try:
            payload = {
                'deviceId': self.device_id,
                'deviceName': self.device_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                f"{self.base_url}/api/device/register",
                headers=self.headers,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"[INFO] Device registered: {self.device_id}")
            else:
                print(f"[WARN] Device registration failed: {response.status_code}")
                
        except Exception as e:
            print(f"[ERROR] Device registration error: {e}")
    
    def _monitor_door_state(self):
        """Background thread to monitor door state changes"""
        check_interval = 1.0
        
        while True:
            try:
                response = requests.get(
                    f"{self.base_url}/api/door/state/device/{self.device_id}",
                    timeout=3
                )
                
                if response.status_code == 200:
                    data = response.json()
                    new_state = data.get('state', 'locked')
                    
                    with self.door_state_lock:
                        if new_state != self.door_state:
                            print(f"[INFO] Door state changed: {self.door_state} -> {new_state}")
                            self.door_state = new_state
                    
                    self.last_door_check = datetime.utcnow()
                else:
                    print(f"[WARN] Failed to get door state: {response.status_code}")
                    
            except Exception as e:
                print(f"[ERROR] Door state monitoring error: {e}")
            
            time.sleep(check_interval)
    
    def _stream_frames(self):
        """Background thread to stream frames to backend"""
        stream_url = f"{self.base_url}/api/video/stream/{self.device_id}/frame"
        
        while self.stream_active:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                # Get frame from queue
                frame_bytes = self.frame_queue.get_nowait()
                
                # Prepare multipart form data
                files = {'frame': ('frame.jpg', frame_bytes, 'image/jpeg')}
                
                # Send frame
                response = requests.post(
                    stream_url,
                    files=files,
                    timeout=5
                )
                
                if response.status_code != 200:
                    print(f"[WARN] Frame upload failed: {response.status_code}")
                    # Re-add frame to queue if it failed
                    if self.frame_queue.qsize() < 5:
                        self.frame_queue.put(frame_bytes)
                
                time.sleep(0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Streaming error: {e}")
                time.sleep(1)
    
    def send_notification(self, status, image_data=None, image_url=None, confidence=None, person_name=None):
        """Send notification to backend"""
        try:
            payload = {
                'deviceId': self.device_id,
                'status': status,
                'timestamp': datetime.utcnow().isoformat(),
                'confidence': confidence,
                'personName': person_name
            }
            
            if image_url:
                payload['imageUrl'] = image_url
            elif image_data:
                # Convert numpy array to bytes
                if isinstance(image_data, np.ndarray):
                    _, buffer = cv2.imencode('.jpg', image_data)
                    image_bytes = buffer.tobytes()
                    payload['imageData'] = base64.b64encode(image_bytes).decode('utf-8')
            
            response = requests.post(
                f"{self.base_url}/api/notifications/device",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"[INFO] Notification sent: {status}")
                return True
            else:
                print(f"[ERROR] Notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to send notification: {e}")
            return False
    
    def start_streaming(self):
        """Start streaming frames to backend"""
        if not self.stream_active:
            self.stream_active = True
            self.stream_thread.start()
            print(f"[INFO] Started streaming to {self.base_url}")
    
    def stop_streaming(self):
        """Stop streaming"""
        self.stream_active = False
        if self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        print("[INFO] Stopped streaming")
    
    def queue_frame(self, frame):
        """Queue a frame for streaming"""
        if not self.stream_active:
            return False
        
        # Convert frame to JPEG bytes
        if isinstance(frame, np.ndarray):
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Add to queue, drop old frames if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame_bytes)
            return True
        
        return False
    
    def get_door_state(self):
        """Get current door state"""
        with self.door_state_lock:
            return self.door_state
    
    def update_door_state(self, state):
        """Update door state on backend"""
        try:
            payload = {'state': state}
            response = requests.put(
                f"{self.base_url}/api/door/state/device/{self.device_id}",
                headers=self.headers,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                with self.door_state_lock:
                    self.door_state = state
                print(f"[INFO] Door state updated to: {state}")
                return True
            else:
                print(f"[ERROR] Failed to update door state: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to update door state: {e}")
            return False
    
    def start_door_monitoring(self):
        """Start door state monitoring"""
        if not self.door_monitor_thread.is_alive():
            self.door_monitor_thread.start()
            print("[INFO] Started door state monitoring")

# Global instance
api_client = None

def init_api_client(base_url, device_id, device_name="Raspberry Pi"):
    global api_client
    api_client = APIClient(base_url, device_id, device_name)
    api_client.start_door_monitoring()
    return api_client