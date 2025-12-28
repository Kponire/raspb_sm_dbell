import base64
from datetime import datetime
import cv2
import numpy as np
import requests


class APIClient:
    def __init__(self, base_url, device_id, device_name="Raspberry Pi"):
        self.base_url = base_url.rstrip("/")
        self.device_id = device_id
        self.device_name = device_name
        self.headers = {'Content-Type': 'application/json'}

    def initiate_call(self):
        try:
            payload = {
                'deviceId': self.device_id,
                'deviceName': self.device_name,
                'timestamp': datetime.utcnow().isoformat(),
                'callType': 'doorbell_button'
            }

            r = requests.post(
                f"{self.base_url}/api/notifications/call",
                json=payload,
                headers=self.headers,
            )

            return r.status_code == 200

        except Exception as e:
            print("[ERROR] Call initiation failed:", e)
            return False

    def upload_captured_face(self, image_bytes, filename,
                             person_name="Unknown", status="unrecognized"):
        try:
            payload = {
                'deviceId': self.device_id,
                'imageData': base64.b64encode(image_bytes).decode(),
                'filename': filename,
                'personName': person_name,
                'status': status,
                'bucket': 'captured-faces',
                'timestamp': datetime.utcnow().isoformat()
            }

            r = requests.post(
                f"{self.base_url}/api/images/upload-captured",
                json=payload,
                headers=self.headers,
            )

            if r.status_code == 200:
                url = r.json().get("url")
                print("[INFO] Image uploaded:", url)
                return url
            else:
                print(f"[WARN] Image upload failed: {r.status_code}")
                return None

        except Exception as e:
            print("[ERROR] Image upload failed:", e)

        return None

    def send_notification(self, status, image_url, confidence=None, person_name=None):
        """Send notification to backend (URL ONLY)"""
        try:
            payload = {
                'deviceId': self.device_id,
                'status': status,
                'imageUrl': image_url,
                'timestamp': datetime.utcnow().isoformat(),
                'confidence': confidence,
                'personName': person_name
            }

            response = requests.post(
                f"{self.base_url}/api/notifications/device",
                headers=self.headers,
                json=payload,
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

# Global instance
api_client = None

def init_api_client(base_url, device_id, device_name="Raspberry Pi"):
    global api_client
    api_client = APIClient(base_url, device_id, device_name)
    return api_client