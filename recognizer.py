import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
import threading
import time
import requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class FaceDetector:
    def __init__(self, prototxt_path="models/deploy.prototxt", 
                 model_path="models/res10_300x300_ssd_iter_140000.caffemodel"):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = 0.5
    
    def detect(self, frame):
        """Detect faces in frame using DNN"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes fall within the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                faces.append({
                    'box': (startX, startY, endX, endY),
                    'confidence': confidence
                })
        
        return faces

class Recognizer:
    def __init__(self, model_name: str = 'Facenet', detector_backend: str = 'opencv', threshold: float = 0.40):
        import os
        from deepface import DeepFace
        
        SUPABASE_URL = os.getenv('SUPABASE_URL')
        SUPABASE_KEY = os.getenv('SUPABASE_KEY')
        SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'images')
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError('SUPABASE_URL and SUPABASE_KEY must be set')
        
        self.sup = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.threshold = threshold
        self.DeepFace = DeepFace
        self.embeddings = []
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Cache for recognized faces
        self.recognized_faces = {}
        self.recognition_lock = threading.Lock()
        
        # Device ID for filtering images
        self.device_id = os.getenv('DEVICE_ID')
        if not self.device_id:
            print("[WARN] DEVICE_ID not set. Will load all images from bucket.")
    
    def build_gallery_from_device(self):
        """Build face recognition gallery from Supabase images bucket for this device"""
        try:
            print(f"[INFO] Building gallery for device: {self.device_id}")
            
            # List files in the device's folder in images bucket
            if self.device_id:
                items = self.sup.storage.from_('images').list(path=self.device_id)
                print(f"[DEBUG] Supabase items: {items}")
            else:
                # Fallback: list all images (not recommended for production)
                items = self.sup.storage.from_('images').list()
                print("[WARN] Loading all images from bucket (device ID not specified)")
            
            self.embeddings = []
            image_count = 0
            
            for obj in items:
                path = obj.get('name')
                path = self.device_id + '/' + path
                print(f"[DEBUG] Processing image: {path}")
                
                # Skip if not in device folder (when device_id is not set)
                if self.device_id and not path.startswith(f"{self.device_id}/"):
                    continue
                
                try:
                    # Get signed URL for the image
                    print("[DEBUG] Getting signed URL")
                    signed = self.sup.storage.from_('images').create_signed_url(path, 3600)
                    url = signed.get('signed_url') or signed.get('signedURL')
                    
                    if not url:
                        print(f"[WARN] No URL for {path}")
                        continue
                    
                    # Download image
                    resp = requests.get(url)
                    resp.raise_for_status()
                    img_bytes = resp.content
                    
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                    
                    if img is None:
                        print(f"[WARN] Failed to decode image: {path}")
                        continue
                    
                    # Extract embedding
                    try:
                        emb = self.DeepFace.represent(
                            img, 
                            model_name=self.model_name,
                            detector_backend="skip",
                            enforce_detection=False,
                            align=False
                        )
                        
                        # Extract person name from filename
                        # Format: deviceID/watchlistId_watchlistName.ext
                        filename = path.split('/')[-1] if '/' in path else path
                        
                        # Remove extension and split
                        name_without_ext = filename.rsplit('.', 1)[0]
                        parts = name_without_ext.split('_')
                        
                        if len(parts) >= 2:
                            # watchlistId_watchlistName format
                            person_name = parts[1] # '_'.join(parts[1:])  # Join all parts after ID
                        else:
                            person_name = "Unknown"
                        
                        self.embeddings.append({
                            'embedding': emb,
                            'person_name': person_name,
                            'path': path,
                            'url': url
                        })
                        
                        image_count += 1
                        print(f"[INFO] Loaded embedding for: {person_name} from {filename}")
                        
                    except Exception as e:
                        print(f"[WARN] Failed to extract embedding from {path}: {e}")
                        continue
                        
                except Exception as e:
                    print(f"[WARN] Failed to process {path}: {e}")
                    continue
            
            print(f"[INFO] Gallery built with {image_count} embeddings")
            
            if image_count == 0:
                print("[WARN] No images found in gallery. Check Supabase bucket and device ID.")
                
        except Exception as e:
            print(f"[ERROR] Failed to build gallery: {e}")
            import traceback
            traceback.print_exc()
    
    def recognize(self, frame):
        """Recognize faces in frame"""
        # First detect faces
        faces = self.face_detector.detect(frame)
        
        if not faces:
            return False, {"reason": "no_faces_detected"}
        
        # For each detected face, try recognition
        recognized = False
        best_match = None
        best_confidence = 0
        
        for face in faces:
            startX, startY, endX, endY = face['box']
            
            # Extract face region
            face_region = frame[startY:endY, startX:endX]
            face_region = cv2.resize(face_region, (160, 160))
            
            if face_region.size == 0:
                continue
            
            try:
                # Get embedding for face region
                probe_emb = self.DeepFace.represent(
                    face_region,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=True
                )
                
                # Compare with gallery
                for entry in self.embeddings:
                    gallery_emb = entry['embedding']
                    
                    # Calculate cosine similarity
                    a = np.array(probe_emb)
                    b = np.array(gallery_emb)
                    
                    if a.size == 0 or b.size == 0:
                        continue
                    
                    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                    
                    if cos_sim > self.threshold and cos_sim > best_confidence:
                        best_confidence = cos_sim
                        best_match = {
                            'name': entry['person_name'],
                            'confidence': cos_sim,
                            'box': face['box'],
                            'distance': 1 - cos_sim,
                            'source_image': entry['path']
                        }
                        recognized = True
                        
            except Exception as e:
                print(f"[WARN] Face recognition error: {e}")
                continue
        
        if recognized and best_match:
            # Update recognized faces cache
            with self.recognition_lock:
                self.recognized_faces[tuple(best_match['box'])] = best_match['name']
            
            return True, {
                'name': best_match['name'],
                'confidence': best_match['confidence'],
                'box': best_match['box'],
                'distance': best_match['distance'],
                'source_image': best_match.get('source_image', '')
            }
        
        return False, {"reason": "no_match", "faces_detected": len(faces)}
    
    def get_recognized_faces(self):
        """Get currently recognized faces"""
        with self.recognition_lock:
            return self.recognized_faces.copy()
    
    def refresh_gallery(self):
        """Refresh the gallery from Supabase"""
        print("[INFO] Refreshing face recognition gallery...")
        self.build_gallery_from_device()
        print("[INFO] Gallery refreshed")


if __name__ == "__main__":
    recognize = Recognizer(model_name="SFace")
    recognize.build_gallery_from_device()