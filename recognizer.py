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
    def __init__(self, model_name: str = 'Facenet', detector_backend: str = 'opencv', threshold: float = 0.40, base_url = None):
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
        self.base_url = base_url
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Cache for recognized faces
        self.recognized_faces = {}
        self.recognition_lock = threading.Lock()
        
        # Device ID for filtering images
        self.device_id = os.getenv('DEVICE_ID')
        if not self.device_id:
            print("[WARN] DEVICE_ID not set. Will load all images from bucket.")

    def l2_normalize(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)
    
    def load_embeddings_from_backend(self):
        """Load embeddings from deployed backend (no images)"""
        try:
            if not self.device_id:
                print("[ERROR] DEVICE_ID not set")
                return

            url = f"{self.base_url}/api/watchlist/device/{self.device_id}/embeddings"

            resp = requests.get(url)
            resp.raise_for_status()

            data = resp.json()
            self.embeddings = []

            for item in data.get("embeddings", []):
                self.embeddings.append({
                    "person_name": item["name"],
                    "embedding": self.l2_normalize(item["embedding"])
                })


            print(f"[INFO] Loaded {len(self.embeddings)} embeddings from backend")

        except Exception as e:
            print("[ERROR] Failed to load embeddings:", e)
    
    
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
            if face_region is None or face_region.size == 0:
                continue

            h, w = face_region.shape[:2]
            if h < 30 or w < 30:
                continue

            face_region = cv2.resize(face_region, (160, 160), interpolation=cv2.INTER_AREA)
            
            try:
                # Get embedding for face region
                rep = self.DeepFace.represent(
                    face_region,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False
                )

                if not rep:
                    continue

                probe_emb = np.array(rep[0]["embedding"], dtype=np.float32)
                
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

    # recognizer.py
    def recognize_face(self, face_region):
        """ Recognize a single cropped face image (160x160)"""
        try:
            face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            rep = self.DeepFace.represent(
                face_region,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                align=True
            )

            if not rep:
                return False, None

            probe_emb_raw = np.array(rep[0]["embedding"], dtype=np.float32)
            probe_emb = self.l2_normalize(probe_emb_raw)

            best_match = None
            best_conf = 0.0

            for entry in self.embeddings:
                gallery_emb = entry["embedding"]

                # Verify gallery embedding is normalized (optional debug)
                gallery_norm = np.linalg.norm(gallery_emb)
                if abs(gallery_norm - 1.0) > 0.01:
                    print(f"[WARN] Gallery embedding not normalized: {gallery_norm}")

                # cos_sim = np.dot(probe_emb, gallery_emb) / (
                #     np.linalg.norm(probe_emb) * np.linalg.norm(gallery_emb) + 1e-10
                # )

                cos_sim = np.dot(probe_emb, gallery_emb)

                print(f"[DEBUG] Similarity: {cos_sim:.4f}")

                if cos_sim > self.threshold and cos_sim > best_conf:
                    best_conf = cos_sim
                    best_match = {
                        "name": entry["person_name"],
                        "confidence": cos_sim,
                        #"source_image": entry["path"]
                    }

            if best_match:
                return True, best_match

            return False, None

        except Exception as e:
            print("[WARN] Face recognition error:", e)
            return False, None

    
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