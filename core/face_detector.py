import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    """Face detector using MediaPipe"""
    
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_faces(self, image, resize=True):
        """Detect faces in image"""
        if resize and (image.shape[0] > 720 or image.shape[1] > 1280):
            image = cv2.resize(image, (640, 480))
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        results = self.detector.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]
                
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                faces.append((x1, y1, x2, y2))
        
        return faces
    
    def extract_face(self, image, bbox):
        """Extract face region"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size > 0:
            return cv2.resize(face_region, (112, 112))
        return None