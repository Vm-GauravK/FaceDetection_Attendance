import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceRecognizer:
    """Face recognition using InsightFace"""
    
    def __init__(self, model_name='buffalo_l'):
        providers = ['CPUExecutionProvider']
        self.model = FaceAnalysis(name=model_name, providers=providers)
        self.model.prepare(ctx_id=-1, det_size=(224, 224))
    
    def get_embedding(self, face_image):
        """Extract face embedding"""
        try:
            faces = self.model.get(face_image)
            if len(faces) > 0:
                return faces[0].embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
        return None