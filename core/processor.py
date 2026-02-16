import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple, Optional
import uuid

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.user_manager import UserManager

class FaceRecognitionProcessor: 
    """Main processor for face recognition"""
    
    def __init__(self, data_dir: str = "face_data"):
        """
        Initialize the face recognition processor
        
        Args:
            data_dir: Directory to store user data
        """
        self.data_dir = data_dir
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.user_manager = UserManager(data_dir)
     
        self.total_frames = 0
        self.processing_times = []
        self.start_time = time.time()
   
        self.load_existing_users()
    
    def load_existing_users(self):
        """Load embeddings for existing users"""
        users = self.user_manager.list_users()
        for username in users:
            self.recognizer.load_user_embeddings(
                username, 
                self.user_manager.get_user_folder(username)
            )
    
    def process_image(self, image: np.ndarray, auto_save: bool = True) -> List[Dict[str, Any]]:
        """
        Process an image for face detection and recognition
        
        Args:
            image: Input image as numpy array
            auto_save: Whether to auto-save samples for recognized users
            
        Returns:
            List of recognition results
        """
        start_time = time.time()

        faces = self.detector.detect_faces(image)
        results = []
        
        for face_info in faces:
        
            face_img = self.detector.extract_face(image, face_info)
            
            if face_img is not None:
          
                username, confidence = self.recognizer.recognize_face(face_img)
           
                if auto_save and username is not None and confidence > 0.7:
                    self.add_samples(username, [face_img])
                
                result = {
                    'bbox': face_info['bbox'],
                    'username': username,
                    'confidence': confidence,
                    'face_img': face_img
                }
                results.append(result)

        self.total_frames += 1
        processing_time = (time.time() - start_time) * 1000  
        self.processing_times.append(processing_time)
   
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return results
    
    def enroll_user(self, username: str, face_images: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Enroll a new user with face images
        
        Args:
            username: Username for the new user
            face_images: List of face images
            
        Returns:
            Tuple of (success, message)
        """
        if len(face_images) == 0:
            return False, "No face images provided"
  
        self.user_manager.create_user(username)

        for i, face_img in enumerate(face_images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_{timestamp}_{i+1:03d}.jpg"
            user_folder = self.user_manager.get_user_folder(username)
            sample_path = os.path.join(user_folder, "samples", filename)

            os.makedirs(os.path.dirname(sample_path), exist_ok=True)

            cv2.imwrite(sample_path, face_img)
 
        success = self.recognizer.train_user_embeddings(
            username,
            self.user_manager.get_user_folder(username)
        )
        
        if success:
            user_info = self.user_manager.get_user_info(username)
            user_info['samples_count'] = len(face_images)
            user_info['updated_at'] = datetime.now().isoformat()
            self.user_manager.save_user_data(username, user_info)
            
            return True, f"User '{username}' enrolled successfully with {len(face_images)} samples"
        else:
            self.user_manager.delete_user(username)
            return False, f"Failed to train embeddings for user '{username}'"
    
    def add_samples(self, username: str, face_images: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Add more face samples to an existing user
        
        Args:
            username: Username
            face_images: List of additional face images
            
        Returns:
            Tuple of (success, message)
        """
        if username not in self.user_manager.list_users():
            return False, f"User '{username}' does not exist"
        
        if len(face_images) == 0:
            return False, "No face images provided"
  
        saved_count = 0
        user_folder = self.user_manager.get_user_folder(username)
        
        for i, face_img in enumerate(face_images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"auto_{timestamp}.jpg"
            sample_path = os.path.join(user_folder, "samples", filename)
    
            success = cv2.imwrite(sample_path, face_img)
            if success:
                saved_count += 1
        
        if saved_count > 0:
 
            success = self.recognizer.train_user_embeddings(
                username,
                user_folder
            )
            
            if success:
  
                user_info = self.user_manager.get_user_info(username)
                user_info['samples_count'] = user_info.get('samples_count', 0) + saved_count
                user_info['updated_at'] = datetime.now().isoformat()
                self.user_manager.save_user_data(username, user_info)
                
                return True, f"Added {saved_count} new samples to user '{username}'"
        
        return False, f"Failed to add samples to user '{username}'"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Dictionary with statistics
        """
        if len(self.processing_times) > 0:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        else:
            avg_processing_time = 0
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            avg_fps = self.total_frames / elapsed_time
        else:
            avg_fps = 0
        
        return {
            'total_frames': self.total_frames,
            'avg_processing_time_ms': avg_processing_time,
            'avg_fps': avg_fps,
            'total_users': len(self.user_manager.list_users()),
            'uptime_seconds': elapsed_time
        }
