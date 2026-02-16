import os
import pickle
import numpy as np
import cv2
from datetime import datetime
import shutil

class UserManager:
    """Manages user data in folder-based system"""
    
    def __init__(self, data_folder="face_data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)
        self.users = {}
        self.load_all_users()
    
    def load_all_users(self):
        """Load all users from folders"""
        self.users = {}
        
        if not os.path.exists(self.data_folder):
            return
        
        for username in os.listdir(self.data_folder):
            user_folder = os.path.join(self.data_folder, username)
            if os.path.isdir(user_folder):
                user_data = self.load_user(username)
                if user_data:
                    self.users[username] = user_data
    
    def load_user(self, username):
        """Load single user data"""
        user_folder = self.get_user_folder(username)
        embeddings_file = os.path.join(user_folder, "embeddings.pkl")
        
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def create_user(self, username):
        """Create user folder structure"""
        user_folder = self.get_user_folder(username)
        samples_folder = os.path.join(user_folder, "samples")
        os.makedirs(samples_folder, exist_ok=True)
        
        user_data = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'embeddings': [],
            'samples_count': 0,
            'samples_folder': samples_folder
        }
        
        self.save_user_data(username, user_data)
        return user_folder
    
    def save_user_data(self, username, user_data):
        """Save user data to disk"""
        user_folder = self.get_user_folder(username)
        embeddings_file = os.path.join(user_folder, "embeddings.pkl")
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(user_data, f)
        
        self.users[username] = user_data
        return True
    
    def add_face_sample(self, username, face_image, embedding):
        """Add face sample for user"""
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        samples_folder = user_data['samples_folder']
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_path = os.path.join(samples_folder, f"sample_{timestamp}.jpg")
        cv2.imwrite(sample_path, face_image)
        
        # Update embeddings
        user_data['embeddings'].append(embedding)
        user_data['samples_count'] += 1
        user_data['updated_at'] = datetime.now().isoformat()
        
        # Calculate average embedding
        if user_data['embeddings']:
            embeddings_array = np.array(user_data['embeddings'])
            user_data['avg_embedding'] = np.mean(embeddings_array, axis=0)
        
        return self.save_user_data(username, user_data)
    
    def recognize_user(self, query_embedding, threshold=0.65):
        """Recognize user from embedding"""
        best_match = None
        best_score = 0.0
        
        for username, user_data in self.users.items():
            if 'avg_embedding' not in user_data:
                continue
            
            avg_embedding = user_data['avg_embedding']
            similarity = np.dot(query_embedding, avg_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(avg_embedding)
            )
            
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = username
        
        return best_match, best_score
    
    def get_user_folder(self, username):
        """Get user folder path"""
        return os.path.join(self.data_folder, username)
    
    def list_users(self):
        """List all users"""
        return list(self.users.keys())
    
    def get_user_info(self, username):
        """Get user information"""
        if username in self.users:
            return self.users[username]
        return None
    
    def delete_user(self, username):
        """Delete user"""
        if username in self.users:
            user_folder = self.get_user_folder(username)
            if os.path.exists(user_folder):
                shutil.rmtree(user_folder)
            del self.users[username]
            return True
        return False
    
    def get_user_samples(self, username, max_samples=10):
        """Get user's sample images"""
        if username not in self.users:
            return []
        
        samples_folder = self.users[username]['samples_folder']
        if not os.path.exists(samples_folder):
            return []
        
        sample_files = []
        for f in os.listdir(samples_folder):
            if f.endswith(('.jpg', '.png')):
                sample_files.append(os.path.join(samples_folder, f))
        
        sample_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return sample_files[:max_samples]