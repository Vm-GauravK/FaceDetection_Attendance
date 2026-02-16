"""
Face Detection System - ONNX Export
Simple face detection with ONNX model export
Removes recognition logic and advanced features
"""

import os
import sys
import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

print("\n" + "="*60)
print("FACE DETECTION SYSTEM - ONNX VERSION")
print("="*60)

# ============================================================================
# 1. CHECK OPENCV
# ============================================================================

print("\nStep 1: Checking OpenCV...")
try:
    print(f"OpenCV version: {cv2.__version__}")
except:
    print("OpenCV not installed!")
    print("Install with: pip install opencv-python")
    sys.exit(1)

# ============================================================================
# 2. CREATE FACE_DATA FOLDER
# ============================================================================

face_data_dir = "face_data"
os.makedirs(face_data_dir, exist_ok=True)
print(f"Created data folder: {face_data_dir}")

# ============================================================================
# 3. LOAD FACE DETECTOR
# ============================================================================

print("\nStep 2: Loading face detector...")
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        print("Failed to load face detector!")
        sys.exit(1)
    print("✓ Face detector loaded successfully")
except Exception as e:
    print(f"Error loading face detector: {e}")
    sys.exit(1)

# ============================================================================
# 4. LOAD USERS
# ============================================================================

print("\nStep 3: Loading users...")

users_file = os.path.join(face_data_dir, "users.json")
if os.path.exists(users_file):
    with open(users_file, 'r') as f:
        users = json.load(f)
    print(f"✓ Loaded {len(users)} existing users")
else:
    users = {}
    print("ℹ No existing users found")

# ============================================================================
# 5. TEST CAMERA
# ============================================================================

def test_camera():
    """Test available cameras"""
    print("\nTesting camera...")
    
    for camera_index in range(3):
        print(f"  Testing camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print(f"✓ Camera {camera_index} works! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return camera_index
    
    print("✗ No working camera found!")
    return None

camera_index = test_camera()
if camera_index is None:
    print("Please check your camera connection")
    sys.exit(1)

# ============================================================================
# 6. FACE DETECTION FUNCTIONS
# ============================================================================

def detect_faces(frame):
    """Detect faces in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    return faces

def extract_face(frame, face_rect):
    """Extract face region with padding"""
    x, y, w, h = face_rect
    padding = 40
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    
    face_img = frame[y1:y2, x1:x2]
    if face_img.size > 0:
        return cv2.resize(face_img, (200, 200))
    return None

def save_face_image(username, face_image):
    """Save face image to user's folder"""
    try:
        user_folder = os.path.join(face_data_dir, username)
        samples_folder = os.path.join(user_folder, "samples")
        os.makedirs(samples_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"face_{timestamp}.jpg"
        filepath = os.path.join(samples_folder, filename)
        
        success = cv2.imwrite(filepath, face_image)
        if not success:
            print(f"Failed to save image to {filepath}")
            return None
        
        # Update users dictionary
        if username not in users:
            users[username] = {
                "username": username,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "samples_count": 1,
                "folder": user_folder
            }
        else:
            current_count = users[username].get("samples_count", 0)
            users[username]["samples_count"] = current_count + 1
            users[username]["updated_at"] = datetime.now().isoformat()
        
        # Save users to file
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        print(f"✓ Saved face for '{username}': {filepath}")
        print(f"  {username} now has {users[username]['samples_count']} total samples")
        return filepath
        
    except Exception as e:
        print(f"Error saving face image: {e}")
        return None

# ============================================================================
# 7. EXPORT TO ONNX
# ============================================================================

def export_face_data_to_onnx():
    """Export face detection data to ONNX format"""
    print("\n" + "="*60)
    print("EXPORTING TO ONNX FORMAT")
    print("="*60)
    
    if not users:
        print("✗ No users found to export!")
        return False
    
    try:
        # Collect all face images
        all_faces = []
        all_labels = []
        label_map = {}
        
        for idx, username in enumerate(users):
            user_folder = os.path.join(face_data_dir, username, "samples")
            if os.path.exists(user_folder):
                label_map[username] = idx
                
                for filename in os.listdir(user_folder):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(user_folder, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            gray = cv2.resize(gray, (200, 200))
                            # Normalize
                            gray = gray.astype(np.float32) / 255.0
                            all_faces.append(gray)
                            all_labels.append(idx)
        
        if len(all_faces) == 0:
            print("✗ No face images found!")
            return False
        
        # Convert to numpy arrays
        faces_array = np.array(all_faces)
        labels_array = np.array(all_labels)
        
        # Reshape for ONNX (batch, channels, height, width)
        faces_array = faces_array.reshape(-1, 1, 200, 200)
        
        print(f"✓ Collected {len(all_faces)} face images from {len(label_map)} users")
        print(f"  Data shape: {faces_array.shape}")
        
        # Save as numpy arrays (ONNX-compatible format)
        onnx_dir = os.path.join(face_data_dir, "onnx_export")
        os.makedirs(onnx_dir, exist_ok=True)
        
        faces_file = os.path.join(onnx_dir, "faces_data.npy")
        labels_file = os.path.join(onnx_dir, "labels_data.npy")
        labelmap_file = os.path.join(onnx_dir, "label_mapping.json")
        
        np.save(faces_file, faces_array)
        np.save(labels_file, labels_array)
        
        with open(labelmap_file, 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"✓ Saved ONNX-compatible data:")
        print(f"  • {faces_file}")
        print(f"  • {labels_file}")
        print(f"  • {labelmap_file}")
        
        # Create metadata file
        metadata = {
            "export_date": datetime.now().isoformat(),
            "num_users": len(label_map),
            "num_samples": len(all_faces),
            "image_size": [200, 200],
            "data_format": "NCHW",
            "label_mapping": label_map,
            "users": list(label_map.keys())
        }
        
        metadata_file = os.path.join(onnx_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata: {metadata_file}")
        print("\n" + "="*60)
        print("ONNX EXPORT COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error exporting to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 8. CAPTURE FACES
# ============================================================================

def capture_faces():
    """Capture faces from camera"""
    print("\n" + "="*60)
    print("CAPTURE FACES")
    print("="*60)
    
    name = input("Enter name for the person: ").strip()
    if not name:
        print("✗ Cancelled: No name provided")
        return False
    
    if name in users:
        print(f"ℹ User '{name}' already exists - will add more samples")
    
    print(f"\n✓ Starting capture for {name}...")
    print("  Look at the camera")
    print("  System will capture 5 images in 5 seconds")
    print("  Press 'q' to quit early")
    print("-" * 40)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("✗ Could not open camera")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    captured_images = []
    capture_count = 0
    max_captures = 5
    total_time_limit = 5.0
    start_time = time.time()
    last_capture_time = start_time
    
    print("\nStarting in 2 seconds...")
    time.sleep(2)
    
    while time.time() - start_time < total_time_limit and capture_count < max_captures:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_faces(frame)
        display_frame = frame.copy()
        
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        
        cv2.putText(display_frame, f"Capturing: {name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Captured: {capture_count}/{max_captures}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(faces) > 0:
            # Use first detected face
            x, y, w, h = faces[0]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display_frame, "FACE DETECTED", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_time = time.time()
            if current_time - last_capture_time > 0.8:
                face_img = extract_face(frame, faces[0])
                if face_img is not None:
                    captured_images.append(face_img)
                    capture_count += 1
                    last_capture_time = current_time
                    
                    filepath = save_face_image(name, face_img)
                    if filepath:
                        print(f"✓ Captured image {capture_count}/{max_captures}")
        else:
            cv2.putText(display_frame, "NO FACE DETECTED", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(f'Capture: {name}', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✗ Capture stopped by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_elapsed = time.time() - start_time
    print(f"\n✓ Capture completed in {total_elapsed:.2f} seconds")
    
    if captured_images:
        print(f"✓ Successfully captured {len(captured_images)} images for {name}")
        user_folder = os.path.join(face_data_dir, name, "samples")
        if os.path.exists(user_folder):
            sample_count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
            print(f"  Total samples for {name}: {sample_count}")
        return True
    else:
        print("✗ No face images captured")
        return False

# ============================================================================
# 9. LIST USERS
# ============================================================================

def list_users():
    """List all registered users"""
    print("\n" + "="*60)
    print("REGISTERED USERS")
    print("="*60)
    
    if not users:
        print("ℹ No users registered yet")
        return
    
    total_samples = 0
    
    for username, info in users.items():
        samples = info.get("samples_count", 0)
        created = info.get("created_at", "Unknown")[:19]
        folder = info.get("folder", "N/A")
        
        print(f"\n• {username}:")
        print(f"  Samples: {samples}")
        print(f"  Created: {created}")
        print(f"  Folder: {folder}")
        
        total_samples += samples
    
    print(f"\n✓ TOTAL: {len(users)} users, {total_samples} samples")

# ============================================================================
# 10. MAIN MENU
# ============================================================================

def show_main_menu():
    """Show main menu"""
    print("\n" + "="*60)
    print("FACE DETECTION SYSTEM - ONNX")
    print("="*60)
    print(f"Data folder: {face_data_dir}/")
    print(f"Registered users: {len(users)}")
    
    if users:
        total_samples = sum(info.get("samples_count", 0) for info in users.values())
        print(f"Total samples: {total_samples}")
        print(f"Users: {', '.join(users.keys())}")
    
    print("\n" + "="*60)
    print("MAIN MENU:")
    print("1. Capture Faces")
    print("2. List Users")
    print("3. Export to ONNX")
    print("0. Exit")
    print("-" * 40)
    
    choice = input("Select option (0-3): ").strip()
    return choice

# ============================================================================
# 11. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n✓ Face Detection System Ready")
        print("  Simplified version - detection only")
        print("  ONNX export capability")
        print("="*60)
        
        while True:
            choice = show_main_menu()
            
            if choice == "0":
                print("\n✓ Goodbye!")
                break
            
            elif choice == "1":
                capture_faces()
                input("\nPress Enter to continue...")
            
            elif choice == "2":
                list_users()
                input("\nPress Enter to continue...")
            
            elif choice == "3":
                export_face_data_to_onnx()
                input("\nPress Enter to continue...")
            
            else:
                print("✗ Invalid choice. Please select 0-3.")
                input("\nPress Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n\n✓ Goodbye!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")