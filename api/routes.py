from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import io
from PIL import Image
from datetime import datetime
import os
from typing import Dict, Any, List, Optional

from .schemas import *

router = APIRouter()

# Processor will be set from main.py
processor = None

def image_to_array(image_file: UploadFile):
    """Convert uploaded image to numpy array"""
    contents = image_file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return img

def base64_to_array(base64_string: str):
    """Convert base64 string to numpy array"""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


@router.get("/", response_model=Dict[str, Any])
async def root(processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Root endpoint with API information"""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "description": "Real-time face recognition with folder-based user storage",
        "endpoints": {
            "documentation": "/docs",
            "users": "/api/users",
            "recognition": "/api/recognize",
            "enrollment": "/api/enroll",
            "stats": "/api/stats"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Face Recognition API is running",
        version="1.0.0",
        total_users=len(processor.user_manager.list_users())
    )

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Get all registered users"""
    users = []
    
    for username in processor.user_manager.list_users():
        user_data = processor.user_manager.get_user_info(username)
        users.append(UserResponse(
            username=username,
            description=user_data.get('description'),
            created_at=user_data.get('created_at', ''),
            updated_at=user_data.get('updated_at', ''),
            samples_count=user_data.get('samples_count', 0),
            folder_path=processor.user_manager.get_user_folder(username)
        ))
    
    return users

@router.get("/users/{username}", response_model=UserResponse)
async def get_user(username: str, processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Get specific user details"""
    if username not in processor.user_manager.list_users():
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    
    user_data = processor.user_manager.get_user_info(username)
    return UserResponse(
        username=username,
        description=user_data.get('description'),
        created_at=user_data.get('created_at', ''),
        updated_at=user_data.get('updated_at', ''),
        samples_count=user_data.get('samples_count', 0),
        folder_path=processor.user_manager.get_user_folder(username)
    )

@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user_data: UserCreate, processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Create a new user (without face samples)"""
    if user_data.username in processor.user_manager.list_users():
        raise HTTPException(status_code=400, detail=f"User '{user_data.username}' already exists")
    
    # Create user folder
    user_folder = processor.user_manager.create_user(user_data.username)
    
    # Update user info with description
    user_info = processor.user_manager.get_user_info(user_data.username)
    if user_data.description:
        user_info['description'] = user_data.description
        processor.user_manager.save_user_data(user_data.username, user_info)
    
    return UserResponse(
        username=user_data.username,
        description=user_data.description,
        created_at=user_info.get('created_at', ''),
        updated_at=user_info.get('updated_at', ''),
        samples_count=user_info.get('samples_count', 0),
        folder_path=user_folder
    )

@router.delete("/users/{username}")
async def delete_user(username: str, processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Delete a user and all their data"""
    if username not in processor.user_manager.list_users():
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    
    success = processor.user_manager.delete_user(username)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete user '{username}'")
    
    return {"message": f"User '{username}' deleted successfully"}

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(
    image: UploadFile = File(...),
    threshold: float = Query(0.65, ge=0.0, le=1.0, description="Recognition threshold"),
    auto_save: bool = Query(True, description="Auto-save samples for recognized users"),
    processor: FaceRecognitionProcessor = Depends(get_processor)
):
    """
    Recognize faces in an uploaded image
    
    - **image**: JPEG or PNG image with faces
    - **threshold**: Confidence threshold (0.0 to 1.0)
    - **auto_save**: Automatically save new samples for recognized users
    """
    # Read and process image
    img = image_to_array(image)
    
    # Process image
    import time
    start_time = time.time()
    results = processor.process_image(img, auto_save=auto_save)
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Filter by threshold
    filtered_results = []
    for result in results:
        if result['confidence'] >= threshold:
            filtered_results.append(result)
    
    # Count recognized faces
    recognized_count = sum(1 for r in filtered_results if r['username'] is not None)
    
    # Convert to response model
    recognition_results = []
    for result in filtered_results:
        recognition_results.append(RecognitionResult(
            bbox=list(result['bbox']),
            username=result['username'],
            confidence=float(result['confidence']),
            recognized=result['username'] is not None
        ))
    
    return RecognitionResponse(
        results=recognition_results,
        processing_time_ms=processing_time_ms,
        total_faces=len(filtered_results),
        recognized_faces=recognized_count
    )

@router.post("/recognize/base64", response_model=RecognitionResponse)
async def recognize_faces_base64(
    image_base64: str = Form(..., description="Base64 encoded image"),
    threshold: float = Form(0.65, ge=0.0, le=1.0),
    auto_save: bool = Form(True),
    processor: FaceRecognitionProcessor = Depends(get_processor)
):
    """
    Recognize faces from base64 encoded image
    """
    # Convert base64 to image
    img = base64_to_array(image_base64)
    
    # Process image
    import time
    start_time = time.time()
    results = processor.process_image(img, auto_save=auto_save)
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Filter by threshold
    filtered_results = []
    for result in results:
        if result['confidence'] >= threshold:
            filtered_results.append(result)
    
    # Count recognized faces
    recognized_count = sum(1 for r in filtered_results if r['username'] is not None)
    
    # Convert to response model
    recognition_results = []
    for result in filtered_results:
        recognition_results.append(RecognitionResult(
            bbox=list(result['bbox']),
            username=result['username'],
            confidence=float(result['confidence']),
            recognized=result['username'] is not None
        ))
    
    return RecognitionResponse(
        results=recognition_results,
        processing_time_ms=processing_time_ms,
        total_faces=len(filtered_results),
        recognized_faces=recognized_count
    )

@router.post("/enroll", response_model=EnrollmentResponse)
async def enroll_user(
    username: str = Form(..., description="Username for new user"),
    images: List[UploadFile] = File(..., description="Face images (1-10 images)"),
    description: Optional[str] = Form(None, description="Optional user description"),
    processor: FaceRecognitionProcessor = Depends(get_processor)
):
    """
    Enroll a new user with face images
    
    - **username**: Unique username
    - **images**: 1-10 face images (different angles recommended)
    - **description**: Optional user description
    """
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
    
    if username in processor.user_manager.list_users():
        raise HTTPException(status_code=400, detail=f"User '{username}' already exists")
    
    # Create user first
    processor.user_manager.create_user(username)
    
    # Update description if provided
    if description:
        user_info = processor.user_manager.get_user_info(username)
        user_info['description'] = description
        processor.user_manager.save_user_data(username, user_info)
    
    # Process each image
    face_images = []
    for image_file in images:
        img = image_to_array(image_file)
        
        # Detect faces
        faces = processor.detector.detect_faces(img)
        if faces:
            # Use the first detected face
            face_img = processor.detector.extract_face(img, faces[0])
            if face_img is not None:
                face_images.append(face_img)
    
    if not face_images:
        # Clean up created user folder
        processor.user_manager.delete_user(username)
        raise HTTPException(status_code=400, detail="No faces detected in uploaded images")
    
    # Enroll user with detected faces
    success, message = processor.enroll_user(username, face_images)
    
    if not success:
        processor.user_manager.delete_user(username)
        raise HTTPException(status_code=400, detail=message)
    
    user_info = processor.user_manager.get_user_info(username)
    
    return EnrollmentResponse(
        success=True,
        message=message,
        username=username,
        samples_added=user_info.get('samples_count', 0)
    )

@router.post("/users/{username}/add-samples", response_model=EnrollmentResponse)
async def add_user_samples(
    username: str,
    images: List[UploadFile] = File(..., description="Additional face images"),
    processor: FaceRecognitionProcessor = Depends(get_processor)
):
    """
    Add more face samples to existing user
    """
    if username not in processor.user_manager.list_users():
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
    
    # Process each image
    face_images = []
    for image_file in images:
        img = image_to_array(image_file)
        
        # Detect faces
        faces = processor.detector.detect_faces(img)
        if faces:
            face_img = processor.detector.extract_face(img, faces[0])
            if face_img is not None:
                face_images.append(face_img)
    
    if not face_images:
        raise HTTPException(status_code=400, detail="No faces detected in uploaded images")
    
    # Add samples
    success, message = processor.add_samples(username, face_images)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    user_info = processor.user_manager.get_user_info(username)
    
    return EnrollmentResponse(
        success=True,
        message=message,
        username=username,
        samples_added=user_info.get('samples_count', 0)
    )

@router.get("/users/{username}/samples")
async def get_user_samples(
    username: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum samples to return"),
    processor: FaceRecognitionProcessor = Depends(get_processor)
):
    """Get sample images for a user"""
    if username not in processor.user_manager.list_users():
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    
    sample_files = processor.user_manager.get_user_samples(username, max_samples=limit)
    
    samples_data = []
    for sample_file in sample_files:
        try:
            with open(sample_file, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                samples_data.append({
                    "filename": os.path.basename(sample_file),
                    "size_bytes": len(image_data),
                    "created": datetime.fromtimestamp(os.path.getctime(sample_file)).isoformat(),
                    "data": f"data:image/jpeg;base64,{base64_data}"
                })
        except Exception as e:
            continue
    
    return {
        "username": username,
        "total_samples": len(sample_files),
        "samples": samples_data
    }

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Get system statistics"""
    stats = processor.get_stats()
    
    return SystemStats(
        total_users=len(processor.user_manager.list_users()),
        total_frames_processed=stats.get('total_frames', 0),
        average_fps=stats.get('avg_fps', 0),
        average_processing_time_ms=stats.get('avg_processing_time_ms', 0)
    )

@router.get("/test")
async def test_endpoint(processor: FaceRecognitionProcessor = Depends(get_processor)):
    """Test endpoint to verify API is working"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "users_count": len(processor.user_manager.list_users()),
        "system": "Face Recognition API",
        "version": "1.0.0"
    }