from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str = "healthy"
    message: str = "Face Recognition API is running"
    version: str = "1.0.0"
    total_users: int = 0

class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = None

class UserResponse(BaseModel):
    username: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    samples_count: int
    folder_path: str

class RecognitionResult(BaseModel):
    bbox: List[int]
    username: Optional[str] = None
    confidence: float
    recognized: bool

class RecognitionResponse(BaseModel):
    results: List[RecognitionResult]
    processing_time_ms: float
    total_faces: int
    recognized_faces: int

class EnrollmentResponse(BaseModel):
    success: bool
    message: str
    username: str
    samples_added: int

class SystemStats(BaseModel):
    total_users: int
    total_frames_processed: int
    average_fps: float
    average_processing_time_ms: float