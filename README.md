# Face Recognition API

A RESTful API for face recognition with folder-based user storage. Each user's face samples are stored in their own folder.

## Features

- ✅ **Folder-based storage**: Each user has `face_data/username/samples/`
- ✅ **Face detection & recognition**: Using InsightFace (CPU optimized)
- ✅ **User management**: Create, read, update, delete users
- ✅ **Interactive documentation**: Swagger UI & ReDoc
- ✅ **Image upload**: JPEG/PNG images
- ✅ **Base64 support**: Send images as base64 strings
- ✅ **Auto-learning**: Save new samples during recognition
- ✅ **Real-time statistics**: FPS, processing time, user count

## Quick Start

### 1. Installation
```bash
# Clone or create project
mkdir face_recognition_api
cd face_recognition_api

# Install dependencies
pip install -r requirements.txt