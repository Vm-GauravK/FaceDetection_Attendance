#!/usr/bin/env python3
"""
Fix script for Face Recognition API
Run this to fix common import issues
"""
import os
import sys

print("🔧 Fixing Face Recognition API imports...")

# Create missing __init__.py files
init_files = ['api/__init__.py', 'core/__init__.py']
for file in init_files:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write('# Package initialization\n')
        print(f"✅ Created {file}")

# Test imports
try:
    from api.routes import router
    print("✅ api.routes import successful")
except Exception as e:
    print(f"❌ api.routes import failed: {e}")

try:
    from core.processor import FaceRecognitionProcessor
    print("✅ core.processor import successful")
except Exception as e:
    print(f"❌ core.processor import failed: {e}")

try:
    from api.schemas import HealthResponse
    print("✅ api.schemas import successful")
except Exception as e:
    print(f"❌ api.schemas import failed: {e}")

print("\n✅ Fix script completed!")
print("\n📋 To run the API:")
print("   python main.py --reload")
print("   or")
print("   uvicorn api.main:app --reload")