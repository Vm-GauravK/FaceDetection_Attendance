from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import os

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("="*60)
    print("🚀 Starting Face Recognition API")
    print("="*60)
    
    # Import and initialize processor
    from core.processor import FaceRecognitionProcessor
    app_state['processor'] = FaceRecognitionProcessor()
    
    # Import routes after processor is initialized
    from .routes import router
    app.include_router(router, prefix="/api")
    
    # Update the processor reference in routes module
    from . import routes
    routes.processor = app_state['processor']
    
    users_count = len(app_state['processor'].user_manager.list_users())
    print(f"✅ Loaded {users_count} registered users")
    print(f"📁 Data folder: {os.path.abspath('face_data')}")
    print("="*60)
    print("📚 Swagger UI: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("="*60)
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down Face Recognition API...")
    app_state.clear()

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="""
    ## Real-time Face Recognition System
    
    **Features:**
    - 👤 User enrollment with face images
    - 🔍 Face detection and recognition
    - 📁 Folder-based storage (each user has their own folder)
    - 🚀 CPU-optimized for performance
    - 📊 Real-time statistics
    
    **How it works:**
    1. Enroll users by uploading face images
    2. Each user gets their own folder: `face_data/username/`
    3. Face samples are stored in `face_data/username/samples/`
    4. Upload images for recognition
    5. System identifies recognized users
    
    **Folder Structure:**
    ```
    face_data/
    ├── john_doe/
    │   ├── samples/
    │   │   ├── sample_20240115_143025_123456.jpg
    │   │   └── auto_20240115_150125_456789.jpg
    │   └── embeddings.pkl
    └── jane_smith/
        ├── samples/
        └── embeddings.pkl
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }
            h1 {
                font-size: 48px;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            .tagline {
                font-size: 20px;
                margin-bottom: 40px;
                opacity: 0.9;
            }
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin: 40px 0;
            }
            .card {
                background: rgba(255, 255, 255, 0.15);
                padding: 30px;
                border-radius: 15px;
                transition: transform 0.3s, background 0.3s;
                cursor: pointer;
                text-decoration: none;
                color: white;
                display: block;
            }
            .card:hover {
                transform: translateY(-10px);
                background: rgba(255, 255, 255, 0.25);
            }
            .card h3 {
                margin-top: 0;
                font-size: 24px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .card p {
                opacity: 0.9;
                line-height: 1.6;
            }
            .endpoints {
                background: rgba(0, 0, 0, 0.2);
                padding: 25px;
                border-radius: 15px;
                margin-top: 40px;
            }
            .endpoint {
                padding: 15px;
                margin: 10px 0;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                display: flex;
                align-items: center;
                gap: 15px;
            }
            .method {
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }
            .get { background: #10b981; }
            .post { background: #3b82f6; }
            .delete { background: #ef4444; }
            .endpoint-path {
                font-family: monospace;
                font-size: 16px;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid rgba(255, 255, 255, 0.2);
                opacity: 0.8;
            }
            .btn {
                display: inline-block;
                padding: 15px 30px;
                background: #fff;
                color: #667eea;
                text-decoration: none;
                border-radius: 30px;
                font-weight: bold;
                margin: 10px;
                transition: all 0.3s;
            }
            .btn:hover {
                transform: scale(1.05);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            }
            .btn-primary {
                background: #10b981;
                color: white;
            }
            .btn-secondary {
                background: #3b82f6;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👁️ Face Recognition API</h1>
            <div class="tagline">
                Real-time face detection and recognition with folder-based user storage
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="/docs" class="btn btn-primary">📚 Open Swagger UI</a>
                <a href="/redoc" class="btn btn-secondary">📖 Open ReDoc</a>
            </div>
            
            <div class="cards">
                <a href="/docs#/default/recognize_faces" class="card">
                    <h3>🔍 Face Recognition</h3>
                    <p>Upload images for face detection and recognition. System identifies registered users with confidence scores.</p>
                </a>
                
                <a href="/docs#/default/enroll_user" class="card">
                    <h3>👤 User Enrollment</h3>
                    <p>Register new users by uploading face images. Each user gets their own folder with samples.</p>
                </a>
                
                <a href="/docs#/default/get_all_users" class="card">
                    <h3>📋 User Management</h3>
                    <p>View all registered users, get details, add more samples, or delete users.</p>
                </a>
                
                <a href="/docs#/default/get_system_stats" class="card">
                    <h3>📊 System Statistics</h3>
                    <p>View real-time statistics: users count, processing speed, and system performance.</p>
                </a>
            </div>
            
            <div class="endpoints">
                <h3>📡 API Endpoints</h3>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/recognize</span>
                    <span>Recognize faces in image</span>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="endpoint-path">/api/enroll</span>
                    <span>Enroll new user</span>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/users</span>
                    <span>List all users</span>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/stats</span>
                    <span>System statistics</span>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="endpoint-path">/api/health</span>
                    <span>Health check</span>
                </div>
            </div>
            
            <div class="footer">
                <p>Face Recognition API v1.0.0 | Each user stored in separate folder</p>
                <p>📁 Data folder: <code>face_data/username/samples/</code></p>
            </div>
        </div>
        
        <script>
            // Add some interactivity
            document.querySelectorAll('.card').forEach(card => {
                card.addEventListener('click', function(e) {
                    if (!e.target.closest('a')) {
                        this.style.transform = 'scale(0.98)';
                        setTimeout(() => {
                            this.style.transform = '';
                        }, 150);
                    }
                });
            });
            
            // Update stats if available
            async function updateStats() {
                try {
                    const response = await fetch('/api/stats');
                    const data = await response.json();
                    
                    const statsElement = document.createElement('div');
                    statsElement.className = 'stats';
                    statsElement.innerHTML = `
                        <h4>Live Stats</h4>
                        <p>Users: ${data.total_users} | FPS: ${data.average_fps.toFixed(1)}</p>
                    `;
                    
                    document.querySelector('.footer').prepend(statsElement);
                } catch (error) {
                    console.log('Stats not available yet');
                }
            }
            
            // Update stats every 10 seconds
            setInterval(updateStats, 10000);
            updateStats();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/test/image")
async def test_image():
    """Serve a test image"""
    # Create a simple test image
    import cv2
    import base64
    import numpy as np
    
    # Create a test image with text
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(img, "Face Recognition API", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Test Image", (150, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', img)
    base64_data = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "message": "Test image generated",
        "image": f"data:image/jpeg;base64,{base64_data}"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )