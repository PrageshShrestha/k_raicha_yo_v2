from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2
import json
import asyncio
import logging
from typing import Any, Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
import torch
import gc

from utils.local_vlm import describe_image_local, describe_image_nepal_flora_fauna

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
segmentation_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[str] = None

class DetectionResult(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox: List[float]
    polygon: List[List[float]]
    area: float

class CaptionRequest(BaseModel):
    image_data: str
    object_id: Optional[int] = None
    use_nepal_mode: bool = False

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global segmentation_model
    try:
        logger.info(f"Loading YOLO11x-seg model on {device}...")
        segmentation_model = YOLO("yolo11x-seg.pt")
        segmentation_model.to(device)
        logger.info("Model loaded successfully!")
        
        # Warm up the model
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            segmentation_model(dummy_img, verbose=False)
        logger.info("Model warmed up!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if segmentation_model:
        del segmentation_model
    torch.cuda.empty_cache()
    gc.collect()

app = FastAPI(
    title="K_raicha_yo - Advanced Object Identification & Captioning",
    description="Production-ready app for real-time object segmentation and AI-powered captioning",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files

@app.middleware("http")
async def add_response_wrapper(request: Request, call_next):
    response = await call_next(request)
    if response.status_code < 400:
        return response
    
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await response.aread()
            error_data = json.loads(body)
            return JSONResponse(
                status_code=response.status_code,
                content=ApiResponse(
                    success=False,
                    message=error_data.get("detail", "Operation failed"),
                    error_code=f"ERR_{response.status_code}",
                    timestamp=datetime.utcnow().isoformat()
                ).dict(exclude_none=True)
            )
        except:
            pass
    return response

@app.get("/health", response_model=ApiResponse)
async def health_check():
    """Health check endpoint with system info"""
    return ApiResponse(
        success=True,
        data={
            "status": "healthy",
            "device": device,
            "model_loaded": segmentation_model is not None,
            "timestamp": datetime.utcnow().isoformat()
        },
        message="Service is running"
    )

@app.post("/segment", response_model=ApiResponse)
async def segment_image(file: UploadFile = File(...)):
    """Advanced image segmentation with YOLO11x-seg"""
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content=ApiResponse(
                success=False, 
                message="Only images allowed", 
                error_code="INVALID_FILE",
                timestamp=datetime.utcnow().isoformat()
            ).dict()
        )

    contents = await file.read()
    if len(contents) > 16 * 1024 * 1024:  # 16MB limit
        raise HTTPException(413, "Image too large (max 16MB)")

    try:
        # Load and preprocess image
        img_pil = Image.open(io.BytesIO(contents))
        original_size = img_pil.size
        
        # Convert to numpy array
        img_array = np.array(img_pil)
        if img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Run segmentation
        with torch.no_grad():
            results = segmentation_model(img_pil, save=False, verbose=False, conf=0.25)[0]

        # Process results
        detections = []
        annotated_img = img_array.copy()
        
        if results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                class_id = int(results.boxes.cls[i])
                class_name = results.names[class_id]
                conf = float(results.boxes.conf[i])
                
                # Get bounding box
                bbox = results.boxes.xyxy[i].cpu().numpy().tolist()
                
                # Get polygon points
                mask_xy = results.masks.xy[i]
                points = mask_xy.tolist()
                
                # Simplify polygon if too many points
                if len(points) > 64:
                    step = max(1, len(points) // 64)
                    points = points[::step]
                
                # Calculate area
                mask_binary = mask.cpu().numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area = cv2.contourArea(contours[0]) if contours else 0
                
                # Draw on annotated image
                contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_img, [contour], True, (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_x, label_y = int(bbox[0]), int(bbox[1]) - 10
                cv2.rectangle(annotated_img, (label_x, label_y - label_size[1] - 4), 
                            (label_x + label_size[0], label_y), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (label_x, label_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                detections.append(DetectionResult(
                    id=i,
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    polygon=points,
                    area=area
                ))

        # Convert annotated image to base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), 
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        annotated_base64 = base64.b64encode(buffer).decode()

        return ApiResponse(
            success=True,
            data={
                "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
                "detections": [det.dict() for det in detections],
                "objects_detected": len(detections),
                "original_size": original_size
            },
            message=f"Found {len(detections)} object(s)",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Segmentation failed: {str(e)}",
            error_code="SEGMENTATION_ERROR",
            timestamp=datetime.utcnow().isoformat()
        )

@app.post("/caption", response_model=ApiResponse)
async def generate_caption(request: CaptionRequest):
    """Generate caption for segmented object using local VLM"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_data.split(',')[1])
        
        # Generate description
        if request.use_nepal_mode:
            description = describe_image_nepal_flora_fauna(image_data)
        else:
            description = describe_image_local(image_data)

        # Parse description
        parts = description.split("\n\n", 1)
        engaging = parts[0].strip()
        factual = parts[1].strip() if len(parts) > 1 else ""

        return ApiResponse(
            success=True,
            data={
                "full_description": description,
                "engaging_description": engaging,
                "factual_description": factual or description,
                "object_id": request.object_id
            },
            message="Caption generated successfully",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Caption generation error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Caption generation failed: {str(e)}",
            error_code="CAPTION_ERROR",
            timestamp=datetime.utcnow().isoformat()
        )

@app.post("/extract-object", response_model=ApiResponse)
async def extract_object(file: UploadFile = File(...), object_id: int = 0):
    """Extract a specific object from the image based on segmentation mask"""
    contents = await file.read()
    
    try:
        img_pil = Image.open(io.BytesIO(contents))
        img_array = np.array(img_pil)
        
        with torch.no_grad():
            results = segmentation_model(img_pil, save=False, verbose=False)[0]

        if results.masks is None or object_id >= len(results.masks.data):
            raise ValueError("Object not found")

        # Extract the specific mask
        mask = results.masks.data[object_id].cpu().numpy()
        mask_binary = (mask * 255).astype(np.uint8)
        
        # Apply mask to original image
        masked_img = img_array.copy()
        mask_3d = np.stack([mask_binary] * 3, axis=-1) > 0
        masked_img[~mask_3d] = [0, 0, 0]  # Set background to black
        
        # Convert to base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
        object_base64 = base64.b64encode(buffer).decode()

        return ApiResponse(
            success=True,
            data={
                "object_image": f"data:image/png;base64,{object_base64}",
                "class_name": results.names[int(results.boxes.cls[object_id])],
                "confidence": float(results.boxes.conf[object_id])
            },
            message="Object extracted successfully",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Object extraction error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Object extraction failed: {str(e)}",
            error_code="EXTRACTION_ERROR",
            timestamp=datetime.utcnow().isoformat()
        )

@app.websocket("/ws/live-segment")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time segmentation"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process real-time frame here
            await manager.send_personal_message("Frame processed", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=6666, 
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for model consistency
        log_level="info"
    )