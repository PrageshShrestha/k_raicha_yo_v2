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
from datetime import datetime, timezone
import uvicorn
from contextlib import asynccontextmanager
import torch
import gc

# NEW imports for Phi-3.5-vision
from transformers import AutoModelForCausalLM, AutoProcessor

from utils.local_vlm import describe_image_local, describe_image_nepal_flora_fauna, set_vlm_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global model variables
segmentation_model = None
vlm_model = None
vlm_processor = None
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
    object_class: Optional[str] = None
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
    global segmentation_model, vlm_model, vlm_processor

    try:
        logger.info(f"Loading YOLO11x-seg model on {device}...")
        segmentation_model = YOLO("yolo11x-seg.pt")
        segmentation_model.to(device)
        logger.info("YOLO loaded successfully!")
        
        # Warm up YOLO
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            segmentation_model(dummy_img, verbose=False)
        logger.info("YOLO warmed up!")

        # NEW: Load Phi-3.5-vision-instruct ONCE on GPU
        logger.info(f"Loading Phi-3.5-vision-instruct on {device}...")
        model_path = "models/phi35_vision"   # your local folder

        vlm_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=2   # Further reduced to 2 to save memory
        )

        vlm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # Use CPU instead of GPU to save memory
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,   # Use bfloat16 for efficiency
            _attn_implementation="eager"   # fallback to "eager" if flash-attn not installed
        )
        logger.info("Phi-3.5-vision loaded successfully!")
        
        # Set global VLM model in local_vlm.py
        set_vlm_model(vlm_model, vlm_processor)

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")
    if segmentation_model:
        del segmentation_model
    if vlm_model:
        del vlm_model
    if vlm_processor:
        del vlm_processor
    if torch.cuda.is_available():
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
                    timestamp=datetime.now(timezone.utc).isoformat()
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
            "timestamp": datetime.now(timezone.utc).isoformat()
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
                timestamp=datetime.now(timezone.utc).isoformat()
            ).dict()
        )

    contents = await file.read()
    if len(contents) > 200 * 1024 * 1024:  # 16MB limit
        raise HTTPException(413, "Image too large (max 200MB)")

    try:
        # Load and preprocess image
        img_pil = Image.open(io.BytesIO(contents))
        original_size = img_pil.size
        print(original_size)
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
                
                # Create detection object
                detection = {
                    "id": i,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bounding_box": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1]
                    },
                    "polygon": points,
                    "area": float(results.boxes.area[i])
                }
                detections.append(detection)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), 
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        annotated_base64 = base64.b64encode(buffer).decode()
        
        return ApiResponse(
            success=True,
            data={
                "image_data": f"data:image/jpeg;base64,{annotated_base64}",
                "objects": [det.dict() for det in detections],
                "objects_detected": len(detections),
                "original_size": original_size
            },
            message=f"Found {len(detections)} object(s)",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Segmentation failed: {str(e)}",
            error_code="SEGMENTATION_ERROR",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

@app.post("/segment/base64", response_model=ApiResponse)
async def segment_image_base64(request: dict):
    """Advanced image segmentation with YOLO11x-seg from base64"""
    try:
        # Extract base64 image data
        image_data = request.get("image_data")
        if not image_data:
            return ApiResponse(
                success=False,
                message="No image_data provided",
                error_code="MISSING_IMAGE_DATA",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Remove data URL prefix if present
        if image_data.startswith("data:image/"):
            image_data = image_data.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Load and preprocess image
        img_pil = Image.open(io.BytesIO(image_bytes))
        original_size = img_pil.size
        print(f"Received base64 image size: {original_size}")
        
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
                
                # Create detection object
                detection = {
                    "id": i,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bounding_box": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1]
                    },
                    "polygon": points,
                    "area": float(results.boxes.area[i])
                }
                detections.append(detection)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), 
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        annotated_base64 = base64.b64encode(buffer).decode()
        
        return ApiResponse(
            success=True,
            data={
                "image_data": f"data:image/jpeg;base64,{annotated_base64}",
                "objects": detections,
                "objects_detected": len(detections),
                "original_size": original_size
            },
            message=f"Found {len(detections)} object(s)",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Base64 segmentation error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Segmentation failed: {str(e)}",
            error_code="SEGMENTATION_ERROR",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

@app.post("/caption", response_model=ApiResponse)
async def generate_caption(request: CaptionRequest):
    """Generate caption for segmented object using local VLM"""
    try:
        # Decode base64 → bytes → PIL Image
        image_data = base64.b64decode(request.image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate description (function now uses persistent global model)
        if request.use_nepal_mode:
            description = describe_image_nepal_flora_fauna(image)
        else:
            description = describe_image_local(image)

        # Parse description (your existing logic)
        parts = description.split("\n\n", 1)
        engaging = parts[0].strip()
        factual = parts[1].strip() if len(parts) > 1 else ""

        # Get object information from segmentation result if available
        object_info = ""
        if hasattr(request, 'object_class') and request.object_class:
            object_info = f"This appears to be a {request.object_class}. "
            
            # Add object-specific context to the description
            if request.object_class.lower() in ['person', 'people', 'human']:
                object_info += "The individual appears to be "
                if request.use_nepal_mode:
                    object_info += "dressed in traditional Nepali attire" if 'traditional' in description.lower() else "wearing modern clothing"
                else:
                    object_info += "wearing casual attire"
                object_info += ". "
                
            elif request.object_class.lower() in ['car', 'vehicle', 'truck', 'bus']:
                object_info += f"This {request.object_class} "
                if request.use_nepal_mode:
                    object_info += "is commonly seen on Nepalese roads, often used for transportation between cities like Kathmandu and Pokhara"
                else:
                    object_info += "is a motor vehicle used for transportation"
                object_info += ". "
                
            elif request.object_class.lower() in ['dog', 'cat', 'animal', 'pet']:
                object_info += f"This {request.object_class} "
                if request.use_nepal_mode:
                    object_info += "could be a domestic animal or pet commonly found in Nepalese households"
                    object_info += "Many Nepalese families keep dogs as guardians for their homes"
                else:
                    object_info += "appears to be a domestic animal or pet"
                object_info += ". "
                
            elif request.object_class.lower() in ['tree', 'plant', 'flower', 'vegetation']:
                object_info += f"This {request.object_class} "
                if request.use_nepal_mode:
                    if 'rhododendron' in description.lower() or 'lali gurans' in description.lower():
                        object_info += "is likely the national flower of Nepal (Lali Gurans), which blooms beautifully in spring"
                    elif 'pine' in description.lower():
                        object_info += "is a pine tree, commonly found in the hilly regions of Nepal"
                    elif 'oak' in description.lower():
                        object_info += "could be an oak tree, providing valuable timber and shade in rural areas"
                    else:
                        object_info += "is a plant species that contributes to Nepal's rich biodiversity"
                else:
                    object_info += "is vegetation that adds to the natural beauty of the Nepalese landscape"
                object_info += ". "
                
            elif request.object_class.lower() in ['building', 'house', 'structure']:
                object_info += f"This {request.object_class} "
                if request.use_nepal_mode:
                    object_info += "features traditional Nepalese architecture with intricate woodwork and stone carvings"
                    object_info += "Many such buildings can be found in historic areas like Kathmandu Durbar Square and Bhaktapur"
                else:
                    object_info += "is a man-made structure used for shelter or commerce"
                object_info += ". "

        # Enhance the factual description with object information
        if object_info:
            enhanced_factual = f"{object_info.strip()}\n\n{factual.strip()}"
        else:
            enhanced_factual = factual

        return ApiResponse(
            success=True,
            data={
                "full_description": description,
                "engagingDescription": engaging,
                "factualDescription": enhanced_factual,
                "object_id": request.object_id
            },
            message="Caption generated successfully",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Caption generation error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Caption generation failed: {str(e)}",
            error_code="CAPTION_ERROR",
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
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Object extraction error: {str(e)}")
        return ApiResponse(
            success=False,
            message=f"Object extraction failed: {str(e)}",
            error_code="EXTRACTION_ERROR",
            timestamp=datetime.now(timezone.utc).isoformat()
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