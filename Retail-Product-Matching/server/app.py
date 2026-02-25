import sys
from pathlib import Path
from contextlib import asynccontextmanager
import time
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports from core logic
from retail_matcher.pipeline import ProductMatcher
from retail_matcher.utils.config import load_config
from retail_matcher.utils.common import logger
from server.schemas import PredictionResponse, HealthResponse, MappedItem, PriceTagResult

# Global Matcher Instance
matcher_instance = None
CONFIG = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup and clean up on shutdown.
    """
    global matcher_instance, CONFIG
    logger.info("Server starting up...")
    
    # Load Configuration
    CONFIG = load_config()
    
    # Initialize Matcher
    try:
        matcher = ProductMatcher(CONFIG)
        if matcher.load_gallery(CONFIG.support_db):
            matcher_instance = matcher
            logger.info("ProductMatcher initialized successfully and Gallery loaded.")
        else:
            logger.error("Failed to load gallery database. Inference will fail.")
            matcher_instance = None
    except Exception as e:
        logger.error(f"Failed to initialize ProductMatcher: {e}")
        matcher_instance = None
        
    yield
    
    # Cleanup (if needed)
    logger.info("Server shutting down...")
    matcher_instance = None

app = FastAPI(title="Retail Product Matching API", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Retail Product Matching API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server matches are ready."""
    is_loaded = matcher_instance is not None
    devices = matcher_instance.devices if is_loaded else {}
    return {
        "status": "ok" if is_loaded else "error", 
        "model_loaded": is_loaded,
        "device_info": devices
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict products in the uploaded image.
    """
    if matcher_instance is None:
        raise HTTPException(status_code=503, detail="Model server is not ready yet.")

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
            
        height, width = img.shape[:2]

        # Run pipeline
        # Note: Since process_image is synchronous (CPU/GPU bound), 
        # normally we should run it in a threadpool, but FastAPI manages sync defs in threadpool automatically.
        # But here 'predict' is async so we should ideally await a thread run.
        # However, for simplicity now, we'll run it directly (it might block event loop briefly).
        
        # NOTE: pipeline.process_image expects a path, but we have an image array.
        # We need to adapt process_image to accept numpy arrays too.
        # Let's modify pipeline.py or create a wrapper. 
        # Modifying pipeline to accept loaded images is better design.
        
        # Temporary workaround: Save to tmp or modify pipeline. 
        # Let's assume we modify pipeline.py to accept numpy image in next step.
        # For now, let's look at pipeline.py lines 62-67: it explicitly reads from path.
        # We MUST refactor pipeline.py to support existing image array.
        
        # ... Wait, I'll refactor pipeline.py in the next tool call.
        # Assuming matcher_instance.process_image supports numpy array or I create a helper.
        
        # Let's assume process_image_memory method exists (I will add it).
        _, results = matcher_instance.process_image(img)
        
        if not results:
             raise HTTPException(status_code=500, detail="Inference returned no results.")

        # Format Response
        matches_formatted = []
        for m in results.get('matches', []):
            # Build optional price_tag sub-object
            pt_raw = m.get('price_tag')
            price_tag_obj = None
            if pt_raw is not None:
                price_tag_obj = PriceTagResult(
                    tag_id=pt_raw.get('tag_id', 0),
                    price=pt_raw.get('price'),
                    box=[float(x) for x in pt_raw['box']],
                )

            matches_formatted.append(MappedItem(
                class_name=m.get('class', 'Unknown'),
                score=float(m.get('score', 0.0)),
                box=[float(x) for x in m['box']],
                matched=m.get('matched', False),
                price_tag=price_tag_obj,
                details=m.get('details'),
            ))

        # Collect all warped price tags (may be empty for single-frame requests)
        all_tags = [
            PriceTagResult(
                tag_id=t.get('tag_id', i),
                price=t.get('price'),
                box=[float(x) for x in t['box']],
            )
            for i, t in enumerate(results.get('price_tags', []))
        ]

        return PredictionResponse(
            matches=matches_formatted,
            price_tags=all_tags,
            inference_time=results.get('timing', {}).get('total', 0.0),
            image_size=[width, height],
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
