
import cv2 as cv
import numpy as np
import uvicorn
import json
import base64
import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.schemas import StitchResponse, MappedProduct
from mapping_core import run_mapping_core

app = FastAPI(title="Stitching & Mapping Service")

# Create temp dir
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/stitch-with-mapping", response_model=StitchResponse)
async def stitch_with_mapping(
    files: List[UploadFile] = File(...),
    detections: str = Form(...) # Expecting JSON string
):
    """
    Upload images and their corresponding detections JSON.
    Returns assembled panorama and mapped bounding boxes.
    """
    try:
        # 1. Parse Detections JSON
        # Map: filename -> list of detections
        try:
            detections_map_raw = json.loads(detections)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for detections")
            
        # 2. Process Images
        images_list = []
        ordered_filenames = []
        ordered_detections_map = {} # Map index -> detections
        
        for idx, file in enumerate(files):
            # Read file into numpy
            content = await file.read()
            nparr = np.frombuffer(content, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
            
            if img is None:
                continue
                
            images_list.append(img)
            ordered_filenames.append(file.filename)
            
            # Map detections to this index using filename
            if file.filename in detections_map_raw:
                ordered_detections_map[idx] = detections_map_raw[file.filename]
            
        if len(images_list) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid images to stitch.")
            
        # 3. Run Stitching & Mapping
        try:
            panorama, mapped_results = run_mapping_core(images_list, ordered_detections_map)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")
            
        # 4. Save/Process Result
        # For simplicity, we'll encode panorama as Base64. 
        # In production, save to S3/Disk and return URL.
        retval, buffer = cv.imencode('.jpg', panorama)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # 5. Format Response
        response_products = []
        for det in mapped_results:
            response_products.append(MappedProduct(
                class_name=det['class_name'],
                box=det['box'],
                score=det['score'],
                original_image=ordered_filenames[det['original_image_idx']]
            ))
            
        return StitchResponse(
            panorama_width=panorama.shape[1],
            panorama_height=panorama.shape[0],
            mapped_products=response_products,
            panorama_url=f"data:image/jpeg;base64,{jpg_as_text}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8001, reload=False)
