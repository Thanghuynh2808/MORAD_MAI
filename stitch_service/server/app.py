
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

import httpx
import asyncio
from typing import List, Dict

# URL of the RPM Service in Docker network or Local
RPM_API_URL = os.getenv("RPM_API_URL", "http://localhost:8000/predict")

@app.get("/")
async def root():
    return {"message": "Stitching & Mapping Service is running", "docs": "/docs"}

async def get_detection_from_rpm(client, file_content, filename):
    """Call RPM API to get detections for one image."""
    try:
        files = {"file": (filename, file_content, "image/jpeg")}
        response = await client.post(RPM_API_URL, files=files, timeout=30.0)
        if response.status_code == 200:
            return filename, response.json().get("matches", [])
        else:
            logger.error(f"RPM API error for {filename}: {response.status_code}")
            return filename, []
    except Exception as e:
        logger.error(f"Failed to call RPM API: {e}")
        return filename, []

@app.post("/upload-batch", response_model=StitchResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Orchestrator endpoint for Mobile App:
    1. Receive multiple images.
    2. Call RPM Service in parallel for each image.
    3. Run stitching & mapping.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images.")

    try:
        # Load all images into memory and prepare for RPM calls
        images_data = []
        for file in files:
            content = await file.read()
            images_data.append((file.filename, content))

        # 1. Parallel call to RPM Service
        async with httpx.AsyncClient() as client:
            tasks = [get_detection_from_rpm(client, content, fname) for fname, content in images_data]
            results = await asyncio.gather(*tasks)
            
        detections_map = {fname: dets for fname, dets in results}

        # 2. Convert raw bytes to CV2 images
        images_list = []
        ordered_filenames = []
        ordered_detections_map = {}
        
        for idx, (fname, content) in enumerate(images_data):
            nparr = np.frombuffer(content, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
            if img is not None:
                images_list.append(img)
                ordered_filenames.append(fname)
                ordered_detections_map[len(images_list)-1] = detections_map[fname]

        # 3. Stitch & Map
        panorama, mapped_results = run_mapping_core(images_list, ordered_detections_map)

        # 4. Finalize
        retval, buffer = cv.imencode('.jpg', panorama)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
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

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
