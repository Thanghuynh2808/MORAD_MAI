import cv2 as cv
import numpy as np
import uvicorn
import json
import base64
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import sys
import httpx
import asyncio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.schemas import StitchResponse, MappedProduct, MappedPriceTag
from mapping_core import run_mapping_core

app = FastAPI(title="Stitching & Mapping Service")

import logging
logger = logging.getLogger("stitch_service")

# URL of the RPM Service in Docker network or Local
RPM_API_URL = os.getenv("RPM_API_URL", "http://localhost:8000/predict")


@app.get("/")
async def root():
    return {"message": "Stitching & Mapping Service is running", "docs": "/docs"}


async def get_detection_from_rpm(client, file_content, filename):
    """
    Call RPM API to get product detections AND price tags for one image.

    Returns:
        (filename, {"products": [...], "price_tags": [...]})
    """
    try:
        files = {"file": (filename, file_content, "image/jpeg")}
        response = await client.post(RPM_API_URL, files=files, timeout=30.0)
        if response.status_code == 200:
            body = response.json()
            # Bug 2 fix: extract BOTH "matches" (products) AND "price_tags" from RPM response
            products = body.get("matches", [])
            price_tags = body.get("price_tags", [])

            # Normalise product dicts to the mapping_core expected shape
            normalised_products = [
                {
                    "box": m["box"],
                    "class_name": m.get("class_name", "unknown"),
                    "score": float(m.get("score", 0.0)),
                }
                for m in products
            ]

            # Normalise tag dicts
            normalised_tags = [
                {
                    "box": t["box"],
                    "price": t.get("price"),
                    "tag_id": t.get("tag_id", i),
                }
                for i, t in enumerate(price_tags)
            ]

            return filename, {"products": normalised_products, "price_tags": normalised_tags}
        else:
            logger.error(f"RPM API error for {filename}: {response.status_code}")
            return filename, {"products": [], "price_tags": []}
    except Exception as e:
        logger.error(f"Failed to call RPM API for {filename}: {e}")
        return filename, {"products": [], "price_tags": []}


def _build_response(panorama, products, price_tags, ordered_filenames):
    """
    Encode panorama as base64 and build the StitchResponse object.
    """
    retval, buffer = cv.imencode('.jpg', panorama)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    response_products = []
    for det in products:
        # Build optional price_tag sub-object
        pt_raw = det.get("price_tag")
        price_tag_obj = None
        if pt_raw is not None:
            price_tag_obj = MappedPriceTag(
                tag_id=int(pt_raw.get("tag_id", 0)),
                price=pt_raw.get("price"),
                box=[float(x) for x in pt_raw["box"]],
            )

        orig_idx = det.get("original_image_idx", 0)
        orig_name = (
            ordered_filenames[orig_idx]
            if orig_idx < len(ordered_filenames)
            else "unknown"
        )
        response_products.append(MappedProduct(
            class_name=det.get("class_name", "unknown"),
            box=[float(x) for x in det["box"]],
            score=float(det.get("score", 0.0)),
            original_image=orig_name,
            price_tag=price_tag_obj,
        ))

    all_tags = [
        MappedPriceTag(
            tag_id=int(t.get("tag_id", i)),
            price=t.get("price"),
            box=[float(x) for x in t["box"]],
        )
        for i, t in enumerate(price_tags)
    ]

    return StitchResponse(
        panorama_width=panorama.shape[1],
        panorama_height=panorama.shape[0],
        mapped_products=response_products,
        price_tags=all_tags,
        panorama_url=f"data:image/jpeg;base64,{jpg_as_text}",
    )


@app.post("/upload-batch", response_model=StitchResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Orchestrator endpoint for Mobile App:
    1. Receive multiple images.
    2. Call RPM Service in parallel for each image (products + price tags).
    3. Run stitching & mapping with Cluster-based Voting.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images.")

    try:
        images_data = []
        for file in files:
            content = await file.read()
            images_data.append((file.filename, content))

        # 1. Parallel call to RPM Service — Bug 2 fix: now receives both products and tags
        async with httpx.AsyncClient() as client:
            tasks = [
                get_detection_from_rpm(client, content, fname)
                for fname, content in images_data
            ]
            results = await asyncio.gather(*tasks)

        detections_map_by_name = {fname: dets for fname, dets in results}

        # 2. Convert raw bytes to CV2 images and build ordered detection map
        images_list = []
        ordered_filenames = []
        ordered_detections_map = {}

        for fname, content in images_data:
            nparr = np.frombuffer(content, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
            if img is not None:
                idx = len(images_list)
                images_list.append(img)
                ordered_filenames.append(fname)
                ordered_detections_map[idx] = detections_map_by_name[fname]

        if len(images_list) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid images to stitch.")

        # 3. Stitch & Map — Bug 1 fix: unpack 3 return values
        panorama, products, price_tags = run_mapping_core(images_list, ordered_detections_map)

        return _build_response(panorama, products, price_tags, ordered_filenames)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stitch-with-mapping", response_model=StitchResponse)
async def stitch_with_mapping(
    files: List[UploadFile] = File(...),
    detections: str = Form(...)  # Expecting JSON string
):
    """
    Upload images and their corresponding detections JSON.
    The JSON may use the new format:
        {"filename.jpg": {"products": [...], "price_tags": [...]}}
    or legacy flat format:
        {"filename.jpg": [...]}   (treated as products only)

    Returns assembled panorama and mapped bounding boxes with price tags.
    """
    try:
        try:
            detections_map_raw = json.loads(detections)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for detections")

        images_list = []
        ordered_filenames = []
        ordered_detections_map = {}

        for file in files:
            content = await file.read()
            nparr = np.frombuffer(content, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)

            if img is None:
                continue

            idx = len(images_list)
            images_list.append(img)
            ordered_filenames.append(file.filename)

            if file.filename in detections_map_raw:
                ordered_detections_map[idx] = detections_map_raw[file.filename]

        if len(images_list) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid images to stitch.")

        # Bug 1 fix: unpack 3 return values from run_mapping_core
        try:
            panorama, products, price_tags = run_mapping_core(images_list, ordered_detections_map)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")

        return _build_response(panorama, products, price_tags, ordered_filenames)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8001, reload=False)
