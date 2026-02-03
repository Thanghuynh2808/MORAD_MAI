from pydantic import BaseModel
from typing import List, Dict, Tuple

class Detection(BaseModel):
    box: List[float] # [x1, y1, x2, y2]
    class_name: str
    score: float

class StitchRequest(BaseModel):
    # Map filename to list of detections
    detections: Dict[str, List[Detection]]

class MappedProduct(BaseModel):
    class_name: str
    box: List[int] # [x1, y1, x2, y2] on panorama
    score: float
    original_image: str

class StitchResponse(BaseModel):
    panorama_width: int
    panorama_height: int
    mapped_products: List[MappedProduct]
    # We might return download URL or Base64 string separately
    panorama_url: str = None
