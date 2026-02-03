from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class MatchResult(BaseModel):
    class_name: str
    score: float
    box: List[float]  # [x1, y1, x2, y2]
    matched: bool
    details: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    matches: List[MatchResult]
    inference_time: float
    image_size: List[int] # [width, height]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device_info: Dict[str, str]
