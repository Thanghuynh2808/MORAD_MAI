from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PriceTagResult(BaseModel):
    """Represents a single price tag detected and OCR-read on the shelf."""
    tag_id: int
    price: Optional[str] = None           # Giá hiển thị chính (= giá KM nếu có, ngược lại = giá thường)
    original_price: Optional[str] = None  # Giá gốc (giá cao hơn trên tag, nếu có 2 giá)
    discount_price: Optional[str] = None  # Giá khuyến mãi (giá thấp hơn trên tag, nếu có 2 giá)
    box: List[float]                      # [x1, y1, x2, y2] in panorama coords


class MappedItem(BaseModel):
    """A detected product together with its matched price tag (if any)."""
    class_name: str
    score: float
    box: List[float]                     # [x1, y1, x2, y2] in panorama coords
    matched: bool
    price_tag: Optional[PriceTagResult] = None  # None when no tag was linked
    details: Optional[Dict[str, Any]] = None


# Backwards-compatibility alias — existing code that imports MatchResult still works
MatchResult = MappedItem


class PredictionResponse(BaseModel):
    matches: List[MappedItem]            # one entry per detected product
    price_tags: List[PriceTagResult]     # all warped price tags in this frame/panorama
    inference_time: float
    image_size: List[int]                # [width, height]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device_info: Dict[str, str]
