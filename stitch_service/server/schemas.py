from pydantic import BaseModel
from typing import List, Dict, Optional


class Detection(BaseModel):
    box: List[float]        # [x1, y1, x2, y2]
    class_name: str
    score: float


class StitchRequest(BaseModel):
    # Map filename to list of detections
    detections: Dict[str, List[Detection]]


class MappedPriceTag(BaseModel):
    """Price tag warped to panorama coordinates."""
    tag_id: int
    price: Optional[str] = None           # Giá hiển thị chính
    original_price: Optional[str] = None  # Giá gốc (cao hơn, nếu có 2 giá)
    discount_price: Optional[str] = None  # Giá khuyến mãi (thấp hơn, nếu có 2 giá)
    box: List[float]                      # [x1, y1, x2, y2] on panorama


class MappedProduct(BaseModel):
    """A detected product together with its matched price tag (if any)."""
    class_name: str
    box: List[float]               # [x1, y1, x2, y2] on panorama
    score: float
    original_image: str
    price_tag: Optional[MappedPriceTag] = None   # None when no tag was linked


class StitchResponse(BaseModel):
    panorama_width: int
    panorama_height: int
    mapped_products: List[MappedProduct]
    price_tags: List[MappedPriceTag] = []        # All warped tags in the panorama
    panorama_url: Optional[str] = None
