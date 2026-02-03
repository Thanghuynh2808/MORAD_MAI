import cv2
import numpy as np
from typing import List, Tuple
from retail_matcher.utils.common import logger

def apply_clahe(img_cv2: np.ndarray) -> np.ndarray:
    """Applies CLAHE to enhance contrast."""
    if img_cv2 is None or img_cv2.size == 0:
        return img_cv2
    try:
        lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"CLAHE failed: {e}, returning original image")
        return img_cv2

def batch_apply_clahe(crops: List[np.ndarray]) -> List[np.ndarray]:
    return [apply_clahe(crop) for crop in crops]

def preprocess_image_for_onnx(img_cv2: np.ndarray, target_size: Tuple[int, int] = (512, 512)):
    """
    Prepares image for SuperPoint ONNX:
    Resize -> Grayscale -> Normalize [0,1] -> Shape (1, 1, H, W)
    Returns: tensor, new_width, new_height
    """
    if img_cv2 is None or img_cv2.size == 0:
        return None, 0, 0

    h, w = img_cv2.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    nw, nh = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img_cv2, (nw, nh), interpolation=cv2.INTER_AREA)

    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized

    img_norm = img_gray.astype(np.float32) / 255.0
    img_tensor = img_norm[None, None, :, :]  # (1, 1, H, W)

    return img_tensor, nw, nh

def normalize_keypoints(kpts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalizes keypoints to [-1, 1] range for LightGlue."""
    size = np.array([w, h], dtype=np.float32)
    shift = size / 2
    scale = size.max() / 2
    kpts_norm = (kpts - shift) / scale
    return kpts_norm.astype(np.float32)
