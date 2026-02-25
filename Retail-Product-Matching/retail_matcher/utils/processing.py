import cv2
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional
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


def map_products_to_price_tags(
    products: List[Dict],
    price_tags: List[Dict],
) -> List[Dict]:
    """
    Map each product bounding box to the most likely price tag using a
    3-step Cluster-based Voting algorithm.

    Steps
    -----
    1. Local Mapping  – For each product, find the closest price tag that is
       *below* the product and *overlaps* it on the X-axis.
    2. Clustering     – Group products by their ``class_name``.
    3. Voting         – Within each class group, the price tag with the most
       votes (local mappings) is assigned to every product in the group.

    Parameters
    ----------
    products : List[Dict]
        Each dict must contain ``"box": [x1, y1, x2, y2]`` and
        ``"class_name": str``.
    price_tags : List[Dict]
        Each dict must contain ``"box": [x1, y1, x2, y2]``. A ``"tag_id"``
        key is added automatically if missing.

    Returns
    -------
    List[Dict]
        Same ``products`` list (mutated in-place) with a ``"price_tag"`` key
        added to every item. Value is the matched tag dict or ``None``.
    """
    # --- Normalise: ensure every tag has a unique tag_id --------------------
    tag_index: Dict[int, Dict] = {}
    for i, tag in enumerate(price_tags):
        if "tag_id" not in tag:
            tag["tag_id"] = i
        tag_index[tag["tag_id"]] = tag

    # --- Step 1: Local Mapping ----------------------------------------------
    local_votes: List[Optional[int]] = []   # one entry per product

    for product in products:
        px1, py1, px2, py2 = product["box"][:4]

        best_tag_id: Optional[int] = None
        best_dist: float = float("inf")

        for tag in price_tags:
            tx1, ty1, tx2, ty2 = tag["box"][:4]

            # Condition A: tag top must be at or below the product bottom.
            # Allow 20% vertical overlap tolerance (e.g. tag slightly overlaps product footer).
            tolerance = (py2 - py1) * 0.2
            if ty1 < (py2 - tolerance):
                continue

            # Condition B: X-ranges must overlap
            if tx2 <= px1 or tx1 >= px2:
                continue

            # Metric: vertical distance between product bottom and tag top
            dist = ty1 - py2
            if dist < best_dist:
                best_dist = dist
                best_tag_id = tag["tag_id"]

        local_votes.append(best_tag_id)

    # --- Step 2: Cluster by class_name --------------------------------------
    class_groups: Dict[str, List[int]] = {}   # class_name -> list of product indices
    for idx, product in enumerate(products):
        cls = product.get("class_name", "__unknown__")
        class_groups.setdefault(cls, []).append(idx)

    # --- Step 3: Voting ------------------------------------------------------
    for cls, indices in class_groups.items():
        votes = [local_votes[i] for i in indices if local_votes[i] is not None]

        if votes:
            winner_tag_id, count = Counter(votes).most_common(1)[0]
            winning_tag = tag_index[winner_tag_id]
            logger.debug(
                f"Class '{cls}': tag_id={winner_tag_id} won with {count}/{len(indices)} votes"
            )
        else:
            winning_tag = None
            logger.debug(f"Class '{cls}': no price tag candidate found")

        for i in indices:
            products[i]["price_tag"] = winning_tag

    return products
