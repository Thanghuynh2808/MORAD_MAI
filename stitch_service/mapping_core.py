import cv2 as cv
import numpy as np
import sys

sys.path.append(".")

from stitching import Stitcher
from stitching.images import Images

# Import Cluster-based Voting algorithm
try:
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent
                          / "Retail-Product-Matching"))
    from retail_matcher.utils.processing import map_products_to_price_tags
except ImportError:
    map_products_to_price_tags = None


def _normalize_det_map(detections_map):
    """
    Normalise detections_map to the new dict-of-dicts format.

    Accepts both the legacy flat format:
        {idx: [{"box": ..., "class_name": ..., "score": ...}, ...]}
    and the new format:
        {idx: {"products": [...], "price_tags": [...]}}

    Always returns:
        {idx: {"products": [...], "price_tags": [...]}}
    """
    normalised = {}
    for idx, value in detections_map.items():
        if isinstance(value, dict) and ("products" in value or "price_tags" in value):
            # New format – just ensure both keys exist
            normalised[idx] = {
                "products": value.get("products", []),
                "price_tags": value.get("price_tags", []),
            }
        elif isinstance(value, list):
            # Legacy flat format – treat everything as products, no tags
            normalised[idx] = {"products": value, "price_tags": []}
        else:
            normalised[idx] = {"products": [], "price_tags": []}
    return normalised


def _warp_box(box, camera, warper, scale_x, scale_y, offset_x, offset_y):
    """
    Warp a single [x1, y1, x2, y2] box through the panorama transform.

    Returns the new [x_min, y_min, x_max, y_max] in panorama coordinates.
    """
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    warped_poly = []
    for (x_orig, y_orig) in corners:
        x_med = x_orig * scale_x
        y_med = y_orig * scale_y
        p_warped = warper.warpPoint(
            (x_med, y_med), camera.K().astype(np.float32), camera.R
        )
        x_canvas = p_warped[0] + offset_x
        y_canvas = p_warped[1] + offset_y
        warped_poly.append((int(x_canvas), int(y_canvas)))

    x_min = min(p[0] for p in warped_poly)
    y_min = min(p[1] for p in warped_poly)
    x_max = max(p[0] for p in warped_poly)
    y_max = max(p[1] for p in warped_poly)
    return [x_min, y_min, x_max, y_max]


def run_mapping_core(images_list, detections_map):
    """
    Core function: stitch images, warp product + price-tag bounding boxes to
    panorama coordinates, then run Cluster-based Voting to assign each product
    its price tag.

    Args
    ----
    images_list : list[np.ndarray]
        Raw input frames (BGR), in capture order.

    detections_map : dict
        Keyed by frame index (int).  Supports two formats:

        **New format** (recommended):
            {
                idx: {
                    "products":   [{"box": [x1,y1,x2,y2], "class_name": str, "score": float}, ...],
                    "price_tags": [{"box": [x1,y1,x2,y2], "price": str|None, "tag_id": int}, ...],
                }
            }

        **Legacy format** (backwards-compatible):
            {idx: [{"box": ..., "class_name": ..., "score": ...}, ...]}
            (treated as products only, no price tags)

    Returns
    -------
    panorama : np.ndarray
    products : list[dict]
        Warped product dicts — each has a ``"price_tag"`` key pointing to the
        matched price-tag dict (or ``None``).
    price_tags : list[dict]
        All warped price-tag dicts.
    """

    # ── 1. Stitch ────────────────────────────────────────────────────────────
    stitcher = Stitcher()
    panorama = stitcher.stitch(images_list)

    if panorama is None:
        raise ValueError("Stitching failed. Images might not have enough overlap.")

    # ── 2. Extract transformation metadata ───────────────────────────────────
    if not hasattr(stitcher, "cameras"):
        raise ValueError("Stitcher metadata (cameras) missing.")

    cameras = stitcher.cameras
    warped_corners = stitcher.warped_corners
    final_corners = stitcher.final_corners
    images_handler = stitcher.images

    warper_type = stitcher.warper.warper_type
    scale = stitcher.warper.scale
    aspect_med_final = images_handler.get_ratio(
        Images.Resolution.MEDIUM, Images.Resolution.FINAL
    )
    aspect_low_final = images_handler.get_ratio(
        Images.Resolution.LOW, Images.Resolution.FINAL
    )

    final_warper_scale = scale * aspect_med_final
    warper = cv.PyRotationWarper(warper_type, final_warper_scale)

    # Normalise detections_map to new format
    det_map = _normalize_det_map(detections_map)

    # ── 3. Warp products AND price tags to panorama coords ───────────────────
    all_warped_products = []
    all_warped_tags = []

    for idx, img in enumerate(images_list):
        frame_dets = det_map.get(idx, {"products": [], "price_tags": []})
        products_raw = frame_dets["products"]
        tags_raw = frame_dets["price_tags"]

        if not products_raw and not tags_raw:
            continue

        camera = cameras[idx]
        h_orig, w_orig = img.shape[:2]

        med_size = images_handler.get_scaled_img_sizes(Images.Resolution.MEDIUM)[idx]
        w_med, h_med = med_size
        scale_x = w_med / w_orig
        scale_y = h_med / h_orig

        tl_warped = warped_corners[idx]
        tl_final = final_corners[idx]
        intersection_rect = stitcher.cropper.intersection_rectangles[idx]
        crop_rect_final = intersection_rect.times(aspect_low_final)

        offset_x = -tl_warped[0] - crop_rect_final.x + tl_final[0]
        offset_y = -tl_warped[1] - crop_rect_final.y + tl_final[1]

        # Warp products
        for det in products_raw:
            warped_box = _warp_box(
                det["box"], camera, warper, scale_x, scale_y, offset_x, offset_y
            )
            all_warped_products.append({
                "box": warped_box,
                "class_name": det.get("class_name", "unknown"),
                "score": float(det.get("score", 0.0)),
                "original_image_idx": idx,
            })

        # Warp price tags — pass all metadata through unchanged
        for tag in tags_raw:
            warped_box = _warp_box(
                tag["box"], camera, warper, scale_x, scale_y, offset_x, offset_y
            )
            warped_tag = {k: v for k, v in tag.items() if k != "box"}
            warped_tag["box"] = warped_box
            warped_tag["original_image_idx"] = idx
            all_warped_tags.append(warped_tag)

    # ── 4. NMS on products only (tags are EXCLUDED to avoid suppression) ─────
    products_after_nms = apply_nms(all_warped_products)

    # ── 5. Cluster-based Voting: assign tag to each product group ────────────
    if map_products_to_price_tags is not None and all_warped_tags:
        products_after_nms = map_products_to_price_tags(
            products_after_nms, all_warped_tags
        )
    else:
        # No tags available or voting module not importable — set price_tag = None
        for p in products_after_nms:
            p.setdefault("price_tag", None)

    return panorama, products_after_nms, all_warped_tags


def apply_nms(detections, iou_thresh=0.5):
    """
    Run Non-Maximum Suppression on a list of product detection dicts.

    NOTE: Do NOT pass price tags into this function — their low scores would
    cause them to be suppressed by neighbouring product boxes.
    """
    if not detections:
        return []

    boxes = [d["box"] for d in detections]
    scores = [d["score"] for d in detections]

    boxes_wh = [
        [x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes
    ]

    indices = cv.dnn.NMSBoxes(
        boxes_wh, scores, score_threshold=0.0, nms_threshold=iou_thresh
    )

    if len(indices) > 0:
        return [detections[i] for i in indices.flatten()]
    return []
