
import cv2 as cv
import numpy as np
import sys
# Add current directory to path so imports work
sys.path.append(".") 

from stitching import Stitcher
from stitching.images import Images

def run_mapping_core(images_list, detections_map):
    """
    Core function to stitch images and map detections without re-running detection.
    
    Args:
        images_list: List of cv2 images (numpy arrays)
        detections_map: Dictionary mapping index (int) or filename to detections list.
                        Each detection is (corners, score, class_name, original_idx/name)
                        Since we receive images as list, we map by INDEX in list.
    
    Returns:
        panorama: cv2 image
        mapped_detections: List of dicts with mapped 'box' and metadata
    """
    
    # 1. Stitch Images
    stitcher = Stitcher()
    # By default, passing list of numpy arrays works.
    panorama = stitcher.stitch(images_list)
    
    if panorama is None:
        raise ValueError("Stitching failed. Images might not have enough overlap.")

    # 2. Extract Transformation Data
    if not hasattr(stitcher, 'cameras'):
        raise ValueError("Stitcher metadata (cameras) missing.")
        
    cameras = stitcher.cameras
    warped_corners = stitcher.warped_corners
    final_corners = stitcher.final_corners
    images_handler = stitcher.images
    
    # Warper configuration
    warper_type = stitcher.warper.warper_type
    scale = stitcher.warper.scale 
    aspect_med_final = images_handler.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
    aspect_low_final = images_handler.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
    
    final_warper_scale = scale * aspect_med_final
    warper = cv.PyRotationWarper(warper_type, final_warper_scale)
    
    # 3. Process Detections
    all_warped_detections = []
    
    for idx, img in enumerate(images_list):
        # Get detections for this image index
        # We assume strict ordering matching images_list
        img_dets = detections_map.get(idx, [])
        if not img_dets:
            continue
            
        camera = cameras[idx]
        h_orig, w_orig = img.shape[:2]
        
        # Get Medium size used for calibration
        med_size = images_handler.get_scaled_img_sizes(Images.Resolution.MEDIUM)[idx]
        w_med, h_med = med_size
        
        # Scale factor Original -> Medium
        scale_x = w_med / w_orig
        scale_y = h_med / h_orig
        
        # Offsets
        tl_warped = warped_corners[idx]
        tl_final = final_corners[idx]
        intersection_rect = stitcher.cropper.intersection_rectangles[idx]
        crop_rect_final = intersection_rect.times(aspect_low_final)
        
        offset_x = -tl_warped[0] - crop_rect_final.x + tl_final[0]
        offset_y = -tl_warped[1] - crop_rect_final.y + tl_final[1]
        
        for det in img_dets:
            # det is expected to be object with .box [x1, y1, x2, y2]
            x1, y1, x2, y2 = det['box']
            corners = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            ]
            
            warped_poly = []
            for (x_orig, y_orig) in corners:
                # 1. Scale
                x_med = x_orig * scale_x
                y_med = y_orig * scale_y
                
                # 2. Warp
                p_warped = warper.warpPoint((x_med, y_med), camera.K().astype(np.float32), camera.R)
                
                # 3. Offset
                x_canvas = p_warped[0] + offset_x
                y_canvas = p_warped[1] + offset_y
                
                warped_poly.append((int(x_canvas), int(y_canvas)))
            
            # Find new bbox
            x_min = min(p[0] for p in warped_poly)
            y_min = min(p[1] for p in warped_poly)
            x_max = max(p[0] for p in warped_poly)
            y_max = max(p[1] for p in warped_poly)
            
            all_warped_detections.append({
                'box': [x_min, y_min, x_max, y_max],
                'class_name': det.get('class_name', 'unknown'),
                'score': det.get('score', 0.0),
                'original_image_idx': idx
            })
            
    # 4. NMS (Optional but recommended)
    # We will implement simple NMS here to clean up overlaps
    final_results = apply_nms(all_warped_detections)
    
    return panorama, final_results

def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return []
        
    boxes = [d['box'] for d in detections]
    scores = [d['score'] for d in detections]
    
    # Convert [x1, y1, x2, y2] to [x, y, w, h] for cv.dnn.NMSBoxes
    # NMSBoxes expects top-left x, y and width, height
    boxes_wh = []
    for (x1, y1, x2, y2) in boxes:
        boxes_wh.append([x1, y1, x2 - x1, y2 - y1])
        
    indices = cv.dnn.NMSBoxes(boxes_wh, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    
    keep_dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            keep_dets.append(detections[i])
            
    return keep_dets
