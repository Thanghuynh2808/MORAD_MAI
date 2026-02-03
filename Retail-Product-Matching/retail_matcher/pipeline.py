import time
import cv2
import numpy as np
import torch
from retail_matcher.utils.common import logger
from retail_matcher.utils.processing import apply_clahe, preprocess_image_for_onnx
from retail_matcher.models.loader import load_model
from retail_matcher.models.extraction import batch_extract_dino_features
from retail_matcher.models.matching import matrix_matching, run_fast_hybrid_matching

class ProductMatcher:
    def __init__(self, config):
        """
        Initialize ProductMatcher with configuration.
        """
        self.cfg = config
        
        # Load models using the flexible loader
        (self.yolo_model, 
         self.dinov3_processor, 
         self.dinov3_model, 
         self.sp_session, 
         self.lg_session, 
         self.devices) = load_model(config)
        
        # Ensure DINO device is a torch device object if needed
        self.dino_device = torch.device(self.devices['dino'])
        
        self.support_db = None

    def load_gallery(self, db_path):
        try:
            db_path = str(db_path)
            self.support_db = torch.load(db_path, map_location='cpu', weights_only=False)
            
            required_keys = ['gallery_matrix', 'sp_features', 'class_names']
            if not all(key in self.support_db for key in required_keys):
                missing = [k for k in required_keys if k not in self.support_db]
                logger.error(f"Invalid support_db structure. Missing keys: {missing}")
                return False
                
            logger.info(f"Support DB loaded: {len(self.support_db['class_names'])} images")
            return True
        except Exception as e:
            logger.error(f"Failed to load support DB: {e}")
            return False

    def process_image(self, image_path):
        """
        Process a single image: Detect -> Extract -> Match.
        Returns: 
            img (cv2 image), results (dict with matches, boxes, timing)
        """
        if self.support_db is None:
            logger.error("Support DB not loaded!")
            return None, None

        start_time = time.time()
        timing_info = {}

        # 1. Load Image
        io_load_start = time.time()
        img = cv2.imread(str(image_path))
        timing_info['io_load'] = time.time() - io_load_start
        
        if img is None:
            logger.error(f"Failed to load image from {image_path}")
            return None, None

        # 2. YOLO Detection
        yolo_start = time.time()
        try:
            yolo_results = self.yolo_model(str(image_path), conf=self.cfg.yolo_conf, iou=0.45, verbose=False)[0]
            timing_info['yolo'] = time.time() - yolo_start
            if self.devices['yolo'] == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            if self.devices['yolo'] == 'cuda':
                torch.cuda.empty_cache()
            return img, {'matches': [], 'boxes': [], 'error': str(e)}

        # ... (rest of YOLO handle) ...
        obb_polygons = []
        boxes = []
        
        if hasattr(yolo_results, 'obb') and yolo_results.obb is not None and len(yolo_results.obb) > 0:
            obb_boxes = yolo_results.obb.xyxyxyxy.cpu().numpy()
            for obb in obb_boxes:
                obb_polygons.append(obb.astype(int))
                x_coords = obb[:, 0]
                y_coords = obb[:, 1]
                boxes.append([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()])
            boxes = [[float(x) for x in box] for box in boxes]
        elif hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes = yolo_results.boxes.xyxy.cpu().numpy().tolist()
            obb_polygons = None

        # Free YOLO result memory
        del yolo_results

        if not boxes:
            return img, {'matches': [], 'boxes': [], 'timing': timing_info}

        # 3. Preprocessing & Cropping
        prep_start = time.time()
        crops_clahe = []
        valid_boxes = []
        valid_obb_indices = []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops_clahe.append(apply_clahe(crop))
                valid_boxes.append((x1, y1, x2, y2))
                valid_obb_indices.append(idx)
        
        timing_info['preprocessing'] = time.time() - prep_start

        if not crops_clahe:
            return img, {'matches': [], 'boxes': [], 'timing': timing_info}

        # 4. DINO Feature Extraction
        dino_start = time.time()
        query_dino = batch_extract_dino_features(
            crops_clahe, self.dinov3_processor, self.dinov3_model, self.dino_device, self.cfg.batch_size
        )
        timing_info['dino'] = time.time() - dino_start
        
        if self.devices['dino'] == 'cuda':
            torch.cuda.empty_cache()

        if query_dino is None:
            return img, {'matches': [], 'boxes': [], 'timing': timing_info, 'error': 'DINO extraction failed'}

        # 5. SuperPoint Extraction
        # ...
        sp_start = time.time()
        query_sp = []
        sp_in_name = self.sp_session.get_inputs()[0].name
        
        for idx, crop in enumerate(crops_clahe):
            img_tensor, nw, nh = preprocess_image_for_onnx(crop)
            if img_tensor is not None:
                try:
                    outs = self.sp_session.run(None, {sp_in_name: img_tensor})
                    if len(outs) >= 3 and outs[0].shape[0] > 0 and outs[2].shape[0] > 0:
                        query_sp.append({'keypoints': outs[0], 'descriptors': outs[2], 'width': nw, 'height': nh})
                    else:
                        query_sp.append(None)
                except Exception:
                    query_sp.append(None)
            else:
                query_sp.append(None)
        timing_info['superpoint'] = time.time() - sp_start

        # 6. Matrix Matching (Coarse)
        matrix_start = time.time()
        candidates_per_query = matrix_matching(query_dino, self.support_db, self.cfg.top_k, self.cfg.dino_thresh)
        timing_info['matrix_matching'] = time.time() - matrix_start

        # Free memory
        del query_dino
        del crops_clahe
        if any(d == 'cuda' for d in self.devices.values()):
            torch.cuda.empty_cache()

        # 7. Batched LightGlue Verification
        lg_start = time.time()
        
        # Pre-calc weights
        total_weight = self.cfg.alpha + self.cfg.beta
        alpha_norm = self.cfg.alpha / total_weight if total_weight > 0 else 0.3
        beta_norm = self.cfg.beta / total_weight if total_weight > 0 else 0.7

        # PHASE 1: Collect all verification jobs
        verification_jobs = [] # List of (query_idx, cand_idx_in_list, query_feat, support_feat, meta_info)
        matches_per_query = [None] * len(boxes) # To store final result for each query

        for i, candidates in enumerate(candidates_per_query):
            x1, y1, x2, y2 = valid_boxes[i]
            obb_poly = obb_polygons[valid_obb_indices[i]] if obb_polygons is not None else None
            
            # Init result structure
            matches_per_query[i] = {
                'box': [x1, y1, x2, y2],
                'obb': obb_poly,
                'class': 'Unknown',
                'score': 0.0,
                'matched': False,
                'candidates': candidates,  # Keep for later
                'best_res': {'score': 0, 'class': "Unknown", 'lg': 0, 'inliers': 0, 'dino_score': 0.0}
            }
            
            if not candidates or not query_sp[i]:
                continue
            
            # Logic to select which candidates to verify
            class_groups = {}
            for idx, name, score in candidates:
                class_groups.setdefault(name, []).append((idx, score))
            
            for name, items in class_groups.items():
                for idx, score in items[:2]: # Top 2 per class
                    # Check for High Confidence DINO -> Skip LG
                    if score >= self.cfg.dino_high_conf_threshold:
                        h_score = alpha_norm * score + beta_norm * 1.0
                        if h_score > matches_per_query[i]['best_res']['score']:
                             matches_per_query[i]['best_res'].update({
                                 'score': h_score, 'class': name, 'lg': 1.0, 
                                 'inliers': self.cfg.skip_lg_inliers_value, 'dino_score': score
                             })
                    else:
                        # Queue for LG verification
                        if idx < len(self.support_db['sp_features']):
                            job = {
                                'query_idx': i,
                                'cls_name': name,
                                'dino_score': score,
                                'query_feat': query_sp[i],
                                'support_feat': self.support_db['sp_features'][idx]
                            }
                            verification_jobs.append(job)

        # PHASE 2: Run Batch Matching
        if verification_jobs:
            pairs_to_verify = [(job['query_feat'], job['support_feat']) for job in verification_jobs]
            
            # Run massive batch match
            lg_results = run_fast_hybrid_matching(
                pairs_to_verify, self.lg_session, batch_size=min(self.cfg.batch_size, 32)
            )
            
            # PHASE 3: Map results back
            for k, (inliers, min_kpts) in enumerate(lg_results):
                job = verification_jobs[k]
                q_idx = job['query_idx']
                d_score = job['dino_score']
                
                lg_score = inliers / min_kpts if min_kpts > 0 else 0
                h_score = alpha_norm * d_score + beta_norm * lg_score
                
                # Update best result for this query
                current_best = matches_per_query[q_idx]['best_res']
                if h_score > current_best['score']:
                    current_best.update({
                        'score': h_score, 
                        'class': job['cls_name'], 
                        'lg': lg_score, 
                        'inliers': inliers, 
                        'dino_score': d_score
                    })

        # Finalize Results
        final_matches = []
        for i, m in enumerate(matches_per_query):
            if m is None: continue # Should not happen based on init logic but safe check
            
            best_res = m['best_res']
            is_dino_good = any(s >= self.cfg.dino_high_conf_threshold for _, _, s in m['candidates'])
            is_lg_good = (best_res['lg'] >= self.cfg.lg_norm_thresh and best_res['inliers'] >= self.cfg.lg_min_inliers)

            if best_res['score'] > 0 and (is_dino_good or is_lg_good):
                m.update({
                    'class': best_res['class'],
                    'score': best_res['score'],
                    'matched': True,
                    'details': best_res
                })
            # Remove temporary keys
            m.pop('candidates', None)
            m.pop('best_res', None)
            final_matches.append(m)

        timing_info['lightglue'] = time.time() - lg_start
        timing_info['total'] = time.time() - start_time
        
        return img, {'matches': final_matches, 'classes': [m['class'] for m in final_matches if m['matched']], 'timing': timing_info}
