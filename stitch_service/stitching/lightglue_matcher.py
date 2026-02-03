
import numpy as np
import onnxruntime as ort
import os
import cv2 as cv

class LightGlueMatcher:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
    def match(self, features1, features2):
        # features1/2 are ImageFeatures or dict-like objects containing keypoints and descriptors
        
        kps1 = self._get_keypoints(features1)
        kps2 = self._get_keypoints(features2)
        desc1 = self._get_descriptors(features1)
        desc2 = self._get_descriptors(features2)
        
        print(f"[LightGlue] Match input: {len(kps1)} kps (img1) x {len(kps2)} kps (img2)")
        
        if len(kps1) == 0 or len(kps2) == 0:
            return np.array([]), np.array([])
        
        kps1_norm = self._normalize_keypoints(kps1, features1.img_size)
        kps2_norm = self._normalize_keypoints(kps2, features2.img_size)
        
        # Pad to max length
        max_kps = max(len(kps1), len(kps2))
        
        # Create batch inputs
        # Shape: [2, max_kps, 2]
        kps_tensor = np.zeros((2, max_kps, 2), dtype=np.float32)
        kps_tensor[0, :len(kps1), :] = kps1_norm
        kps_tensor[1, :len(kps2), :] = kps2_norm
        
        # Shape: [2, max_kps, 256]
        desc_tensor = np.zeros((2, max_kps, 256), dtype=np.float32)
        desc_tensor[0, :len(desc1), :] = desc1
        desc_tensor[1, :len(desc2), :] = desc2
        
        # Run inference
        try:
            inputs = {
                self.session.get_inputs()[0].name: kps_tensor,
                self.session.get_inputs()[1].name: desc_tensor
            }
            outputs = self.session.run(None, inputs)
        except Exception as e:
            print(f"[LightGlue] Inference failed: {e}")
            return [], []
        
        matches_tensor = outputs[0]
        scores_tensor = outputs[1]
        
        matches_list = []
        confidences_list = []
        
        for i in range(len(matches_tensor)):
            b_idx, idx0, idx1 = matches_tensor[i]
            # Since we assume batch of 2 is the pair, we only care about "matches within the batch"? 
            # Actually standard LightGlue with batch>1 treats them as independent batches, or cross?
            # If shape is [B, N, D], matches are usually [match_index, batch_index, idx0, idx1]?
            # Wait. "Name: matches, Shape: ['num_matches', 3]"
            # Typically 3 columns: [batch_index, match_index_in_src, match_index_in_dst]
            
            # Since we constructed a batch of 2 images: [img1, img2]...
            # LightGlue ONNX usually takes 2 images as inputs (image0, image1).
            # But the input names were "keypoints" and "descriptors" with shape [batch_size_x2, ...].
            # This implies the model expects a stacked batch where the first half is set A and second half is set B?
            # OR it treats odd/even as pairs? 
            # If we fed shape [2, max_kps, ...], does it treat 0 and 1 as a pair?
            
            # If the model is exported with "batch_size_x2", it likely means it takes pairs.
            # If we assume it matched 0 against 1:
            # batch_idx should be 0? Or maybe it allows multiple pairs?
            
            # Indices check
            if idx0 < len(kps1) and idx1 < len(kps2):
                 match = cv.DMatch()
                 match.queryIdx = int(idx0)
                 match.trainIdx = int(idx1)
                 match.imgIdx = 0 
                 
                 score = float(scores_tensor[i])
                 match.distance = 1.0 - score 
                 
                 matches_list.append(match)
                 confidences_list.append(score)
        
        print(f"[LightGlue] Found {len(matches_list)} matches. Avg Conf: {np.mean(confidences_list) if confidences_list else 0:.4f}")
        return matches_list, confidences_list

    def _get_keypoints(self, features):
        if hasattr(features, 'keypoints'):
             # cv.detail.ImageFeatures has keypoints as list of cv.KeyPoint
             return np.array([kp.pt for kp in features.keypoints], dtype=np.float32)
        return np.array([])
        
    def _get_descriptors(self, features):
        if hasattr(features, 'descriptors'):
             desc = features.descriptors
             if isinstance(desc, cv.UMat):
                 desc = desc.get()
             return desc.astype(np.float32)
        return np.array([])
        
    def _normalize_keypoints(self, kps, img_size):
        if len(kps) == 0:
            return kps
            
        w, h = img_size
        size = np.array([w, h], dtype=np.float32)
        
        # Standard LightGlue normalization: 2 * (x / size) - 1 ??
        # Or (x - size/2) / (size/2) ??
        # 2*x/w - 1 maps [0, w] to [-1, 1]
        
        kps_norm = (kps / size) * 2.0 - 1.0
        return kps_norm

