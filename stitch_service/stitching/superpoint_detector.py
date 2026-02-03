
import cv2 as cv
import numpy as np
import onnxruntime as ort
import os

class SuperPointDetector:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # Filter available providers to avoid warnings/errors
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
    def detect(self, img):
        # Preprocess
        # Ensure grayscale
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
            
        h, w = gray.shape
        img_tensor = gray.astype(np.float32) / 255.0
        img_tensor = img_tensor[None, None, :, :] # (1, 1, H, W)
        
        # Inference
        inputs = {self.session.get_inputs()[0].name: img_tensor}
        outputs = self.session.run(None, inputs)
        
        # Unpack outputs
        # outputs[0] = keypoints (B, N, 2)
        # outputs[1] = scores (B, N)
        # outputs[2] = descriptors (B, N, D) or similar.
        # Based on inspection:
        # 0: keypoints
        # 1: scores
        # 2: descriptors
        
        keypoints_raw = outputs[0][0] # (N, 2)
        scores_raw = outputs[1][0]    # (N,)
        descriptors_raw = outputs[2][0] # (N, 256)
        
        # Create cv.KeyPoints
        keypoints = []
        for i in range(len(keypoints_raw)):
            pt = keypoints_raw[i]
            score = scores_raw[i]
            # KeyPoint(x, y, size, angle, response, octave, class_id)
            kp = cv.KeyPoint(float(pt[0]), float(pt[1]), 1.0, -1, float(score), 0, -1)
            keypoints.append(kp)
            
        # Descriptors
        descriptors = descriptors_raw.astype(np.float32)
        
        return keypoints, descriptors

