from collections import OrderedDict

import cv2 as cv
import numpy as np

from .stitching_error import StitchingError


class FeatureDetector:
    """https://docs.opencv.org/4.x/d0/d13/classcv_1_1Feature2D.html"""

    DETECTOR_CHOICES = OrderedDict()

    DETECTOR_CHOICES["orb"] = cv.ORB.create
    DETECTOR_CHOICES["sift"] = cv.SIFT_create
    DETECTOR_CHOICES["brisk"] = cv.BRISK_create
    DETECTOR_CHOICES["akaze"] = cv.AKAZE_create
    DETECTOR_CHOICES["superpoint"] = None # Handled specially

    DEFAULT_DETECTOR = list(DETECTOR_CHOICES.keys())[0]

    def __init__(self, detector=DEFAULT_DETECTOR, **kwargs):
        self.detector_name = detector
        if detector == "superpoint":
            from .superpoint_detector import SuperPointDetector
            # Default path if not provided
            model_path = kwargs.get("model_path", "/home/thanghuynh/work/project/fpt/image_stitching/weight_superpoint/superpoint_batch.onnx")
            self.detector = SuperPointDetector(model_path=model_path)
        else:
            self.detector = FeatureDetector.DETECTOR_CHOICES[detector](**kwargs)

    def detect_features(self, img, *args, **kwargs):
        if self.detector_name == "superpoint":
            return self._detect_superpoint(img, *args, **kwargs)
        # Remove img_idx from kwargs as computeImageFeatures2 doesn't accept it as kwarg
        kwargs.pop('img_idx', None)
        return cv.detail.computeImageFeatures2(self.detector, img, *args, **kwargs)

    def _detect_superpoint(self, img, mask=None, **kwargs):
        keypoints, descriptors = self.detector.detect(img)
        
        if mask is not None:
             # Filter keypoints by mask
             valid_indices = []
             filtered_kps = []
             for i, kp in enumerate(keypoints):
                 x, y = int(kp.pt[0]), int(kp.pt[1])
                 if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                     if mask[y, x] > 0:
                         filtered_kps.append(kp)
                         valid_indices.append(i)
             keypoints = filtered_kps
             if len(valid_indices) > 0:
                 descriptors = descriptors[valid_indices]
             else:
                 descriptors = np.empty((0, descriptors.shape[1]), dtype=np.float32)

        # Create dummy features to get the structure
        # Use ORB simple detection on a tiny image to be fast
        dummy_orb = cv.ORB_create(nfeatures=1)
        dummy_img = np.zeros((10, 10), dtype=np.uint8)
        features = cv.detail.computeImageFeatures2(dummy_orb, dummy_img)
        
        # Populate with SuperPoint data
        features.img_idx = kwargs.get("img_idx", 0) # Usually not set here but handled by caller? Actually caller doesn't set it. computeImageFeatures2 sets it to 0.
        features.img_size = (img.shape[1], img.shape[0])
        features.keypoints = keypoints
        features.descriptors = cv.UMat(descriptors)
        
        return features

    def detect(self, imgs):
        return [self.detect_features(img, img_idx=i) for i, img in enumerate(imgs)]

    def detect_with_masks(self, imgs, masks):
        features = []
        for idx, (img, mask) in enumerate(zip(imgs, masks)):
            assert len(img.shape) == 3 and len(mask.shape) == 2
            if not len(imgs) == len(masks):
                raise StitchingError("image and mask lists must be of same length")
            if not np.array_equal(img.shape[:2], mask.shape):
                raise StitchingError(
                    f"Resolution of mask {idx + 1} {mask.shape} does not match"
                    f" the resolution of image {idx + 1} {img.shape[:2]}."
                )
            features.append(self.detect_features(img, mask=mask))
        return features

    @staticmethod
    def draw_keypoints(img, features, **kwargs):
        kwargs.setdefault("color", (0, 255, 0))
        keypoints = features.getKeypoints()
        return cv.drawKeypoints(img, keypoints, None, **kwargs)
