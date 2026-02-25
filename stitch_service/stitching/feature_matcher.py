import math
from pathlib import Path

import cv2 as cv
import numpy as np


class FeatureMatcher:
    """https://docs.opencv.org/4.x/da/d87/classcv_1_1detail_1_1FeaturesMatcher.html"""

    MATCHER_CHOICES = ("homography", "affine", "lightglue")
    DEFAULT_MATCHER = "homography"
    DEFAULT_RANGE_WIDTH = -1

    def __init__(
        self, matcher_type=DEFAULT_MATCHER, range_width=DEFAULT_RANGE_WIDTH, **kwargs
    ):
        self.matcher_type = matcher_type
        if matcher_type == "lightglue":
            from .lightglue_matcher import LightGlueMatcher
            # Default path: Retail-Product-Matching/data/weights/lightglue/lightglue_batch.onnx
            default_path = Path(__file__).parent.parent.parent / "Retail-Product-Matching" / "data" / "weights" / "lightglue" / "lightglue_batch.onnx"
            model_path = kwargs.get("model_path", str(default_path))
            self.matcher = LightGlueMatcher(model_path=model_path)
        elif matcher_type == "affine":
            self.matcher = cv.detail_AffineBestOf2NearestMatcher(**kwargs)
        elif range_width == -1:
            self.matcher = cv.detail_BestOf2NearestMatcher(**kwargs)
        else:
            self.matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, **kwargs)

    def match_features(self, features, *args, **kwargs):
        if self.matcher_type == "lightglue":
            return self._match_lightglue(features, *args, **kwargs)
        
        pairwise_matches = self.matcher.apply2(features, *args, **kwargs)
        self.matcher.collectGarbage()
        return pairwise_matches

    def _match_lightglue(self, features, *args, **kwargs):
        num_imgs = len(features)
        pairwise_matches = []
        
        # We need to construct the full pairwise grid (num_imgs x num_imgs)
        # Order expected by stitching pipeline: row-major flattened?
        # Actually cv.detail.MatchesInfo structs.
        # usually 0->0, 0->1, ... 0->N, 1->0, ...
        
        for i in range(num_imgs):
            for j in range(num_imgs):
                matches_info = cv.detail.MatchesInfo()
                matches_info.src_img_idx = i
                matches_info.dst_img_idx = j
                
                if i == j:
                    matches_info.confidence = 1.0 # Self match
                    # matches_info.matches = ... # Can leave empty or fake identity matches if needed, but usually redundant for pipeline usually ignores i==j
                else:
                    # Run LightGlue
                    matches_list, confs = self.matcher.match(features[i], features[j])
                    matches_info.matches = tuple(matches_list) # Must be tuple/list of DMatch
                    
                    if len(matches_list) < 20: # Arbitrary threshold
                        matches_info.confidence = 0.0
                    else:
                        # Confidence usually: sum of scores / total? or number of inliers / (8+0.3*matches)?
                        # Standard OpenCV confidence: ratio of inliers.
                        # LightGlue confidences are 0-1. 
                        # We can sum them or take mean?
                        # Or a simple robust metric.
                        matches_info.confidence = float(np.mean(confs) * 2.0) if len(confs) > 0 else 0.0
                        # matches_info.confidence = float(len(matches_list)) / 100.0 # Heuristic
                        
                    # matches_info.H = ... # Homography?
                    # Pipeline usually estimates H later using these matches. 
                    # But MatchesInfo usually has H if we use AffineBestOf2NearestMatcher?
                    # Actually BestOf2NearestMatcher does NOT compute H. It just finds matches.
                    # Estimator step computes H.
                    
                pairwise_matches.append(matches_info)
                
        return pairwise_matches

    @staticmethod
    def draw_matches_matrix(
        imgs, features, matches, conf_thresh=1, inliers=False, **kwargs
    ):
        matches_matrix = FeatureMatcher.get_matches_matrix(matches)
        for idx1, idx2 in FeatureMatcher.get_all_img_combinations(len(imgs)):
            match = matches_matrix[idx1, idx2]
            if match.confidence < conf_thresh or len(match.matches) == 0:
                continue
            if inliers:
                kwargs["matchesMask"] = match.getInliers()
            yield idx1, idx2, FeatureMatcher.draw_matches(
                imgs[idx1], features[idx1], imgs[idx2], features[idx2], match, **kwargs
            )

    @staticmethod
    def draw_matches(img1, features1, img2, features2, match1to2, **kwargs):
        kwargs.setdefault("flags", cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        keypoints1 = features1.getKeypoints()
        keypoints2 = features2.getKeypoints()
        matches = match1to2.getMatches()

        return cv.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None, **kwargs
        )

    @staticmethod
    def get_matches_matrix(pairwise_matches):
        return FeatureMatcher.array_in_square_matrix(pairwise_matches)

    @staticmethod
    def get_confidence_matrix(pairwise_matches):
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        match_confs = [[m.confidence for m in row] for row in matches_matrix]
        match_conf_matrix = np.array(match_confs)
        return match_conf_matrix

    @staticmethod
    def array_in_square_matrix(array):
        matrix_dimension = int(math.sqrt(len(array)))
        rows = []
        for i in range(0, len(array), matrix_dimension):
            rows.append(array[i : i + matrix_dimension])
        return np.array(rows)

    def get_all_img_combinations(number_imgs):
        ii, jj = np.triu_indices(number_imgs, k=1)
        for i, j in zip(ii, jj):
            yield i, j

    @staticmethod
    def get_match_conf(match_conf, feature_detector_type):
        if match_conf is None:
            match_conf = FeatureMatcher.get_default_match_conf(feature_detector_type)
        return match_conf

    @staticmethod
    def get_default_match_conf(feature_detector_type):
        if feature_detector_type == "orb":
            return 0.3
        return 0.65
