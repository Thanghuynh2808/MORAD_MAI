import argparse
import sys
from pathlib import Path
import cv2
import torch
from types import SimpleNamespace

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retail_matcher.pipeline import ProductMatcher
from retail_matcher.utils.visualization import Visualizer
from retail_matcher.utils.common import logger
from retail_matcher.utils.config import load_config

def merge_config_with_args(config, args):
    """Override config file settings with command line arguments if they are explicitly provided."""
    # We check if the arg is different from the default in the parser to detect if user provided it
    # However, a simpler way since we already have a flat config is to just update it.
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="RPM Modified App")
    
    # Config File
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    
    # Paths (Defaults are now None to detect if user provided them)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--support_db", type=str, default=None)
    parser.add_argument("--yolo_path", type=str, default=None)
    
    # Parameters
    parser.add_argument("--yolo_conf", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--dino_thresh", type=float, default=None)
    parser.add_argument("--lg_norm_thresh", type=float, default=None)
    parser.add_argument("--lg_min_inliers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--dino_high_conf_threshold", type=float, default=None)
    parser.add_argument("--skip_lg_inliers_value", type=int, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load config from YAML
    config = load_config(args.config)
    
    # 2. Merge with CLI overrides
    config = merge_config_with_args(config, args)
    
    # Ensure dirs exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Matcher
    logger.info("Initializing ProductMatcher...")
    matcher = ProductMatcher(config)
    
    # Load Gallery
    if not matcher.load_gallery(config.support_db):
        logger.error("Failed to load gallery. Please run scripts/build_gallery.py first.")
        return

    # Process Images
    test_dir = Path(config.test_dir)
    images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    
    if not images:
        logger.warning(f"No images found in {test_dir}")
        return

    logger.info(f"Processing {len(images)} images...")
    
    for img_path in images:
        logger.info(f"Processing: {img_path.name}")
        
        # Run Pipeline
        original_img, results = matcher.process_image(img_path)
        
        if original_img is None:
            continue
            
        annotated_img = original_img.copy()
        matches = results.get('matches', [])
        
        # Visualization
        for m in matches:
            x1, y1, x2, y2 = m['box']
            obb_poly = m.get('obb')
            
            if m['matched']:
                annotated_img = Visualizer.draw_match(
                    annotated_img, int(x1), int(y1), int(x2), int(y2), 
                    m['details'], obb_poly
                )
            else:
                annotated_img = Visualizer.draw_unknown(
                    annotated_img, int(x1), int(y1), int(x2), int(y2), obb_poly
                )
                
        # Save Result
        out_path = Path(args.output_dir) / img_path.name
        cv2.imwrite(str(out_path), annotated_img)
        logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
