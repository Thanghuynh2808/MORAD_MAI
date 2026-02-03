import argparse
import sys
from pathlib import Path
import cv2
import torch
from types import SimpleNamespace

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Custom imports
from retail_matcher.pipeline import ProductMatcher
from retail_matcher.utils.visualization import Visualizer
from retail_matcher.utils.common import logger

def parse_args():
    parser = argparse.ArgumentParser(description="RPM Modified App")
    
    # Paths
    parser.add_argument("--test_dir", type=str, default=str(PROJECT_ROOT / "data" / "test_images"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "data" / "result_images"))
    parser.add_argument("--support_db", type=str, default=str(PROJECT_ROOT / "data" / "support_db.pt"))
    parser.add_argument("--yolo_path", type=str, default=str(PROJECT_ROOT / "data" / "weights" / "yolo" / "best-obb.pt"))
    
    # Parameters
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dino_thresh", type=float, default=0.65)
    parser.add_argument("--lg_norm_thresh", type=float, default=0.2)
    parser.add_argument("--lg_min_inliers", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--dino_high_conf_threshold", type=float, default=0.8)
    parser.add_argument("--skip_lg_inliers_value", type=int, default=999)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure dirs exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Matcher
    logger.info("Initializing ProductMatcher...")
    matcher = ProductMatcher(args)
    
    # Load Gallery
    if not matcher.load_gallery(args.support_db):
        logger.error("Failed to load gallery. Please run scripts/build_gallery.py first.")
        return

    # Process Images
    test_dir = Path(args.test_dir)
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
