import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from retail_matcher.models.loader import load_model
from retail_matcher.utils.common import load_support_images, logger
from retail_matcher.models.extraction import aggregate_support_features

def build_feature_bank():
    # Setup paths
    base_dir = PROJECT_ROOT
    yolo_path = base_dir / "data" / "weights" / "yolo" / "best-obb.pt"
    support_dir = base_dir / "data" / "support_images"
    output_path = base_dir / "data" / "support_db.pt"

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device_str}")
    
    if not support_dir.exists():
        logger.error(f"Support directory not found at {support_dir}")
        return

    # Load models
    try:
        models = load_model(str(yolo_path), device_str, base_weights_dir=base_dir/"data"/"weights")
        _, dinov3_processor, dinov3_model, sp_session, _, device = models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    logger.info("Loading support images...")
    support_images = load_support_images(support_dir)
    
    if not support_images:
        logger.error("No images found in support directory!")
        return

    logger.info(f"Creating feature bank for {len(support_images)} images...")
    
    support_db = aggregate_support_features(support_images, dinov3_model, dinov3_processor, sp_session, device)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(support_db, str(output_path))
    logger.info(f"Feature bank created successfully and saved to {output_path.absolute()}")

if __name__ == "__main__":
    build_feature_bank()
