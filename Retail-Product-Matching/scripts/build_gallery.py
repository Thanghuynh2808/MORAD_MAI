import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from retail_matcher.models.loader import load_model
from retail_matcher.utils.common import load_support_images, logger
from retail_matcher.models.extraction import aggregate_support_features

from retail_matcher.utils.config import load_config

def build_feature_bank():
    # 1. Load config
    config = load_config()
    
    # 2. Setup Device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device_str}")
    
    # 3. Check paths
    support_dir = Path(config.support_dir)
    if not support_dir.exists():
        logger.error(f"Support directory not found at {support_dir}")
        return

    # 4. Load models
    try:
        models = load_model(config.yolo_path, device_str)
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
    
    output_path = Path(config.support_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(support_db, str(output_path))
    logger.info(f"Feature bank created successfully and saved to {output_path.absolute()}")

if __name__ == "__main__":
    build_feature_bank()
