import logging
import hashlib
import colorsys
import glob
from pathlib import Path
from typing import List, Dict
import cv2


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("retail_match")
    return logger

# create module-level logger
logger = setup_logging()

def get_class_color(class_name: str):
    """Generates a consistent BGR color for a given class name."""
    hash_value = int(hashlib.md5(class_name.encode('utf-8')).hexdigest(), 16)
    h = (hash_value % 100) / 100.0
    s = 0.8
    v = 0.9
    rgb = colorsys.hsv_to_rgb(h, s, v)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

def load_support_images(support_dir: Path) -> List[Dict]:
    """Loads support images from directories."""
    support_images = []
    # Using pathlib glob directly might be cleaner but keeping original logic
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in glob.glob(str(support_dir / '**' / ext), recursive=True):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    class_name = Path(img_path).parent.name
                    support_images.append({'image': img, 'class_name': class_name, 'path': img_path})
                    logger.debug(f"Loaded: {class_name} - {Path(img_path).name}")
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")
    return support_images
