import yaml
from pathlib import Path
from types import SimpleNamespace
import torch
from retail_matcher.utils.common import logger

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    If config_path is None, look for default in configs/settings.yaml
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    
    if config_path is None:
        config_path = project_root / "configs" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using hardcoded defaults.")
        return get_default_config(project_root)

    try:
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Flatten the nested dictionary for easier access as attributes
        flat_cfg = {}
        for section, values in cfg_dict.items():
            for key, value in values.items():
                if section == 'paths':
                    flat_cfg[key] = str(project_root / value)
                elif section == 'devices':
                    flat_cfg[f"{key}_device"] = value
                else:
                    flat_cfg[key] = value
        
        return SimpleNamespace(**flat_cfg)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return get_default_config(project_root)

def get_default_config(project_root):
    """Fallback defaults if config file is missing or broken"""
    return SimpleNamespace(
        test_dir=str(project_root / "data" / "test_images"),
        support_dir=str(project_root / "data" / "support_images"),
        output_dir=str(project_root / "data" / "result_images"),
        support_db=str(project_root / "data" / "support_db.pt"),
        yolo_path=str(project_root / "data" / "weights" / "yolo" / "best-obb.pt"),
        yolo_conf=0.25,
        top_k=5,
        dino_thresh=0.65,
        lg_norm_thresh=0.2,
        lg_min_inliers=30,
        batch_size=32,
        alpha=0.2,
        beta=0.8,
        dino_high_conf_threshold=0.8,
        skip_lg_inliers_value=999,
        # Default devices
        yolo_device="cuda" if torch.cuda.is_available() else "cpu",
        dino_device="cpu", # Default to CPU for safe demo
        lg_device="cpu"    # Default to CPU for safe demo
    )
