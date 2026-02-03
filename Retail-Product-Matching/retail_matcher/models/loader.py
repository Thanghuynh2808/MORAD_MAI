import time
from pathlib import Path
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
import onnxruntime as ort
from retail_matcher.utils.common import logger

def load_model(config, base_weights_dir=None):
    """
    Load all models based on config.
    config should have: yolo_path, yolo_device, dino_device, lg_device
    """
    logger.info("Loading models...")
    start_time = time.time()
    
    yolo_path = config.yolo_path
    yolo_device = config.yolo_device
    dino_device = config.dino_device
    lg_device = config.lg_device

    # 1. YOLO
    try:
        yolo_model = YOLO(str(yolo_path))
        # Ultralytics handles device during inference or explicitly here
        yolo_model.to(yolo_device)
        logger.info(f"YOLO loaded successfully on {yolo_device}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model from {yolo_path}: {e}")
        raise

    # 2. DINOv3
    try:
        dinov3_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        dinov3_model = model.to(dino_device)
        dinov3_model.eval()
        logger.info(f"DINOv3 loaded successfully on {dino_device}")
    except Exception as e:
        logger.error(f"Failed to load DINOv3: {e}")
        raise

    # 3. ONNX Models (SuperPoint & LightGlue)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in lg_device else ['CPUExecutionProvider']
    
    if base_weights_dir:
        sp_path = Path(base_weights_dir) / "lightglue" / "superpoint_batch.onnx"
        lg_path = Path(base_weights_dir) / "lightglue" / "lightglue_batch.onnx"
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sp_path = project_root / "data" / "weights" / "lightglue" / "superpoint_batch.onnx"
        lg_path = project_root / "data" / "weights" / "lightglue" / "lightglue_batch.onnx"

    try:
        if not sp_path.exists():
            raise FileNotFoundError(f"SuperPoint ONNX model not found at {sp_path}")
        if not lg_path.exists():
            raise FileNotFoundError(f"LightGlue ONNX model not found at {lg_path}")

        sp_session = ort.InferenceSession(str(sp_path), providers=providers)
        lg_session = ort.InferenceSession(str(lg_path), providers=providers)
        logger.info(f"ONNX Models loaded on {sp_session.get_providers()[0]}")
    except Exception as e:
        logger.error(f"Error loading ONNX models: {e}")
        raise

    logger.info(f"Total loading time: {time.time() - start_time:.2f}s")
    
    # We return the devices in a dict for pipeline to use
    devices = {
        'yolo': yolo_device,
        'dino': dino_device,
        'lg': lg_device
    }
    
    return yolo_model, dinov3_processor, dinov3_model, sp_session, lg_session, devices
