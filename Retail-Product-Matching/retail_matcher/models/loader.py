import time
from pathlib import Path
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
import onnxruntime as ort
from retail_matcher.utils.common import logger

def load_model(yolo_path, device_str, base_weights_dir=None):
    logger.info("Loading models...")
    start_time = time.time()
    device = torch.device(device_str)

    # 1. YOLO
    try:
        yolo_model = YOLO(str(yolo_path))
        logger.info("YOLO loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model from {yolo_path}: {e}")
        raise

    # 2. DINOv3
    try:
        dinov3_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        dinov3_model = model.to(device)
        dinov3_model.eval()
        logger.info("DINOv3 loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load DINOv3: {e}")
        raise

    # 3. ONNX Models (SuperPoint & LightGlue)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device_str else ['CPUExecutionProvider']
    
    # Adjust path strategy: look in provided base_weights_dir or default relative path
    if base_weights_dir:
        sp_path = Path(base_weights_dir) / "lightglue" / "superpoint_batch.onnx"
        lg_path = Path(base_weights_dir) / "lightglue" / "lightglue_batch.onnx"
    else:
        # Fallback to assuming this file is in retail_matcher/models/loader.py
        # and weights are in data/weights/lightglue relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        sp_path = project_root / "data" / "weights" / "lightglue" / "superpoint_batch.onnx"
        lg_path = project_root / "data" / "weights" / "lightglue" / "lightglue_batch.onnx"

    try:
        if not sp_path.exists():
            raise FileNotFoundError(f"SuperPoint ONNX model not found at {sp_path}")
        if not lg_path.exists():
            raise FileNotFoundError(f"LightGlue ONNX model not found at {lg_path}")

        # Try loading with the requested providers
        try:
            sp_session = ort.InferenceSession(str(sp_path), providers=providers)
            lg_session = ort.InferenceSession(str(lg_path), providers=providers)
            logger.info(f"ONNX Models loaded on {sp_session.get_providers()[0]}")
        except Exception as e:
            if 'CUDAExecutionProvider' in providers:
                logger.warning(f"Failed to load ONNX on GPU ({e}). Falling back to CPU...")
                sp_session = ort.InferenceSession(str(sp_path), providers=['CPUExecutionProvider'])
                lg_session = ort.InferenceSession(str(lg_path), providers=['CPUExecutionProvider'])
                logger.info("ONNX Models loaded on CPU fallback")
            else:
                raise e
    except Exception as e:
        logger.error(f"Error loading ONNX models: {e}")
        raise

    logger.info(f"Total loading time: {time.time() - start_time:.2f}s")
    return yolo_model, dinov3_processor, dinov3_model, sp_session, lg_session, device
