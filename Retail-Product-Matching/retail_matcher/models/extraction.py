import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import List
import time
from retail_matcher.utils.common import logger
from retail_matcher.utils.processing import apply_clahe, preprocess_image_for_onnx

@torch.no_grad()
def extract_fetures(image, processor, model, device="cuda"):
    """Extract single DINOv3 feature."""
    try:
        if isinstance(image, np.ndarray):
            if image.size == 0:
                logger.warning("Empty image array provided to extract_fetures")
                return None
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(**inputs, output_hidden_states=True)
        return output.last_hidden_state[:, 0, :].squeeze(0)
    except Exception as e:
        logger.error(f"Error extracting DINO features: {e}")
        return None


@torch.no_grad()
def batch_extract_dino_features(crops: List[np.ndarray], processor, model, device, batch_size=128):
    """Extract DINOv3 features for a batch of crops."""
    if not crops:
        logger.warning("Empty crops list provided to batch_extract_dino_features")
        return None

    pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops if c.size > 0]
    if not pil_images:
        logger.warning("No valid PIL images after conversion")
        return None

    all_features = []
    try:
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(**inputs, output_hidden_states=True)

            features = output.last_hidden_state[:, 0, :]
            # Normalize and move to CPU to save VRAM during collection
            norm_features = F.normalize(features, p=2, dim=-1).cpu()
            all_features.append(norm_features)
            
            # Cleanup
            del output, inputs, features
            if device == 'cuda':
                torch.cuda.empty_cache()

        return torch.cat(all_features, dim=0) if all_features else None
    except Exception as e:
        logger.error(f"Error in batch_extract_dino_features: {e}")
        if device == 'cuda':
            torch.cuda.empty_cache()
        return None


@torch.no_grad()
def aggregate_support_features(support_images, dinov3_model, dinov3_processor, sp_session, device, batch_size=50):
    """Builds the Feature Bank."""
    start_time = time.time()
    support_dino = []
    support_sp = []
    support_names = []
    sp_input_name = sp_session.get_inputs()[0].name

    total = len(support_images)
    for idx, img_info in enumerate(support_images):
        if idx % batch_size == 0 and idx > 0:
            logger.info(f"Processed {idx}/{total} images... ({idx/total*100:.1f}%)")
            torch.cuda.empty_cache()
        
        image_clahe = apply_clahe(img_info['image'])

        # DINO Extraction
        dino_feat = extract_fetures(image_clahe, dinov3_processor, dinov3_model, device)

        # SuperPoint Extraction
        img_tensor, nw, nh = preprocess_image_for_onnx(image_clahe)
        sp_feat = None
        if img_tensor is not None:
            try:
                outs = sp_session.run(None, {sp_input_name: img_tensor})
                if len(outs) < 3:
                    logger.warning(f"SuperPoint output has {len(outs)} elements, expected 3")
                elif outs[0].shape[0] == 0 or outs[2].shape[0] == 0:
                    logger.warning(f"Empty keypoints or descriptors for image {img_info.get('path', 'unknown')}")
                else:
                    sp_feat = {
                        'keypoints': outs[0], 'scores': outs[1], 'descriptors': outs[2],
                        'width': nw, 'height': nh
                    }
            except Exception as e:
                logger.warning(f"SuperPoint extraction failed for {img_info.get('path', 'unknown')}: {e}")

        if dino_feat is not None and sp_feat is not None:
            support_dino.append(dino_feat)
            support_sp.append(sp_feat)
            support_names.append(img_info['class_name'])

    gallery_matrix = torch.stack(support_dino, dim=0) if support_dino else None

    logger.info(f"Gallery built: {len(support_names)} images. Time: {time.time() - start_time:.2f}s")
    return {'gallery_matrix': gallery_matrix, 'sp_features': support_sp, 'class_names': support_names}
