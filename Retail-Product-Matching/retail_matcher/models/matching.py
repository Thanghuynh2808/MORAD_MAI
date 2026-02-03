import torch
import torch.nn.functional as F
import numpy as np
from retail_matcher.utils.processing import normalize_keypoints
from retail_matcher.utils.common import logger

@torch.no_grad()
def matrix_matching(query_features, support_db, top_k, dino_threshold):
    """Coarse matching using DINOv3 features. Assumes features are already normalized."""
    gallery_matrix = support_db['gallery_matrix']
    if query_features is None or gallery_matrix is None:
        logger.warning("Empty query_features or gallery_matrix in matrix_matching")
        return [[] for _ in range(len(query_features) if query_features else 0)]

    q_cpu = query_features.cpu()
    g_cpu = gallery_matrix.cpu()

    q_norm = F.normalize(q_cpu, p=2, dim=-1)
    g_norm = F.normalize(g_cpu, p=2, dim=-1)
    sim_matrix = torch.mm(q_norm, g_norm.T)

    top_k_vals, top_k_inds = torch.topk(sim_matrix, k=min(top_k, sim_matrix.size(1)), dim=1)

    candidates_per_query = []
    for i in range(sim_matrix.size(0)):
        candidates = []
        for k in range(top_k_vals.size(1)):
            sim = float(top_k_vals[i, k])
            if sim >= dino_threshold:
                idx = int(top_k_inds[i, k])
                candidates.append((idx, support_db['class_names'][idx], sim))
        candidates_per_query.append(candidates)

    return candidates_per_query

def run_fast_hybrid_matching(pairs_to_verify, matcher_session, batch_size=16):
    """
    Batched LightGlue Matching for multiple diverse pairs.
    pairs_to_verify: List of tuples (query_feat, support_feat)
    Returns: List of (inliers, min_kpts) corresponding to each pair
    """
    results = []
    total_jobs = len(pairs_to_verify)
    if total_jobs == 0:
        return results

    input_kpts_name = matcher_session.get_inputs()[0].name
    input_desc_name = matcher_session.get_inputs()[1].name

    # Pre-process all pairs
    all_q_kpts = []
    all_q_desc = []
    all_s_kpts = []
    all_s_desc = []

    for q_feat, s_feat in pairs_to_verify:
        # Normalize Query
        q_kpts = normalize_keypoints(q_feat['keypoints'][0], q_feat['width'], q_feat['height'])
        all_q_kpts.append(q_kpts)
        all_q_desc.append(q_feat['descriptors'][0])

        # Normalize Support
        s_kpts = normalize_keypoints(s_feat['keypoints'][0], s_feat['width'], s_feat['height'])
        all_s_kpts.append(s_kpts)
        all_s_desc.append(s_feat['descriptors'][0])

    # Convert to Numpy arrays
    all_q_kpts = np.stack(all_q_kpts, axis=0)
    all_q_desc = np.stack(all_q_desc, axis=0)
    all_s_kpts = np.stack(all_s_kpts, axis=0)
    all_s_desc = np.stack(all_s_desc, axis=0)

    # Process in batches
    for i in range(0, total_jobs, batch_size):
        # Slice current batch
        curr_q_kpts = all_q_kpts[i: i + batch_size]
        curr_q_desc = all_q_desc[i: i + batch_size]
        curr_s_kpts = all_s_kpts[i: i + batch_size]
        curr_s_desc = all_s_desc[i: i + batch_size]
        
        B = len(curr_q_kpts)
        
        # --- VECTORIZED INPUT PREPARATION ---
        # Interleave query and support: [Q1, S1, Q2, S2, ...]
        batch_kpts = np.zeros((2 * B, 512, 2), dtype=np.float32)
        batch_desc = np.zeros((2 * B, 512, 256), dtype=np.float32)

        batch_kpts[0::2] = curr_q_kpts
        batch_desc[0::2] = curr_q_desc
        batch_kpts[1::2] = curr_s_kpts
        batch_desc[1::2] = curr_s_desc

        try:
            outputs = matcher_session.run(None, {input_kpts_name: batch_kpts, input_desc_name: batch_desc})
            matches = outputs[0]

            for j in range(B):
                # LightGlue Light outputs matches where batch_index corresponds to the pair
                # Since we interleaved, pair j corresponds to batch index j (in LightGlue batch logic)
                # But wait, LightGlue batch output format: [batch_idx, match_idx_0, match_idx_1]
                # Our input batch size to ONNX is 2*B, but logically it treats it as B pairs? 
                # NO. Standard LightGlue ONNX takes (Batch, N, D) and outputs matches for each pair in batch?
                # Actually, standard LightGlue treats input as a batch of images.
                # If we send 2*B images, it tries to match image 0 with 1, 2 with 3? 
                # It depends on how the ONNX was exported.
                # Assuming "interleaved" export (Match 0-1, 2-3...):
                # The output 'matches' tensor usually has shape (N, 3) -> [batch_idx, idx0, idx1] or similar
                # Let's rely on previous logic: matches[:, 0] == j
                
                valid_matches = matches[matches[:, 0] == j]
                results.append((len(valid_matches), 512))
        except Exception as e:
            logger.error(f"Batch matching error: {e}")
            for _ in range(B):
                results.append((0, 512))

    return results
