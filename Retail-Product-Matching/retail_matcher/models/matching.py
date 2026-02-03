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

def run_fast_hybrid_matching(query_feat, support_feats_list, matcher_session, batch_size=16):
    results = []
    total_jobs = len(support_feats_list)
    if total_jobs == 0:
        return results

    input_kpts_name = matcher_session.get_inputs()[0].name
    input_desc_name = matcher_session.get_inputs()[1].name

    # Normalize Query Keypoints
    q_kpts_norm = normalize_keypoints(query_feat['keypoints'][0], query_feat['width'], query_feat['height'])
    q_desc = query_feat['descriptors'][0]

    s_kpts_norm_list = []
    s_desc_list = []

    for s_feat in support_feats_list:
        s_kpts_norm = normalize_keypoints(s_feat['keypoints'][0], s_feat['width'], s_feat['height'])
        s_kpts_norm_list.append(s_kpts_norm)
        s_desc_list.append(s_feat['descriptors'][0])

    all_s_kpts = np.stack(s_kpts_norm_list, axis=0)  # (N, 1024, 2)
    all_s_desc = np.stack(s_desc_list, axis=0)      # (N, 1024, 256)

    for i in range(0, total_jobs, batch_size):
        current_s_kpts = all_s_kpts[i: i + batch_size]
        current_s_desc = all_s_desc[i: i + batch_size]
        B = len(current_s_kpts)

        # --- VECTORIZED INPUT PREPARATION ---
        batch_kpts = np.zeros((2 * B, 512, 2), dtype=np.float32)
        batch_desc = np.zeros((2 * B, 512, 256), dtype=np.float32)

        batch_kpts[0::2] = q_kpts_norm
        batch_desc[0::2] = q_desc

        batch_kpts[1::2] = current_s_kpts
        batch_desc[1::2] = current_s_desc

        try:
            outputs = matcher_session.run(None, {input_kpts_name: batch_kpts, input_desc_name: batch_desc})
            matches = outputs[0]

            for j in range(B):
                valid_matches = matches[matches[:, 0] == j]
                results.append((len(valid_matches), 512))
        except Exception as e:
            logger.error(f"Batch matching error: {e}")
            for _ in range(B):
                results.append((0, 512))

    return results
