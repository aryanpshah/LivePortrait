"""
FaceMesh Expression Assist - Phase 2: Full Implementation

Provides FaceMesh-based mouth driving signal projected into LivePortrait's
exp_delta keypoint space. This adds a correction term to improve mouth
asymmetry transfer without modifying LivePortrait model internals.

Key pipeline:
1. Compute donor asymmetry delta using existing FaceMesh infrastructure
2. Align delta vectors from donor frame to target frame (Umeyama similarity)
3. Project mouth deltas to LP keypoints via KNN interpolation
4. Convert pixel displacements to exp units
5. Return exp_delta_fm to be added to exp_delta

Author: Generated for liveportrait_asymmetry
Date: 2026-01-18
"""

import os
import json
import warnings
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# ---------------------------------------------------------------------------
# Configuration and Constants
# ---------------------------------------------------------------------------

FACEMESH_EXP_ASSIST_DEBUG_DIR = "outputs/diagnostics/facemesh_exp_assist"

# MediaPipe FaceMesh mouth-related landmark indices
# Outer lips ring
FACEMESH_OUTER_LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
]
# Inner lips ring
FACEMESH_INNER_LIPS_IDX = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]
# Lip corners (left=61, right=291)
FACEMESH_LIP_CORNERS_IDX = [61, 291]
# Combined mouth landmarks
FACEMESH_MOUTH_ALL_IDX = list(set(FACEMESH_OUTER_LIPS_IDX + FACEMESH_INNER_LIPS_IDX))

# Anchor indices for similarity transform (forehead, chin, left cheek, right cheek)
FACEMESH_ANCHOR_IDX = [10, 152, 234, 454]


# ---------------------------------------------------------------------------
# Umeyama Similarity Transform (no SciPy)
# ---------------------------------------------------------------------------

def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Umeyama similarity transform: dst = s * R @ src + t
    
    Deterministic, no randomness. Uses SVD for rotation.
    
    Args:
        src: Source points (N, 2)
        dst: Destination points (N, 2)
    
    Returns:
        scale (float), R (2x2 rotation matrix), t (2,) translation vector
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    
    # Centroids
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    
    # Centered coordinates
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    
    # Variance of source
    src_var = np.sum(src_centered ** 2) / n
    if src_var < 1e-10:
        # Degenerate case: all points at same location
        return 1.0, np.eye(2, dtype=np.float32), (dst_mean - src_mean).astype(np.float32)
    
    # Covariance matrix
    cov = (dst_centered.T @ src_centered) / n
    
    # SVD
    U, S, Vt = np.linalg.svd(cov)
    
    # Handle reflection
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.array([[1, 0], [0, d]], dtype=np.float64)
    
    # Rotation
    R = U @ D @ Vt
    
    # Scale
    scale = np.sum(S * np.diag(D)) / src_var
    
    # Translation
    t = dst_mean - scale * (R @ src_mean)
    
    return float(scale), R.astype(np.float32), t.astype(np.float32)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def compute_facemesh_donor_delta_px(
    donor_rgb: np.ndarray,
    extractor: Any,
    regions_config: Optional[Dict] = None,
    *,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Compute donor asymmetry delta in pixel space using FaceMesh.
    
    Reuses the existing compute_donor_asymmetry_delta from facemesh_landmarks.py.
    
    Args:
        donor_rgb: Donor image as HxWx3 RGB numpy array.
        extractor: FaceMeshLandmarkExtractor instance.
        regions_config: Region config dict (optional, will use defaults).
        verbose: Print debug info.
    
    Returns:
        Dictionary with:
            - 'ok': bool
            - 'Ld_px': np.ndarray (468, 2) - donor landmarks in pixels
            - 'delta_px': np.ndarray (468, 2) - asymmetry delta in pixels
            - 'mouth_mask': np.ndarray (468,) bool - mouth landmark mask
            - 'donor_shape': tuple (H, W)
        Or None if extraction failed.
    """
    # Validate inputs
    if donor_rgb is None:
        warnings.warn("[facemesh_exp_assist] donor_rgb is None")
        return None
    
    if not isinstance(donor_rgb, np.ndarray) or donor_rgb.ndim != 3:
        warnings.warn(f"[facemesh_exp_assist] donor_rgb has unexpected shape")
        return None
    
    if extractor is None:
        warnings.warn("[facemesh_exp_assist] extractor is None")
        return None
    
    try:
        # Import the existing function
        from scripts.facemesh_landmarks import (
            compute_donor_asymmetry_delta,
            load_facemesh_regions_config,
        )
        
        # Load default regions config if not provided
        if regions_config is None:
            regions_config = load_facemesh_regions_config(None)
        
        # Call existing infrastructure
        result = compute_donor_asymmetry_delta(
            donor_rgb=donor_rgb,
            extractor=extractor,
            regions_config=regions_config,
            apply_bias_removal=True,
            max_delta_px=None,
            clamp_percentile=98.0,
            cheek_radius_px=None,
            verbose=verbose,
        )
        
        if not result.get("ok", False):
            if verbose:
                print("[facemesh_exp_assist] compute_donor_asymmetry_delta failed")
            return None
        
        # Extract what we need
        L_d = result.get("L_d")  # (468, 2) or (478, 2)
        delta = result.get("delta")  # (468, 2) or (478, 2)
        
        if L_d is None or delta is None:
            warnings.warn("[facemesh_exp_assist] Missing L_d or delta in result")
            return None
        
        # Truncate to 468 if we got 478
        if L_d.shape[0] > 468:
            L_d = L_d[:468]
        if delta.shape[0] > 468:
            delta = delta[:468]
        
        # Create mouth mask
        mouth_mask = np.zeros(468, dtype=bool)
        for idx in FACEMESH_MOUTH_ALL_IDX:
            if 0 <= idx < 468:
                mouth_mask[idx] = True
        
        H, W = donor_rgb.shape[:2]
        
        if verbose:
            lip_delta = delta[mouth_mask]
            lip_mag = np.linalg.norm(lip_delta, axis=1)
            print(f"[facemesh_exp_assist] Donor delta computed: "
                  f"mouth mean|delta|={lip_mag.mean():.2f}px, max={lip_mag.max():.2f}px")
        
        return {
            "ok": True,
            "Ld_px": L_d.astype(np.float32),
            "delta_px": delta.astype(np.float32),
            "mouth_mask": mouth_mask,
            "donor_shape": (H, W),
        }
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] compute_facemesh_donor_delta_px failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def align_delta_to_target_px(
    Ld_px: np.ndarray,
    delta_donor_px: np.ndarray,
    donor_shape: Tuple[int, int],
    target_rgb: np.ndarray,
    extractor: Any,
    anchors: List[int] = None,
    *,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Align donor delta vectors into target coordinate frame.
    
    Uses Umeyama similarity transform on anchor points to compute
    scale and rotation, then applies only sR (no translation) to delta vectors.
    
    Args:
        Ld_px: Donor landmarks (468, 2) in pixels.
        delta_donor_px: Donor delta (468, 2) in pixels.
        donor_shape: (H, W) of donor image.
        target_rgb: Target RGB image.
        extractor: FaceMeshLandmarkExtractor instance.
        anchors: Anchor point indices for transform. Default [10, 152, 234, 454].
        verbose: Print debug info.
    
    Returns:
        Dictionary with:
            - 'ok': bool
            - 'Lt_px': np.ndarray (468, 2) - target landmarks in pixels
            - 'delta_target_px': np.ndarray (468, 2) - aligned delta
            - 'transform': dict with scale, rotation_deg, translation
        Or None on failure.
    """
    if anchors is None:
        anchors = FACEMESH_ANCHOR_IDX.copy()
    
    # Validate inputs
    if Ld_px is None or delta_donor_px is None:
        warnings.warn("[facemesh_exp_assist] Missing Ld_px or delta_donor_px")
        return None
    
    if target_rgb is None or extractor is None:
        warnings.warn("[facemesh_exp_assist] Missing target_rgb or extractor")
        return None
    
    try:
        # Extract landmarks on target
        ok_t, Lt_px, _ = extractor.extract(target_rgb)
        
        if not ok_t or Lt_px is None:
            warnings.warn("[facemesh_exp_assist] Failed to extract target landmarks")
            return None
        
        # Truncate to 468 if needed
        if Lt_px.shape[0] > 468:
            Lt_px = Lt_px[:468]
        
        # Filter valid anchor indices
        valid_anchors = [i for i in anchors if 0 <= i < 468 and i < Ld_px.shape[0] and i < Lt_px.shape[0]]
        if len(valid_anchors) < 3:
            warnings.warn(f"[facemesh_exp_assist] Not enough valid anchors: {len(valid_anchors)}")
            return None
        
        # Extract anchor points
        src_anchors = Ld_px[valid_anchors].astype(np.float64)
        dst_anchors = Lt_px[valid_anchors].astype(np.float64)
        
        # Compute Umeyama similarity transform
        scale, R, t = _umeyama_similarity(src_anchors, dst_anchors)
        
        # Apply scale and rotation only to delta vectors (not translation)
        # delta_target = s * R @ delta_donor.T
        sR = scale * R
        delta_target_px = (sR @ delta_donor_px.T).T
        
        # Compute rotation angle for debug
        rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        
        if verbose:
            print(f"[facemesh_exp_assist] Similarity transform: "
                  f"scale={scale:.4f}, rotation={rotation_deg:.2f}°")
        
        return {
            "ok": True,
            "Lt_px": Lt_px.astype(np.float32),
            "delta_target_px": delta_target_px.astype(np.float32),
            "transform": {
                "scale": scale,
                "rotation_deg": rotation_deg,
                "translation": t.tolist(),
            },
        }
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] align_delta_to_target_px failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def _infer_mouth_kp_roles(
    kp_px: np.ndarray,
    lips_mask: np.ndarray,
) -> Dict[str, int]:
    """
    Infer which LivePortrait keypoint indices correspond to mouth roles
    using spatial heuristics.
    
    Returns dict with keys:
      - left_corner, right_corner, upper_center, lower_center, center
    Each maps to an LP keypoint index (0-20).
    """
    mouth_kp_idx = np.where(lips_mask)[0]
    
    if len(mouth_kp_idx) < 5:
        # Fallback: return empty/dummy dict
        return {
            "left_corner": -1,
            "right_corner": -1,
            "upper_center": -1,
            "lower_center": -1,
            "center": -1,
        }
    
    mouth_kp_px = kp_px[mouth_kp_idx]
    cx = mouth_kp_px[:, 0].mean()
    cy = mouth_kp_px[:, 1].mean()
    
    # Distance from center
    dx = mouth_kp_px[:, 0] - cx
    dy = mouth_kp_px[:, 1] - cy
    dist = np.sqrt(dx**2 + dy**2)
    
    # Corners: largest |x| displacement
    corner_mask = np.abs(dx) > np.percentile(np.abs(dx), 70)
    if corner_mask.sum() >= 2:
        corner_candidates = mouth_kp_idx[corner_mask]
        left_corner = corner_candidates[kp_px[corner_candidates, 0].argmin()]
        right_corner = corner_candidates[kp_px[corner_candidates, 0].argmax()]
    else:
        # Fallback: leftmost and rightmost
        left_corner = mouth_kp_idx[kp_px[mouth_kp_idx, 0].argmin()]
        right_corner = mouth_kp_idx[kp_px[mouth_kp_idx, 0].argmax()]
    
    # Remove corners from remaining
    remaining = np.array([idx for idx in mouth_kp_idx if idx not in [left_corner, right_corner]])
    
    if len(remaining) >= 3:
        # Upper vs lower
        upper_mask = kp_px[remaining, 1] < cy
        if upper_mask.sum() >= 1:
            upper_candidates = remaining[upper_mask]
            upper_center = upper_candidates[np.abs(kp_px[upper_candidates, 0] - cx).argmin()]
        else:
            upper_center = remaining[0]
        
        lower_candidates = remaining[kp_px[remaining, 1] >= cy]
        if len(lower_candidates) >= 1:
            lower_center = lower_candidates[np.abs(kp_px[lower_candidates, 0] - cx).argmin()]
        else:
            lower_center = remaining[-1]
        
        # Center: closest to mean
        center = remaining[dist[remaining - mouth_kp_idx[0]].argmin()] if len(remaining) > 0 else mouth_kp_idx[0]
    else:
        # Fallback
        upper_center = mouth_kp_idx[kp_px[mouth_kp_idx, 1].argmin()]
        lower_center = mouth_kp_idx[kp_px[mouth_kp_idx, 1].argmax()]
        center = mouth_kp_idx[dist.argmin()]
    
    return {
        "left_corner": int(left_corner),
        "right_corner": int(right_corner),
        "upper_center": int(upper_center),
        "lower_center": int(lower_center),
        "center": int(center),
    }


def _compute_basis_vectors(
    delta_target_px: np.ndarray,
    lips_fm_idx: List[int],
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute basis vectors for mouth shape preservation:
    - v_corner: average delta at FM corners (61, 291)
    - v_upper_mid: average delta at FM upper midline (0, 13)
    - v_lower_mid: average delta at FM lower midline (17, 14)
    
    Derived:
    - v_tilt = v_upper_mid - v_lower_mid
    - v_center = 0.5 * (v_upper_mid + v_lower_mid)
    
    Returns:
        v_corner (2,), v_tilt (2,), v_center (2,), stats dict
    """
    # FaceMesh landmark indices (constants from facemesh_landmarks)
    FACEMESH_LIP_CORNERS_IDX = [61, 291]
    FACEMESH_LIP_MID_TOP_IDX = [0, 13]
    FACEMESH_LIP_MID_BOT_IDX = [17, 14]
    
    stats = {
        "v_corner": [0.0, 0.0],
        "v_upper_mid": [0.0, 0.0],
        "v_lower_mid": [0.0, 0.0],
        "v_tilt": [0.0, 0.0],
        "v_center": [0.0, 0.0],
    }
    
    lips_fm_idx_set = set(lips_fm_idx)
    
    # Extract deltas at special indices
    v_corner = np.zeros(2, dtype=np.float32)
    v_upper_mid = np.zeros(2, dtype=np.float32)
    v_lower_mid = np.zeros(2, dtype=np.float32)
    
    # Average at corners
    corner_count = 0
    for idx in FACEMESH_LIP_CORNERS_IDX:
        if idx in lips_fm_idx_set:
            fm_idx_local = lips_fm_idx.index(idx)
            v_corner += delta_target_px[fm_idx_local, :2]
            corner_count += 1
    if corner_count > 0:
        v_corner /= corner_count
    
    # Average at upper midline
    upper_count = 0
    for idx in FACEMESH_LIP_MID_TOP_IDX:
        if idx in lips_fm_idx_set:
            fm_idx_local = lips_fm_idx.index(idx)
            v_upper_mid += delta_target_px[fm_idx_local, :2]
            upper_count += 1
    if upper_count > 0:
        v_upper_mid /= upper_count
    
    # Average at lower midline
    lower_count = 0
    for idx in FACEMESH_LIP_MID_BOT_IDX:
        if idx in lips_fm_idx_set:
            fm_idx_local = lips_fm_idx.index(idx)
            v_lower_mid += delta_target_px[fm_idx_local, :2]
            lower_count += 1
    if lower_count > 0:
        v_lower_mid /= lower_count
    
    # Derived vectors
    v_tilt = v_upper_mid - v_lower_mid
    v_center = 0.5 * (v_upper_mid + v_lower_mid)
    
    # Update stats
    stats["v_corner"] = v_corner.tolist()
    stats["v_upper_mid"] = v_upper_mid.tolist()
    stats["v_lower_mid"] = v_lower_mid.tolist()
    stats["v_tilt"] = v_tilt.tolist()
    stats["v_center"] = v_center.tolist()
    
    if verbose:
        print(f"[facemesh_exp_assist] Basis vectors computed:")
        print(f"  v_corner: {v_corner}")
        print(f"  v_tilt:   {v_tilt}")
        print(f"  v_center: {v_center}")
    
    return v_corner, v_tilt, v_center, stats


def project_delta_to_exp_basis(
    kp_px: np.ndarray,
    lips_fm_idx: List[int],
    delta_target_px: np.ndarray,
    mouth_kp_roles: Dict[str, int],
    *,
    mouth_alpha: float = 1.0,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Project FaceMesh lip deltas to LivePortrait keypoints using basis method.
    
    This preserves mouth shape characteristics:
    - Corner droop asymmetry (left vs right)
    - Lip-line tilt (upper vs lower midline shift)
    - Center droop
    
    Args:
        kp_px: LP keypoints in pixel space (N, 2).
        lips_fm_idx: FaceMesh indices for lip landmarks.
        delta_target_px: Aligned delta (468, 2) or subset.
        mouth_kp_roles: Dict with left_corner, right_corner, upper_center, lower_center, center.
        mouth_alpha: Scaling factor for mouth displacements.
        verbose: Print debug info.
    
    Returns:
        Tuple of:
            - disp_px: (N, 2) displacement in pixels for each LP keypoint
            - stats: dict with statistics
    """
    stats = {
        "n_kp": kp_px.shape[0],
        "n_lip_fm": len(lips_fm_idx),
        "method": "basis",
        "mean_disp_px": 0.0,
        "max_disp_px": 0.0,
        "basis_stats": {},
    }
    
    if kp_px is None or lips_fm_idx is None or delta_target_px is None:
        return None, stats
    
    try:
        N = kp_px.shape[0]
        
        # Compute basis vectors
        v_corner, v_tilt, v_center, basis_stats = _compute_basis_vectors(
            delta_target_px, lips_fm_idx, verbose=verbose
        )
        stats["basis_stats"] = basis_stats
        
        # Initialize displacement array
        disp_px = np.zeros((N, 2), dtype=np.float32)
        
        # Get roles
        left_corner_idx = mouth_kp_roles.get("left_corner", -1)
        right_corner_idx = mouth_kp_roles.get("right_corner", -1)
        upper_center_idx = mouth_kp_roles.get("upper_center", -1)
        lower_center_idx = mouth_kp_roles.get("lower_center", -1)
        center_idx = mouth_kp_roles.get("center", -1)
        
        # Apply basis displacements to mouth keypoints
        # Corners get v_corner + ±0.5*v_tilt
        if left_corner_idx >= 0:
            disp_px[left_corner_idx] = mouth_alpha * (v_corner + 0.5 * v_tilt)
        
        if right_corner_idx >= 0:
            disp_px[right_corner_idx] = mouth_alpha * (v_corner - 0.5 * v_tilt)
        
        # Upper/lower center get v_center ± 0.5*v_tilt
        if upper_center_idx >= 0:
            disp_px[upper_center_idx] = mouth_alpha * (v_center + 0.5 * v_tilt)
        
        if lower_center_idx >= 0:
            disp_px[lower_center_idx] = mouth_alpha * (v_center - 0.5 * v_tilt)
        
        # Center gets pure v_center
        if center_idx >= 0:
            disp_px[center_idx] = mouth_alpha * v_center
        
        # Compute stats
        mags = np.linalg.norm(disp_px, axis=1)
        stats["mean_disp_px"] = float(mags.mean())
        stats["max_disp_px"] = float(mags.max())
        
        if verbose:
            print(f"[facemesh_exp_assist] Basis projection: "
                  f"mean_disp={stats['mean_disp_px']:.3f}px max_disp={stats['max_disp_px']:.3f}px")
        
        return disp_px, stats
    
    except Exception as e:
        print(f"[facemesh_exp_assist] Error in basis projection: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None, stats


def project_delta_to_exp_knn(
    kp_px: np.ndarray,
    Lt_px: np.ndarray,
    delta_target_px: np.ndarray,
    lips_fm_idx: List[int],
    *,
    mouth_alpha: float = 1.0,
    corner_boost: float = 1.5,
    k: int = 8,
    smooth: bool = False,
    smooth_k: int = 6,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Project FaceMesh lip deltas to LivePortrait keypoints using KNN.
    
    For each LP keypoint position, find K nearest FaceMesh lip landmarks
    and compute inverse-distance-weighted average of their deltas.
    
    Args:
        kp_px: LP keypoints in pixel space (N, 2).
        Lt_px: Target FaceMesh landmarks (468, 2).
        delta_target_px: Aligned delta (468, 2).
        lips_fm_idx: FaceMesh indices for lip landmarks.
        mouth_alpha: Scaling factor for mouth deltas.
        corner_boost: Extra boost for corner landmarks (61, 291).
        k: Number of nearest neighbors.
        smooth: Enable smoothing in FaceMesh lip space (before projection).
        smooth_k: K for lip vector smoothing.
        verbose: Print debug info.
    
    Returns:
        Tuple of:
            - disp_px: (N, 2) displacement in pixels for each LP keypoint
            - stats: dict with statistics
    """
    stats = {
        "n_kp": 0,
        "n_lip_fm": 0,
        "mean_disp_px": 0.0,
        "max_disp_px": 0.0,
        "smoothing_enabled": False,
        "smoothing_k": 0,
        "smoothing_lam": 0.0,
        "smoothing_mean_l2_diff": 0.0,
        "lip_vec_raw": None,
        "lip_vec_smoothed": None,
    }
    
    if kp_px is None or Lt_px is None or delta_target_px is None:
        return None, stats
    
    try:
        N = kp_px.shape[0]
        stats["n_kp"] = N
        
        # Filter valid lip indices
        valid_lips = [i for i in lips_fm_idx if 0 <= i < 468]
        stats["n_lip_fm"] = len(valid_lips)
        
        if len(valid_lips) == 0:
            warnings.warn("[facemesh_exp_assist] No valid lip indices")
            return np.zeros((N, 2), dtype=np.float32), stats
        
        # Get lip landmarks and deltas
        lip_pts = Lt_px[valid_lips]  # (M, 2)
        lip_deltas = delta_target_px[valid_lips]  # (M, 2)
        
        # Apply mouth_alpha and corner boost
        delta_scaled = lip_deltas * mouth_alpha
        for i, fm_idx in enumerate(valid_lips):
            if fm_idx in FACEMESH_LIP_CORNERS_IDX:
                delta_scaled[i] *= corner_boost
        
        # Store raw lip vectors BEFORE smoothing for debug comparison
        lip_vec_raw = delta_scaled.copy()
        lip_vec_smoothed = delta_scaled.copy()
        smoothing_stats = {}
        
        # ===== SMOOTHING: Apply in FaceMesh lip space BEFORE projection =====
        if smooth:
            lip_vec_smoothed, smoothing_stats = _smooth_lip_vectors_knn(
                lip_xy=lip_pts,
                lip_vec=delta_scaled,
                k=smooth_k,
                lam=0.5,
                verbose=verbose,
            )
            stats["smoothing_enabled"] = True
            stats["smoothing_k"] = smooth_k
            stats["smoothing_lam"] = 0.5
            stats["smoothing_mean_l2_diff"] = smoothing_stats.get("mean_l2_diff", 0.0)
        else:
            # Smoothing disabled: smoothed = raw
            if verbose:
                print(f"[facemesh_exp_assist] Lip smoothing disabled")
        
        # Store for debug output
        stats["lip_vec_raw"] = lip_vec_raw
        stats["lip_vec_smoothed"] = lip_vec_smoothed
        
        # Use smoothed (or raw if smoothing disabled) for projection
        delta_to_project = lip_vec_smoothed
        
        # KNN projection for each LP keypoint
        disp_px = np.zeros((N, 2), dtype=np.float32)
        
        for p_idx in range(N):
            p = kp_px[p_idx]  # (2,)
            
            # Compute distances to all lip landmarks
            dists = np.linalg.norm(lip_pts - p, axis=1)  # (M,)
            
            # Find K nearest
            k_actual = min(k, len(valid_lips))
            nearest_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            nearest_dists = dists[nearest_idx]
            nearest_deltas = delta_to_project[nearest_idx]
            
            # Inverse distance weights (with epsilon for numerical stability)
            eps = 1e-6
            weights = 1.0 / (nearest_dists + eps)
            weights /= weights.sum()  # Normalize
            
            # Weighted average displacement
            disp_px[p_idx] = np.sum(weights[:, None] * nearest_deltas, axis=0)
        
        # Compute stats
        disp_mag = np.linalg.norm(disp_px, axis=1)
        stats["mean_disp_px"] = float(disp_mag.mean())
        stats["max_disp_px"] = float(disp_mag.max())
        
        if verbose:
            print(f"[facemesh_exp_assist] KNN projection: "
                  f"mean|disp|={stats['mean_disp_px']:.3f}px, max={stats['max_disp_px']:.3f}px")
        
        return disp_px, stats
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] project_delta_to_exp_knn failed: {e}")
        return None, stats


# ---------------------------------------------------------------------------
# Guardrails: Capping, Smoothing, Stabilization
# ---------------------------------------------------------------------------

def _cap_displacement_px(
    disp_px: np.ndarray,
    mouth_mask: np.ndarray,
    max_disp_px: Optional[float] = None,
    cap_percentile: float = 98.0,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cap displacement magnitude in pixel space.
    
    Args:
        disp_px: (N, 2) displacements in pixels.
        mouth_mask: (N,) bool mask for mouth keypoints.
        max_disp_px: Absolute cap in pixels. If None, use percentile.
        cap_percentile: Percentile for auto cap (default 98).
        verbose: Print debug info.
    
    Returns:
        Tuple of:
            - disp_px_capped: Capped displacements
            - stats: dict with pre/post-cap magnitudes
    """
    stats = {
        "max_before_cap": 0.0,
        "max_after_cap": 0.0,
        "cap_value": 0.0,
        "n_affected": 0,
    }
    
    try:
        if disp_px is None or mouth_mask is None:
            return disp_px, stats
        
        disp_px_capped = disp_px.copy()
        
        # Compute magnitude
        disp_mag = np.linalg.norm(disp_px, axis=1)
        stats["max_before_cap"] = float(disp_mag.max())
        
        # Determine cap value
        if max_disp_px is not None:
            cap_val = float(max_disp_px)
        else:
            # Use percentile within mouth keypoints only
            mouth_mag = disp_mag[mouth_mask]
            if len(mouth_mag) > 0:
                cap_val = float(np.percentile(mouth_mag, cap_percentile))
            else:
                cap_val = float(np.percentile(disp_mag, cap_percentile))
        
        stats["cap_value"] = cap_val
        
        # Apply cap (clamp magnitude)
        if cap_val > 0:
            disp_mag_capped = np.minimum(disp_mag, cap_val)
            scale_factor = np.where(disp_mag > 1e-6, disp_mag_capped / disp_mag, 1.0)
            disp_px_capped = disp_px * scale_factor[:, None]
            
            # Count affected
            stats["n_affected"] = int(np.sum(disp_mag > cap_val))
        
        disp_mag_after = np.linalg.norm(disp_px_capped, axis=1)
        stats["max_after_cap"] = float(disp_mag_after.max())
        
        if verbose:
            print(f"[facemesh_exp_assist] Cap: max_before={stats['max_before_cap']:.2f}px, "
                  f"cap={cap_val:.2f}px, max_after={stats['max_after_cap']:.2f}px, "
                  f"affected={stats['n_affected']}")
        
        return disp_px_capped, stats
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] _cap_displacement_px failed: {e}")
        return disp_px, stats


def _smooth_lip_vectors_knn(
    lip_xy: np.ndarray,
    lip_vec: np.ndarray,
    k: int = 6,
    lam: float = 0.5,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Smooth lip displacement vectors in FaceMesh landmark space via KNN.
    
    For each lip landmark, find K nearest neighbors (by position) and blend
    its vector with the mean of neighbor vectors:
    output[i] = (1-lam)*vec[i] + lam*mean(vec[neighbors])
    
    This operates BEFORE projection to LivePortrait keypoints, ensuring
    we never exceed the lip landmark count (~40).
    
    Args:
        lip_xy: (Nlip, 2) lip landmark positions in pixels.
        lip_vec: (Nlip, 2) lip displacement vectors in pixels.
        k: Number of nearest neighbors (excluding self).
        lam: Blending weight (0=no smoothing, 1=full neighbor average).
        verbose: Print debug info.
    
    Returns:
        Tuple of:
            - vec_smoothed: (Nlip, 2) smoothed displacement vectors
            - stats: dict with smoothing statistics
    """
    stats = {
        "mean_mag_before": 0.0,
        "mean_mag_after": 0.0,
        "max_mag_before": 0.0,
        "max_mag_after": 0.0,
        "mean_l2_diff": 0.0,
    }
    
    try:
        if lip_xy is None or lip_vec is None:
            return lip_vec, stats
        
        Nlip = lip_xy.shape[0]
        if Nlip == 0:
            return lip_vec, stats
        
        vec_smoothed = lip_vec.copy()
        
        for i in range(Nlip):
            # Compute distances to all other lip landmarks
            dists = np.linalg.norm(lip_xy - lip_xy[i], axis=1)  # (Nlip,)
            
            # Find K+1 nearest (includes self at distance 0)
            k_actual = min(k + 1, Nlip)
            nearest_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            
            # Exclude self (distance = 0)
            neighbor_idx = nearest_idx[dists[nearest_idx] > 1e-6][:k]
            
            if len(neighbor_idx) > 0:
                neighbor_mean = lip_vec[neighbor_idx].mean(axis=0)
                vec_smoothed[i] = (1.0 - lam) * lip_vec[i] + lam * neighbor_mean
        
        # Compute statistics
        before_mag = np.linalg.norm(lip_vec, axis=1)
        after_mag = np.linalg.norm(vec_smoothed, axis=1)
        diff_mag = np.linalg.norm(vec_smoothed - lip_vec, axis=1)
        
        stats["mean_mag_before"] = float(before_mag.mean())
        stats["mean_mag_after"] = float(after_mag.mean())
        stats["max_mag_before"] = float(before_mag.max())
        stats["max_mag_after"] = float(after_mag.max())
        stats["mean_l2_diff"] = float(diff_mag.mean())
        
        if verbose:
            print(f"[facemesh_exp_assist] Lip smoothing (pre-projection): "
                  f"mean|vec| before={stats['mean_mag_before']:.3f}, after={stats['mean_mag_after']:.3f}, "
                  f"max before={stats['max_mag_before']:.3f}, max after={stats['max_mag_after']:.3f}, "
                  f"mean_diff={stats['mean_l2_diff']:.3f}")
        
        return vec_smoothed, stats
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] _smooth_lip_vectors_knn failed: {e}")
        return lip_vec, stats


def _compute_metrics(
    donor_rgb: np.ndarray,
    target_rgb: np.ndarray,
    img_asym: np.ndarray,
    extractor: Any,
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute simple mouth-related metrics: droop and tilt.
    
    Droop: y difference of lip corners
    Tilt: slope of lip line
    
    Args:
        donor_rgb: Donor image.
        target_rgb: Target image (before asymmetry).
        img_asym: Output image (after asymmetry).
        extractor: FaceMeshLandmarkExtractor.
        verbose: Print debug info.
    
    Returns:
        dict with metrics for donor, target_before, target_after.
    """
    metrics = {
        "donor": {},
        "target_before": {},
        "target_after": {},
    }
    
    try:
        # Try to use existing facemesh_metrics if available
        try:
            from scripts.facemesh_metrics import compute_metrics
            
            for img_name, img in [("donor", donor_rgb), ("target_before", target_rgb), ("target_after", img_asym)]:
                ok, L, vis = extractor.extract(img)
                if ok and L is not None:
                    m = compute_metrics(L, vis)
                    if isinstance(m, dict):
                        metrics[img_name] = m
                        if verbose:
                            print(f"[facemesh_exp_assist] {img_name} metrics: {m}")
                    
        except (ImportError, Exception) as e:
            # Fallback: compute simple droop/tilt
            if verbose:
                print(f"[facemesh_exp_assist] facemesh_metrics not available, using simple fallback: {e}")
            
            for img_name, img in [("donor", donor_rgb), ("target_before", target_rgb), ("target_after", img_asym)]:
                ok, L, vis = extractor.extract(img)
                if ok and L is not None:
                    # Simple droop: y difference of corners (61, 291)
                    if L.shape[0] > 291:
                        y_left = L[61, 1]
                        y_right = L[291, 1]
                        droop = float(abs(y_right - y_left))
                        
                        # Simple tilt: slope of midline (0, 17)
                        if L.shape[0] > 17:
                            y_top = L[0, 1]
                            y_bot = L[17, 1]
                            x_top = L[0, 0]
                            x_bot = L[17, 0]
                            tilt = float(abs(y_bot - y_top) / (abs(x_bot - x_top) + 1e-6))
                            
                            metrics[img_name] = {
                                "droop": droop,
                                "tilt": tilt,
                            }
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] _compute_metrics failed: {e}")
    
    return metrics


def _draw_debug_images(
    target_rgb: np.ndarray,
    kp_px: np.ndarray,
    disp_px: np.ndarray,
    Lt_px: np.ndarray,
    delta_target_px: np.ndarray,
    lips_fm_idx: List[int],
    output_dir: str,
    *,
    verbose: bool = False,
) -> None:
    """
    Draw debug images: keypoint vectors and FaceMesh vectors.
    
    Args:
        target_rgb: Target image.
        kp_px: LP keypoints (N, 2).
        disp_px: LP displacements (N, 2).
        Lt_px: FaceMesh landmarks (468, 2).
        delta_target_px: FaceMesh deltas (468, 2).
        lips_fm_idx: Mouth FaceMesh indices.
        output_dir: Where to save images.
        verbose: Print debug info.
    """
    try:
        import cv2
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Draw LP keypoint vectors
        img_kp = target_rgb.copy()
        for i in range(len(kp_px)):
            pt = tuple(kp_px[i].astype(int))
            disp = disp_px[i]
            if np.linalg.norm(disp) > 0.1:
                pt_end = tuple((kp_px[i] + disp).astype(int))
                cv2.arrowedLine(img_kp, pt, pt_end, (0, 255, 0), 2, tipLength=0.3)
                cv2.circle(img_kp, pt, 3, (0, 255, 0), -1)
        
        kp_out = os.path.join(output_dir, "kp_disp_vectors.png")
        cv2.imwrite(kp_out, cv2.cvtColor(img_kp, cv2.COLOR_RGB2BGR))
        if verbose:
            print(f"[facemesh_exp_assist] Saved {kp_out}")
        
        # 2. Draw FaceMesh lip vectors
        img_fm = target_rgb.copy()
        valid_lips = [i for i in lips_fm_idx if 0 <= i < 468]
        for idx in valid_lips:
            pt = tuple(Lt_px[idx].astype(int))
            disp = delta_target_px[idx]
            if np.linalg.norm(disp) > 0.1:
                pt_end = tuple((Lt_px[idx] + disp).astype(int))
                cv2.arrowedLine(img_fm, pt, pt_end, (255, 0, 0), 1, tipLength=0.2)
                cv2.circle(img_fm, pt, 2, (255, 0, 0), -1)
        
        fm_out = os.path.join(output_dir, "fm_lips_vectors.png")
        cv2.imwrite(fm_out, cv2.cvtColor(img_fm, cv2.COLOR_RGB2BGR))
        if verbose:
            print(f"[facemesh_exp_assist] Saved {fm_out}")
        
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] _draw_debug_images failed: {e}")



def apply_facemesh_exp_assist(
    donor_rgb: np.ndarray,
    target_rgb: np.ndarray,
    exp_delta: Any,  # torch.Tensor
    extractor: Any,
    lp_keypoints_px: np.ndarray,
    *,
    beta: float = 1.0,
    mouth_alpha: float = 1.0,
    method: str = "knn",
    knn_k: int = 8,
    tps_reg: float = 1e-3,
    inject_stage: str = "post_drift",
    max_disp_px: Optional[float] = None,
    cap_percentile: float = 98.0,
    smooth: bool = True,
    smooth_k: int = 6,
    zero_stable: bool = True,
    debug: bool = False,
    lips_mask_indices: Optional[List[int]] = None,
    corner_mask_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Orchestrator: compute FaceMesh-based exp delta correction with guardrails.
    
    Full pipeline:
    1. Compute donor asymmetry delta via FaceMesh
    2. Extract target landmarks and align delta
    3. Project lip deltas to LP keypoints via KNN
    4. CAP displacements (guardrail 1)
    5. SMOOTH displacements on FaceMesh graph (guardrail 2)
    6. ZERO out stable keypoints (guardrail 3)
    7. Convert pixel displacements to exp units
    8. Build exp_delta_fm matching exp_delta shape
    9. Compute metrics and save debug outputs
    
    Args:
        donor_rgb: Donor image (HxWx3 RGB).
        target_rgb: Target image (HxWx3 RGB).
        exp_delta: Current exp_delta tensor from LP pipeline (B, N, 3).
        extractor: FaceMeshLandmarkExtractor instance.
        lp_keypoints_px: (N, 2) LP keypoints in pixel coordinates.
        beta: Overall scaling for the FaceMesh correction.
        mouth_alpha: Extra scaling for mouth region.
        method: Projection method - 'knn' or 'tps'.
        knn_k: K for KNN projection.
        tps_reg: Regularization for TPS projection.
        inject_stage: When to inject - 'pre_drift' or 'post_drift'.
        max_disp_px: Absolute cap in pixels. If None, use percentile.
        cap_percentile: Percentile for auto-cap (default 98).
        smooth: Enable smoothing (default True).
        smooth_k: K for smoothing neighbors (default 6).
        zero_stable: Zero out stable keypoints (default True).
        debug: Save debug outputs.
        lips_mask_indices: LP keypoint indices for lips.
        corner_mask_indices: LP keypoint indices for lip corners.
        verbose: Print debug info.
    
    Returns:
        Tuple of:
            - exp_delta_fm: Correction tensor (same shape as exp_delta), or None
            - debug_dict: Dictionary with debug info
    """
    import torch
    
    debug_dict: Dict[str, Any] = {
        "ok": False,
        "beta": beta,
        "mouth_alpha": mouth_alpha,
        "method": method,
        "knn_k": knn_k,
        "tps_reg": tps_reg,
        "inject_stage": inject_stage,
        "exp_shape": None,
        "exp_dtype": None,
        "lips_mask_count": 0,
        "corner_mask_count": 0,
        "transform": None,
        "mouth_disp_px_before_cap_mean": 0.0,
        "mouth_disp_px_before_cap_max": 0.0,
        "mouth_disp_px_after_cap_mean": 0.0,
        "mouth_disp_px_after_cap_max": 0.0,
        "smoothing_enabled": False,
        "smoothing_k": 0,
        "smoothing_lam": 0.0,
        "smoothing_mean_l2_diff": 0.0,
        "mean_lip_disp_exp": 0.0,
        "max_lip_disp_exp": 0.0,
        "mouth_kps_affected": 0,
        "error": None,
    }
    
    # === Validate inputs ===
    if donor_rgb is None:
        debug_dict["error"] = "donor_rgb is None"
        return None, debug_dict
    
    if target_rgb is None:
        debug_dict["error"] = "target_rgb is None"
        return None, debug_dict
    
    if exp_delta is None:
        debug_dict["error"] = "exp_delta is None"
        return None, debug_dict
    
    if extractor is None:
        debug_dict["error"] = "extractor is None"
        return None, debug_dict
    
    if lp_keypoints_px is None:
        debug_dict["error"] = "lp_keypoints_px is None"
        return None, debug_dict
    
    # Get exp_delta shape info
    try:
        exp_shape = tuple(exp_delta.shape)
        debug_dict["exp_shape"] = exp_shape
        debug_dict["exp_dtype"] = str(exp_delta.dtype)
    except Exception as e:
        debug_dict["error"] = f"failed to get exp_delta info: {e}"
        return None, debug_dict
    
    # Count mask indices
    if lips_mask_indices is not None:
        debug_dict["lips_mask_count"] = len(lips_mask_indices)
    if corner_mask_indices is not None:
        debug_dict["corner_mask_count"] = len(corner_mask_indices)
    
    try:
        # Step 1: Compute donor asymmetry delta
        donor_result = compute_facemesh_donor_delta_px(
            donor_rgb=donor_rgb,
            extractor=extractor,
            regions_config=None,
            verbose=verbose,
        )
        
        if donor_result is None or not donor_result.get("ok", False):
            debug_dict["error"] = "Failed to compute donor delta"
            return None, debug_dict
        
        Ld_px = donor_result["Ld_px"]
        delta_donor_px = donor_result["delta_px"]
        donor_shape = donor_result["donor_shape"]
        mouth_mask_fm = donor_result["mouth_mask"]
        
        # Step 2: Align delta to target frame
        align_result = align_delta_to_target_px(
            Ld_px=Ld_px,
            delta_donor_px=delta_donor_px,
            donor_shape=donor_shape,
            target_rgb=target_rgb,
            extractor=extractor,
            anchors=FACEMESH_ANCHOR_IDX,
            verbose=verbose,
        )
        
        if align_result is None or not align_result.get("ok", False):
            debug_dict["error"] = "Failed to align delta to target"
            return None, debug_dict
        
        Lt_px = align_result["Lt_px"]
        delta_target_px = align_result["delta_target_px"]
        debug_dict["transform"] = align_result["transform"]
        
        # Step 3: Project lip deltas to LP keypoints
        if method == "basis":
            # Basis method: preserve mouth shape (tilt + center droop)
            if verbose:
                print("[facemesh_exp_assist] Using basis method for mouth shape preservation")
            
            # First, create a lips_mask for the basis method
            lips_mask_for_basis = np.zeros(lp_keypoints_px.shape[0], dtype=bool)
            if lips_mask_indices is not None:
                for idx in lips_mask_indices:
                    if 0 <= idx < lp_keypoints_px.shape[0]:
                        lips_mask_for_basis[idx] = True
            if corner_mask_indices is not None:
                for idx in corner_mask_indices:
                    if 0 <= idx < lp_keypoints_px.shape[0]:
                        lips_mask_for_basis[idx] = True
            
            # Infer mouth keypoint roles
            mouth_kp_roles = _infer_mouth_kp_roles(lp_keypoints_px, lips_mask_for_basis)
            
            if verbose:
                print(f"[facemesh_exp_assist] Inferred mouth roles: {mouth_kp_roles}")
            
            disp_px, proj_stats = project_delta_to_exp_basis(
                kp_px=lp_keypoints_px,
                lips_fm_idx=FACEMESH_MOUTH_ALL_IDX,
                delta_target_px=delta_target_px,
                mouth_kp_roles=mouth_kp_roles,
                mouth_alpha=mouth_alpha,
                verbose=verbose,
            )
            
            # Store basis stats for debug
            debug_dict["basis_stats"] = proj_stats.get("basis_stats", {})
            debug_dict["mouth_kp_roles"] = mouth_kp_roles
            
        elif method == "knn":
            disp_px, proj_stats = project_delta_to_exp_knn(
                kp_px=lp_keypoints_px,
                Lt_px=Lt_px,
                delta_target_px=delta_target_px,
                lips_fm_idx=FACEMESH_MOUTH_ALL_IDX,
                mouth_alpha=mouth_alpha,
                corner_boost=1.5,
                k=knn_k,
                smooth=smooth,
                smooth_k=smooth_k,
                verbose=verbose,
            )
        else:
            # TPS not implemented yet, fall back to KNN
            if verbose:
                print("[facemesh_exp_assist] TPS not implemented, using KNN")
            disp_px, proj_stats = project_delta_to_exp_knn(
                kp_px=lp_keypoints_px,
                Lt_px=Lt_px,
                delta_target_px=delta_target_px,
                lips_fm_idx=FACEMESH_MOUTH_ALL_IDX,
                mouth_alpha=mouth_alpha,
                corner_boost=1.5,
                k=knn_k,
                smooth=smooth,
                smooth_k=smooth_k,
                verbose=verbose,
            )
        
        if disp_px is None:
            debug_dict["error"] = "Failed to project delta to LP keypoints"
            return None, debug_dict
        
        # Store smoothing info from projection stats
        debug_dict["smoothing_enabled"] = proj_stats.get("smoothing_enabled", False)
        debug_dict["smoothing_k"] = proj_stats.get("smoothing_k", 0)
        debug_dict["smoothing_lam"] = proj_stats.get("smoothing_lam", 0.0)
        debug_dict["smoothing_mean_l2_diff"] = proj_stats.get("smoothing_mean_l2_diff", 0.0)
        
        # ===== Define mouth_kp_idx in LP keypoint space =====
        # Combine lips_mask and corner_mask indices
        mouth_kp_idx = np.zeros(disp_px.shape[0], dtype=bool)
        if lips_mask_indices is not None:
            for idx in lips_mask_indices:
                if 0 <= idx < disp_px.shape[0]:
                    mouth_kp_idx[idx] = True
        if corner_mask_indices is not None:
            for idx in corner_mask_indices:
                if 0 <= idx < disp_px.shape[0]:
                    mouth_kp_idx[idx] = True
        
        # ===== Compute stats BEFORE cap =====
        # Store disp_px_before_cap for before-cap statistics
        disp_px_before_cap = disp_px.copy()
        disp_mag_before_cap = np.linalg.norm(disp_px_before_cap, axis=1)
        
        # Compute magnitude-based stats on mouth keypoints only (before cap)
        mouth_mag_before_cap = disp_mag_before_cap[mouth_kp_idx]
        debug_dict["mouth_disp_px_before_cap_mean"] = (
            float(mouth_mag_before_cap.mean()) if len(mouth_mag_before_cap) > 0 else 0.0
        )
        debug_dict["mouth_disp_px_before_cap_max"] = (
            float(mouth_mag_before_cap.max()) if len(mouth_mag_before_cap) > 0 else 0.0
        )
        
        # ===== GUARDRAIL 1: Cap displacements =====
        disp_px, cap_stats = _cap_displacement_px(
            disp_px=disp_px,
            mouth_mask=mouth_kp_idx,
            max_disp_px=max_disp_px,
            cap_percentile=cap_percentile,
            verbose=verbose,
        )
        debug_dict.update({
            "cap_value_px": cap_stats["cap_value"],
            "n_capped": cap_stats["n_affected"],
        })
        
        # ===== Compute stats AFTER cap =====
        # Note: Smoothing is now applied in FaceMesh lip space (inside project_delta_to_exp_knn)
        # before projection to LivePortrait keypoints. This prevents index out of bounds errors.
        
        # Recompute magnitude-based stats after cap on mouth keypoints
        disp_mag_after_cap = np.linalg.norm(disp_px, axis=1)
        mouth_mag_after_cap = disp_mag_after_cap[mouth_kp_idx]
        debug_dict["mouth_disp_px_after_cap_mean"] = (
            float(mouth_mag_after_cap.mean()) if len(mouth_mag_after_cap) > 0 else 0.0
        )
        debug_dict["mouth_disp_px_after_cap_max"] = (
            float(mouth_mag_after_cap.max()) if len(mouth_mag_after_cap) > 0 else 0.0
        )

        
        # Step 4: Convert pixel displacements to exp units
        # exp values are typically in [-1, 1] normalized space
        # Conversion: dx_exp = dx_px * 2 / (W - 1), dy_exp = dy_px * 2 / (H - 1)
        H_t, W_t = target_rgb.shape[:2]
        
        # Check if exp values are in [-1, 1] range by looking at exp_delta statistics
        exp_np = exp_delta.detach().cpu().numpy() if isinstance(exp_delta, torch.Tensor) else exp_delta
        exp_range = np.abs(exp_np).max()
        
        if exp_range <= 1.5:
            # Normalized [-1, 1] space
            scale_x = 2.0 / (W_t - 1)
            scale_y = 2.0 / (H_t - 1)
        else:
            # Pixel-like space (less common)
            scale_x = 1.0 / (W_t - 1)
            scale_y = 1.0 / (H_t - 1)
        
        disp_exp = np.zeros_like(disp_px)
        disp_exp[:, 0] = disp_px[:, 0] * scale_x
        disp_exp[:, 1] = disp_px[:, 1] * scale_y
        
        # Apply beta scaling
        disp_exp *= beta
        
        # Step 5: Build exp_delta_fm with same shape as exp_delta
        # exp_delta has shape (B, N, 3) where 3 is (x, y, z)
        # We only modify x and y channels
        
        N_kp = exp_shape[1] if len(exp_shape) >= 2 else disp_exp.shape[0]
        
        # Create mask for which keypoints to affect (lips + corners)
        affect_mask = np.zeros(N_kp, dtype=bool)
        if lips_mask_indices is not None:
            for idx in lips_mask_indices:
                if 0 <= idx < N_kp:
                    affect_mask[idx] = True
        if corner_mask_indices is not None:
            for idx in corner_mask_indices:
                if 0 <= idx < N_kp:
                    affect_mask[idx] = True
        
        # ===== GUARDRAIL 3: Zero out stable keypoints =====
        # Only allow mouth keypoints to have non-zero displacement
        if zero_stable:
            # Non-mouth keypoints get zeroed
            non_mouth_mask = ~affect_mask
            disp_exp[non_mouth_mask] = 0.0
        
        # If no mask provided, affect all keypoints (less ideal but safe fallback)
        if not affect_mask.any():
            affect_mask[:] = True
            if verbose:
                print("[facemesh_exp_assist] Warning: no mask indices provided, affecting all keypoints")
        
        debug_dict["mouth_kps_affected"] = int(affect_mask.sum())
        
        # Build exp_delta_fm
        exp_delta_fm_np = np.zeros(exp_shape, dtype=np.float32)
        
        # Only write to affected keypoints
        for i in range(min(N_kp, disp_exp.shape[0])):
            if affect_mask[i]:
                exp_delta_fm_np[0, i, 0] = disp_exp[i, 0]  # x
                exp_delta_fm_np[0, i, 1] = disp_exp[i, 1]  # y
                # z channel left at 0
        
        # Compute final stats
        affected_disp = disp_exp[affect_mask[:disp_exp.shape[0]]]
        if len(affected_disp) > 0:
            affected_mag = np.linalg.norm(affected_disp, axis=1)
            debug_dict["mean_lip_disp_exp"] = float(affected_mag.mean())
            debug_dict["max_lip_disp_exp"] = float(affected_mag.max())
        
        # Convert to torch tensor matching exp_delta
        if isinstance(exp_delta, torch.Tensor):
            exp_delta_fm = torch.from_numpy(exp_delta_fm_np).to(
                dtype=exp_delta.dtype,
                device=exp_delta.device
            )
        else:
            exp_delta_fm = exp_delta_fm_np
        
        debug_dict["ok"] = True
        
        # Build summary line with smoothing info
        smoothing_info = ""
        if debug_dict["smoothing_enabled"]:
            smoothing_info = (f", smooth_enabled=True "
                            f"k={debug_dict['smoothing_k']} "
                            f"diff={debug_dict['smoothing_mean_l2_diff']:.3f}px")
        else:
            smoothing_info = ", smooth_enabled=False"
        
        # Print summary line with magnitude-based stats (after cap)
        print(f"[FaceMesh-EXP] added exp_delta_fm: "
              f"mouth_disp_px_before_cap_mean={debug_dict['mouth_disp_px_before_cap_mean']:.3f}, "
              f"mouth_disp_px_before_cap_max={debug_dict['mouth_disp_px_before_cap_max']:.3f}, "
              f"mouth_disp_px_after_cap_mean={debug_dict['mouth_disp_px_after_cap_mean']:.3f}, "
              f"mouth_disp_px_after_cap_max={debug_dict['mouth_disp_px_after_cap_max']:.3f}{smoothing_info}, "
              f"mouth_kps={debug_dict['mouth_kps_affected']}")
        
        # Save debug outputs if requested
        if debug:
            _save_debug_outputs(
                debug_dict=debug_dict,
                kp_px=lp_keypoints_px,
                disp_px=disp_px,
                exp_delta_fm_np=exp_delta_fm_np,
                lip_vec_raw=proj_stats.get("lip_vec_raw"),
                lip_vec_smoothed=proj_stats.get("lip_vec_smoothed"),
                target_rgb=target_rgb,
                mouth_kp_roles=debug_dict.get("mouth_kp_roles"),
                verbose=verbose,
            )
            
            # Draw debug images (arrow overlays for visualization)
            _draw_debug_images(
                target_rgb=target_rgb,
                kp_px=lp_keypoints_px,
                disp_px=disp_px,
                Lt_px=Lt_px,
                delta_target_px=delta_target_px,
                lips_fm_idx=FACEMESH_MOUTH_ALL_IDX,
                output_dir=FACEMESH_EXP_ASSIST_DEBUG_DIR,
                verbose=verbose,
            )
        
        return exp_delta_fm, debug_dict
        
    except Exception as e:
        debug_dict["error"] = f"Exception: {e}"
        warnings.warn(f"[facemesh_exp_assist] {debug_dict['error']}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None, debug_dict


def _save_debug_outputs(
    debug_dict: Dict[str, Any],
    kp_px: Optional[np.ndarray] = None,
    disp_px: Optional[np.ndarray] = None,
    exp_delta_fm_np: Optional[np.ndarray] = None,
    lip_vec_raw: Optional[np.ndarray] = None,
    lip_vec_smoothed: Optional[np.ndarray] = None,
    target_rgb: Optional[np.ndarray] = None,
    mouth_kp_roles: Optional[Dict[str, int]] = None,
    *,
    verbose: bool = False,
) -> None:
    """
    Save debug outputs to diagnostics directory.
    """
    try:
        os.makedirs(FACEMESH_EXP_ASSIST_DEBUG_DIR, exist_ok=True)
        
        # Save JSON summary (including basis_stats if present)
        save_dict = {}
        for k, v in debug_dict.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                save_dict[k] = v
            elif isinstance(v, (list, tuple)):
                save_dict[k] = list(v)
            elif isinstance(v, dict):
                save_dict[k] = {str(kk): vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool, type(None), list))}
            else:
                save_dict[k] = str(v)
        
        json_path = os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "summary.json")
        with open(json_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        # Save numpy arrays
        if kp_px is not None:
            np.save(os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "kp_px.npy"), kp_px)
        
        if disp_px is not None:
            np.save(os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "lip_disp_px.npy"), disp_px)
        
        if exp_delta_fm_np is not None:
            np.save(os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "exp_delta_fm.npy"), exp_delta_fm_np)
        
        # Save lip vectors for smoothing verification
        if lip_vec_raw is not None:
            np.save(os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "vec_lips_raw.npy"), lip_vec_raw)
        
        if lip_vec_smoothed is not None:
            np.save(os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "vec_lips_smoothed.npy"), lip_vec_smoothed)
        
        # Save basis vectors as JSON if available
        if "basis_stats" in debug_dict:
            basis_json_path = os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "fm_basis_vectors.json")
            with open(basis_json_path, 'w') as f:
                json.dump(debug_dict["basis_stats"], f, indent=2)
        
        # Draw mouth role labels on target image if basis method was used
        if mouth_kp_roles is not None and kp_px is not None and target_rgb is not None:
            try:
                import cv2
                overlay = target_rgb.copy()
                
                role_colors = {
                    "left_corner": (255, 0, 0),      # Blue
                    "right_corner": (0, 0, 255),     # Red
                    "upper_center": (0, 255, 0),     # Green
                    "lower_center": (255, 255, 0),   # Cyan
                    "center": (255, 0, 255),         # Magenta
                }
                
                for role, idx in mouth_kp_roles.items():
                    if idx >= 0 and idx < kp_px.shape[0]:
                        x, y = int(kp_px[idx, 0]), int(kp_px[idx, 1])
                        color = role_colors.get(role, (128, 128, 128))
                        cv2.circle(overlay, (x, y), 8, color, -1)
                        cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
                        
                        # Label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(overlay, role, (x + 10, y - 10),
                                   font, 0.5, color, 2)
                
                kp_mouth_roles_path = os.path.join(FACEMESH_EXP_ASSIST_DEBUG_DIR, "kp_mouth_roles.png")
                cv2.imwrite(kp_mouth_roles_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if verbose:
                    print(f"[facemesh_exp_assist] Saved mouth role labels to {kp_mouth_roles_path}")
            except Exception as e:
                if verbose:
                    print(f"[facemesh_exp_assist] Warning: Failed to draw mouth roles: {e}")
        
        if verbose:
            print(f"[facemesh_exp_assist] Saved debug outputs to {FACEMESH_EXP_ASSIST_DEBUG_DIR}/")
            
    except Exception as e:
        warnings.warn(f"[facemesh_exp_assist] Failed to save debug outputs: {e}")
