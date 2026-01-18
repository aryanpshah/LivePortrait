"""
Guardrails for FaceMesh-based warping (Phase 6).

This module is self-contained and operates only on FaceMesh landmarks, deltas,
weights, and image shape. It does not depend on LivePortrait internals.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from scripts.facemesh_landmarks import (
    FACEMESH_LIPS_IDX,
    FACEMESH_FACE_OVAL_IDX,
)

# Default anchor set (stable, central points)
DEFAULT_ANCHOR_IDX = [10, 152, 234, 454, 1, 4, 5, 197, 133, 362, 159, 386]


def cap_delta_magnitude(
    delta: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_px: Optional[float] = None,
    percentile: float = 98.0,
    region: str = "weighted_only",
    eps: float = 1e-6
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Cap per-point delta magnitude to avoid extreme spikes.
    """
    delta = np.asarray(delta, dtype=np.float32)
    if weights is None:
        weights = np.ones(delta.shape[0], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    mag = np.linalg.norm(delta, axis=1)
    if region == "weighted_only":
        mask = weights > 0
    else:
        mask = np.ones_like(weights, dtype=bool)

    mag_valid = mag[mask]
    if mag_valid.size == 0:
        return delta, 0.0, {
            "before_mean": 0.0,
            "before_p98": 0.0,
            "before_max": 0.0,
            "after_mean": 0.0,
            "after_p98": 0.0,
            "after_max": 0.0,
            "cap_value": 0.0,
        }

    if max_px is not None:
        cap_value = float(max_px)
    else:
        cap_value = float(np.percentile(mag_valid, percentile))

    before_mean = float(np.mean(mag_valid))
    before_p98 = float(np.percentile(mag_valid, 98))
    before_max = float(np.max(mag_valid))

    delta_capped = delta.copy()
    over = mag > cap_value
    if np.any(over):
        scale = cap_value / (mag + eps)
        scale = np.where(over, scale, 1.0)
        delta_capped = delta * scale[:, None]

    after_mag = np.linalg.norm(delta_capped, axis=1)
    after_valid = after_mag[mask]
    stats = {
        "before_mean": before_mean,
        "before_p98": before_p98,
        "before_max": before_max,
        "after_mean": float(np.mean(after_valid)) if after_valid.size else 0.0,
        "after_p98": float(np.percentile(after_valid, 98)) if after_valid.size else 0.0,
        "after_max": float(np.max(after_valid)) if after_valid.size else 0.0,
        "cap_value": cap_value,
    }
    return delta_capped, cap_value, stats


def build_knn_graph(L: np.ndarray, k: int = 8) -> np.ndarray:
    """Brute-force KNN graph on 468 points. Returns neighbors indices (N, k)."""
    L = np.asarray(L, dtype=np.float32)
    N = L.shape[0]
    neighbors = np.zeros((N, min(k, N - 1)), dtype=np.int32)
    for i in range(N):
        # Distance to all others
        diff = L - L[i]
        dist = np.sum(diff * diff, axis=1)
        dist[i] = np.inf  # exclude self
        nn = np.argsort(dist)
        neighbors[i] = nn[: neighbors.shape[1]]
    return neighbors


def smooth_delta_knn(
    L: np.ndarray,
    delta: np.ndarray,
    weights: Optional[np.ndarray] = None,
    k: int = 8,
    iters: int = 2,
    lam: float = 0.6,
    weight_aware: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Iterative Laplacian smoothing of delta using KNN graph.
    Returns smoothed delta and per-iteration RMS change.
    """
    L = np.asarray(L, dtype=np.float32)
    delta_smooth = np.asarray(delta, dtype=np.float32).copy()
    if weights is None:
        weights = np.ones(L.shape[0], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    neighbors = build_knn_graph(L, k=k)
    rms_history: List[float] = []

    for _ in range(max(iters, 0)):
        prev = delta_smooth.copy()
        for i in range(L.shape[0]):
            nb_idx = neighbors[i]
            nb = delta_smooth[nb_idx]
            avg = np.mean(nb, axis=0) if nb.size else np.zeros(2, dtype=np.float32)
            if weight_aware and weights[i] <= 0:
                delta_smooth[i] = np.zeros(2, dtype=np.float32)
            else:
                delta_smooth[i] = (1.0 - lam) * delta_smooth[i] + lam * avg
        rms = float(np.sqrt(np.mean((delta_smooth - prev) ** 2)))
        rms_history.append(rms)
    return delta_smooth, rms_history


def apply_anchor_zeroing(
    delta: np.ndarray,
    anchor_idx: Optional[List[int]] = None,
    strength: float = 0.95
) -> np.ndarray:
    """Push anchor deltas toward zero by (1 - strength)."""
    delta_out = np.asarray(delta, dtype=np.float32).copy()
    idx_list = anchor_idx if anchor_idx is not None else DEFAULT_ANCHOR_IDX
    for idx in idx_list:
        if 0 <= idx < delta_out.shape[0]:
            delta_out[idx] *= max(0.0, 1.0 - strength)
    return delta_out


def make_soft_face_effect_mask(
    img_shape: Tuple[int, int],
    L_out: np.ndarray,
    region_idxs: Optional[Dict[str, List[int]]] = None,
    sigma_px: float = 25.0,
    forehead_fade: bool = True,
    forehead_yfrac: float = 0.22,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Build a soft effect mask emphasizing lips/jaw/cheeks and fading to forehead.
    """
    H, W = img_shape
    if L_out is None or L_out.shape[0] < 468:
        return np.zeros((H, W), dtype=np.float32)

    mask = np.zeros((H, W), dtype=np.float32)
    regions = region_idxs or {}
    lips_idx = regions.get("lips", FACEMESH_LIPS_IDX)
    jaw_idx = regions.get("face_oval", FACEMESH_FACE_OVAL_IDX)
    cheeks_idx = regions.get("cheeks", [234, 454])

    def _accumulate(idx_list: List[int], value: float, radius: int = 8):
        for idx in idx_list:
            if 0 <= idx < 468:
                x, y = L_out[idx]
                cv2.circle(mask, (int(round(x)), int(round(y))), radius, float(value), -1, lineType=cv2.LINE_AA)

    _accumulate(lips_idx, 1.0, radius=8)
    _accumulate(jaw_idx, 0.7, radius=6)
    _accumulate(cheeks_idx, 0.5, radius=10)

    mask = np.clip(mask, 0.0, 1.0)
    if sigma_px > 0:
        ksize = max(3, int(2 * round(sigma_px) + 1))
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)

    if forehead_fade and forehead_yfrac > 0:
        ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)
        fade = np.clip((ramp - forehead_yfrac) / (1.0 - forehead_yfrac + 1e-6), 0.0, 1.0)
        mask = mask * fade[:, None]

    mask = np.clip(mask, min_val, max_val)
    return mask.astype(np.float32)


def make_mouth_only_mask(
    img_shape: Tuple[int, int],
    L_out: np.ndarray,
    lips_idx: Optional[List[int]] = None,
    radius_px: int = 90,
    sigma_px: float = 25.0
) -> np.ndarray:
    H, W = img_shape
    if L_out is None or L_out.shape[0] < 468:
        return np.zeros((H, W), dtype=np.float32)
    lips_idx = lips_idx or FACEMESH_LIPS_IDX
    lips_pts = np.array([L_out[i] for i in lips_idx if 0 <= i < 468], dtype=np.float32)
    if lips_pts.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    center = np.mean(lips_pts, axis=0)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), radius_px, 1.0, -1, lineType=cv2.LINE_AA)
    if sigma_px > 0:
        ksize = max(3, int(2 * round(sigma_px) + 1))
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)
    return np.clip(mask, 0.0, 1.0)


def build_face_mask_from_hull(
    img_shape: Tuple[int, int],
    L_out: np.ndarray,
    hull_idx: Optional[List[int]] = None,
    dilate: int = 12,
    erode: int = 0,
    blur: int = 11
) -> np.ndarray:
    H, W = img_shape
    if L_out is None or L_out.shape[0] < 468:
        return np.zeros((H, W), dtype=np.float32)
    hull_idx = hull_idx or FACEMESH_FACE_OVAL_IDX
    pts = np.array([L_out[i] for i in hull_idx if 0 <= i < 468], dtype=np.float32)
    if pts.shape[0] < 3:
        return np.zeros((H, W), dtype=np.float32)

    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.float32)).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    if blur > 0 and blur % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def composite_face_only(
    original_img: np.ndarray,
    warped_img: np.ndarray,
    face_mask: np.ndarray,
    effect_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    if original_img is None or warped_img is None:
        return original_img
    face_mask = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
    total_mask = face_mask
    if effect_mask is not None:
        effect_mask = np.clip(effect_mask.astype(np.float32), 0.0, 1.0)
        total_mask = total_mask * effect_mask
    total_mask_3c = np.repeat(total_mask[:, :, None], 3, axis=2)
    out = warped_img.astype(np.float32) * total_mask_3c + original_img.astype(np.float32) * (1.0 - total_mask_3c)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_guardrails(
    L_out: np.ndarray,
    delta_out: np.ndarray,
    weights: np.ndarray,
    img_shape: Tuple[int, int],
    regions: Optional[Dict[str, List[int]]],
    args: Dict[str, object],
    debug_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Apply guardrails pipeline. Returns (delta_guarded, weights_guarded, debug_dict).
    """
    delta_guarded = np.asarray(delta_out, dtype=np.float32).copy()
    weights_guarded = np.asarray(weights, dtype=np.float32).copy()
    H, W = img_shape
    debug: Dict[str, object] = {"warnings": []}

    def _save_vec(field: np.ndarray, name: str):
        if debug_dir is None:
            return
        os.makedirs(debug_dir, exist_ok=True)
        # Simple visualization: magnitude heatmap
        mag = np.linalg.norm(field, axis=1)
        mag_img = np.zeros((H, W), dtype=np.float32)
        for i in range(min(468, field.shape[0])):
            x, y = L_out[i]
            cv2.circle(mag_img, (int(round(x)), int(round(y))), 3, float(mag[i]), -1, lineType=cv2.LINE_AA)
        if mag.max() > 0:
            mag_img = mag_img / mag.max()
        mag_img = (mag_img * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), mag_img)

    # Early mouth-only mode
    mouth_only = bool(args.get("guard_mouth_only", False))
    if mouth_only:
        lips_idx = regions.get("lips") if regions else FACEMESH_LIPS_IDX
        keep = np.zeros_like(weights_guarded, dtype=bool)
        for idx in lips_idx:
            if 0 <= idx < keep.shape[0]:
                keep[idx] = True
        # Stabilizers: corners and anchors
        for idx in [61, 291, 0, 17]:
            if 0 <= idx < keep.shape[0]:
                keep[idx] = True
        for idx in DEFAULT_ANCHOR_IDX:
            if 0 <= idx < keep.shape[0]:
                keep[idx] = True
        weights_guarded = np.where(keep, weights_guarded, 0.0)
        delta_guarded = np.where(keep[:, None], delta_guarded, 0.0)
        alpha = float(args.get("facemesh_warp_alpha", 1.0))
        if alpha > 0.6:
            debug["warnings"].append("Mouth-only mode recommends alpha <= 0.6")

    if args.get("guard_cap_after_align", True):
        delta_guarded, cap_value, cap_stats = cap_delta_magnitude(
            delta_guarded,
            weights=weights_guarded,
            max_px=args.get("guard_max_delta_px"),
            percentile=float(args.get("guard_cap_percentile", 98.0)),
            region=str(args.get("guard_cap_region", "weighted_only")),
        )
        debug.update({"cap_value": cap_value, "cap_stats": cap_stats})
        _save_vec(delta_guarded, "delta_after_cap")

    if args.get("guard_smooth_delta", True):
        delta_guarded, rms_hist = smooth_delta_knn(
            L_out,
            delta_guarded,
            weights=weights_guarded,
            k=int(args.get("guard_knn_k", 8)),
            iters=int(args.get("guard_smooth_iterations", 2)),
            lam=float(args.get("guard_smooth_lambda", 0.6)),
            weight_aware=True,
        )
        debug["smooth_rms"] = rms_hist
        _save_vec(delta_guarded, "delta_after_smooth")

    if args.get("guard_zero_anchor", True):
        delta_guarded = apply_anchor_zeroing(
            delta_guarded,
            anchor_idx=args.get("guard_anchor_idx", None),
            strength=float(args.get("guard_anchor_strength", 0.95))
        )
        _save_vec(delta_guarded, "delta_after_anchor")

    # Optional second cap after smoothing/anchors
    delta_guarded, cap_value_final, cap_stats_final = cap_delta_magnitude(
        delta_guarded,
        weights=weights_guarded,
        max_px=args.get("guard_max_delta_px"),
        percentile=float(args.get("guard_cap_percentile", 98.0)),
        region=str(args.get("guard_cap_region", "weighted_only")),
    )
    debug["cap_value_final"] = cap_value_final
    debug["cap_stats_final"] = cap_stats_final
    _save_vec(delta_guarded, "delta_after_final_cap")

    # Effect mask
    effect_mask = None
    if args.get("guard_softmask", True):
        effect_mask = make_soft_face_effect_mask(
            (H, W),
            L_out,
            regions,
            sigma_px=float(args.get("guard_softmask_sigma", 25.0)),
            forehead_fade=bool(args.get("guard_softmask_forehead_fade", True)),
            forehead_yfrac=float(args.get("guard_softmask_forehead_yfrac", 0.22)),
            min_val=float(args.get("guard_softmask_min", 0.0)),
            max_val=float(args.get("guard_softmask_max", 1.0)),
        )
    if mouth_only:
        effect_mask = make_mouth_only_mask(
            (H, W),
            L_out,
            lips_idx=regions.get("lips") if regions else FACEMESH_LIPS_IDX,
            radius_px=int(args.get("guard_mouth_radius_px", 90)),
            sigma_px=float(args.get("guard_softmask_sigma", 25.0))
        )

    # Face mask
    face_mask = None
    if args.get("guard_face_mask", True):
        face_mask = build_face_mask_from_hull(
            (H, W),
            L_out,
            hull_idx=FACEMESH_FACE_OVAL_IDX,
            dilate=int(args.get("guard_face_mask_dilate", 12)),
            erode=int(args.get("guard_face_mask_erode", 0)),
            blur=int(args.get("guard_face_mask_blur", 11)),
        )

    # Debug saves
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        _save_vec(delta_out, "delta_before")
        _save_vec(delta_guarded, "delta_after_final")
        if effect_mask is not None:
            cv2.imwrite(os.path.join(debug_dir, "effect_mask.png"), (np.clip(effect_mask, 0, 1) * 255).astype(np.uint8))
        if face_mask is not None:
            cv2.imwrite(os.path.join(debug_dir, "face_mask.png"), (np.clip(face_mask, 0, 1) * 255).astype(np.uint8))
        if effect_mask is not None and face_mask is not None:
            composite_mask = np.clip(effect_mask * face_mask, 0, 1)
            cv2.imwrite(os.path.join(debug_dir, "composite_mask.png"), (composite_mask * 255).astype(np.uint8))
        with open(os.path.join(debug_dir, "guards_summary.json"), "w") as f:
            json.dump(debug, f, indent=2)

    debug.update({
        "effect_mask": effect_mask,
        "face_mask": face_mask,
    })
    return delta_guarded, weights_guarded, debug
