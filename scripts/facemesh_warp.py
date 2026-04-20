"""
FaceMesh-based post-process warp for asymmetry transfer

Phase 3-5: Deformation field estimation and image warping using Thin Plate Spline (TPS).

This module applies donor asymmetry to LivePortrait output via dense landmark-based warping.

Key components:
- Procrustes similarity fitting (Umeyama)
- Thin Plate Spline (TPS) deformation field
- Image warping via inverse mapping and cv2.remap
- Folding detection and validation
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import math

from scripts.facemesh_guards import apply_guardrails, composite_face_only
from scripts.facemesh_landmarks import FACEMESH_LIPS_IDX, FACEMESH_ANCHOR_IDX


# =====================================================================
# SIMILARITY TRANSFORM (UMEYAMA / PROCRUSTES)
# =====================================================================

def estimate_similarity_umeyama(
    X: np.ndarray,
    Y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate similarity transform (scale + rotation) from correspondence.

    Solves: minimize Σ w_i || (sR @ X_i + t) - Y_i ||^2

    Args:
        X: Source points, shape (N, 2), float32
        Y: Target points, shape (N, 2), float32
        weights: Optional weights, shape (N,), float32. If None, use uniform weights.

    Returns:
        sR: (2, 2) matrix = scale * rotation
        t: (2,) translation vector

    Note:
        For transforming delta vectors (no translation), apply only sR.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    if X.shape[0] < 2:
        # Degenerate case: return identity
        return np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

    if weights is None:
        weights = np.ones(X.shape[0], dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

    # Normalize weights
    weights = weights / (weights.sum() + 1e-8)

    # Weighted centroids
    mean_X = (X * weights[:, None]).sum(axis=0)
    mean_Y = (Y * weights[:, None]).sum(axis=0)

    # Center points
    X_c = X - mean_X
    Y_c = Y - mean_Y

    # Weighted covariance
    H = np.zeros((2, 2), dtype=np.float32)
    for i in range(X.shape[0]):
        H += weights[i] * np.outer(X_c[i], Y_c[i])

    # SVD for rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1, no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Verify rotation matrix validity (assertion for robustness)
    det_R = np.linalg.det(R)
    assert np.abs(det_R - 1.0) < 1e-4, f"Rotation matrix determinant invalid: det(R)={det_R:.6f}, expected 1.0"

    # Scale estimation
    numerator = (Y_c * (X_c @ R.T)).sum()
    denominator = (X_c * X_c).sum()
    scale = numerator / (denominator + 1e-8)

    sR = scale * R
    t = mean_Y - (mean_X @ sR.T)

    return sR.astype(np.float32), t.astype(np.float32)


def align_delta_to_output(
    delta_d: np.ndarray,
    sR: np.ndarray
) -> np.ndarray:
    """
    Align delta vectors from donor to output frame via similarity transform.

    Apply only sR (rotation + scale), not translation.

    Args:
        delta_d: Donor delta vectors, shape (468, 2), float32
        sR: Similarity matrix (scale * rotation), shape (2, 2), float32

    Returns:
        delta_out: Aligned delta vectors, shape (468, 2), float32
    """
    delta_d = np.asarray(delta_d, dtype=np.float32)
    sR = np.asarray(sR, dtype=np.float32)

    # delta_out[i] = sR @ delta_d[i]
    delta_out = delta_d @ sR.T

    return delta_out.astype(np.float32)


# =====================================================================
# CONTROL POINT SELECTION
# =====================================================================

def select_control_points(
    L_out: np.ndarray,
    delta_out: np.ndarray,
    weights: np.ndarray,
    vis: Optional[np.ndarray] = None,
    roi_idx: Optional[List[int]] = None,
    lock_boundary: bool = True,
    image_shape: Optional[Tuple[int, int]] = None,
    alpha: float = 1.0,
    vis_threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select control points for warp and build source/target correspondences.

    Args:
        L_out: Landmark positions in output frame, shape (468, 2), float32
        delta_out: Aligned delta vectors, shape (468, 2), float32
        weights: Landmark weights, shape (468,), float32
        vis: Visibility scores, shape (468,), float32. Optional.
        roi_idx: Indices to include (if None, use where weights > 0)
        lock_boundary: If True, add boundary anchor points with zero displacement
        image_shape: (H, W) for boundary and bounds checking
        alpha: Scale factor for displacements
        vis_threshold: Minimum visibility to include point
        verbose: Print diagnostics

    Returns:
        src_pts: Source control points, shape (M, 2), float32
        dst_pts: Target control points, shape (M, 2), float32
        sel_idx: Indices of selected landmarks (for tracking)
    """
    L_out = np.asarray(L_out, dtype=np.float32)
    delta_out = np.asarray(delta_out, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    H, W = image_shape if image_shape else (L_out[:, 1].max() + 1, L_out[:, 0].max() + 1)
    H, W = int(H), int(W)

    # Select landmark-based control points
    if roi_idx is None:
        roi_idx = np.where(weights > 0)[0].tolist()

    sel_idx = []
    src_list = []
    dst_list = []

    for idx in roi_idx:
        if idx < 0 or idx >= 468:
            continue

        # Check validity
        if np.isnan(L_out[idx]).any():
            continue

        x, y = L_out[idx]
        if x < 0 or x > W - 1 or y < 0 or y > H - 1:
            continue

        if vis is not None and vis[idx] < vis_threshold:
            continue

        src = L_out[idx].copy()
        dst = src + alpha * delta_out[idx]

        # Clamp dst to bounds
        dst[0] = np.clip(dst[0], 0, W - 1)
        dst[1] = np.clip(dst[1], 0, H - 1)

        src_list.append(src)
        dst_list.append(dst)
        sel_idx.append(idx)

    # Add boundary anchors (zero displacement)
    if lock_boundary:
        boundary_pts = [
            [0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1],
            [W / 2, 0], [W / 2, H - 1],
            [0, H / 2], [W - 1, H / 2]
        ]
        for pt in boundary_pts:
            src_list.append(np.array(pt, dtype=np.float32))
            dst_list.append(np.array(pt, dtype=np.float32))

    # Convert to arrays
    src_pts = np.array(src_list, dtype=np.float32)
    dst_pts = np.array(dst_list, dtype=np.float32)
    sel_idx = np.array(sel_idx, dtype=np.int32)

    if verbose:
        print(f"[FaceMesh Warp] Selected {len(sel_idx)} landmark + {8 if lock_boundary else 0} boundary control points")

    return src_pts, dst_pts, sel_idx


def deduplicate_control_points(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    tolerance_px: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove near-duplicate control points that could destabilize TPS.

    TPS solver can become numerically unstable if many control points
    are duplicates or very close together in SOURCE space.

    Args:
        src_pts: Source control points, shape (N, 2), float32
        dst_pts: Target control points, shape (N, 2), float32
        tolerance_px: Merge points closer than this in SOURCE space (default 0.1px)
        verbose: Print diagnostics

    Returns:
        src_pts_dedup: Deduplicated source points
        dst_pts_dedup: Deduplicated target points
    """
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    if len(src_pts) == 0:
        return src_pts, dst_pts

    # Deduplicate based on SOURCE space proximity only
    # (the real problem is duplicate SOURCE points, not mismatch)
    src_pts_rounded = np.round(src_pts / tolerance_px) * tolerance_px

    # Find unique source points
    _, unique_idx = np.unique(src_pts_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    n_original = len(src_pts)
    n_unique = len(unique_idx)
    n_removed = n_original - n_unique

    if verbose and n_removed > 0:
        print(f"[TPS Dedup] Removed {n_removed} near-duplicate source control points ({n_original} → {n_unique})")

    return src_pts[unique_idx], dst_pts[unique_idx]


# =====================================================================
# THIN PLATE SPLINE (TPS)
# =====================================================================

def tps_fit(
    control_src: np.ndarray,
    control_dst: np.ndarray,
    reg: float = 1e-3,
    verbose: bool = False
) -> Dict:
    """
    Fit Thin Plate Spline from src to dst.

    We fit TPS in INVERSE mapping: dst -> src
    This is correct for cv2.remap usage.

    Args:
        control_src: Source control points, shape (N, 2), float32
        control_dst: Target control points, shape (N, 2), float32
        reg: Regularization parameter (added to K matrix)
        verbose: Print diagnostics

    Returns:
        Dictionary with TPS parameters (for use in tps_warp_image)
    """
    control_src = np.asarray(control_src, dtype=np.float32)
    control_dst = np.asarray(control_dst, dtype=np.float32)

    N = control_src.shape[0]

    if N < 3:
        raise ValueError(f"Need at least 3 control points, got {N}")

    # TPS kernel function
    def U(r_sq):
        eps = 1e-6
        return r_sq * np.log(r_sq + eps)

    # Build K matrix: pairwise distances
    K = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                K[i, j] = 0
            else:
                r_sq = np.sum((control_dst[i] - control_dst[j]) ** 2)
                K[i, j] = U(r_sq)

    # Build P matrix: affine part [1, x, y]
    P = np.ones((N, 3), dtype=np.float32)
    P[:, 1:] = control_dst

    # Augmented system
    L = np.vstack([
        np.hstack([K + reg * np.eye(N), P]),
        np.hstack([P.T, np.zeros((3, 3), dtype=np.float32)])
    ])

    # Fit x and y separately
    v_x = np.hstack([control_src[:, 0], np.zeros(3, dtype=np.float32)])
    v_y = np.hstack([control_src[:, 1], np.zeros(3, dtype=np.float32)])

    try:
        w_x = np.linalg.solve(L, v_x)
        w_y = np.linalg.solve(L, v_y)
    except np.linalg.LinAlgError:
        if verbose:
            print("[TPS] Singular matrix, using least squares")
        w_x = np.linalg.lstsq(L, v_x, rcond=None)[0]
        w_y = np.linalg.lstsq(L, v_y, rcond=None)[0]

    return {
        "control_dst": control_dst.astype(np.float32),
        "w_x": w_x.astype(np.float32),
        "w_y": w_y.astype(np.float32),
        "N": N,
    }


def tps_warp_image(
    img: np.ndarray,
    tps_model: Dict,
    out_shape: Tuple[int, int],
    grid_step: int = 2,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp image using fitted TPS model via inverse mapping.

    Args:
        img: Input image, shape (H, W, 3), uint8
        tps_model: TPS model from tps_fit()
        out_shape: Output shape (H, W)
        grid_step: Coarse-to-fine step size (1 for full resolution)
        verbose: Print diagnostics

    Returns:
        warped_img: Warped image, shape (H, W, 3), uint8
        disp_field: Displacement field, shape (H, W, 2), float32
                    disp = (x_out - x_src, y_out - y_src)
    """
    H_out, W_out = out_shape

    control_dst = tps_model["control_dst"]
    w_x = tps_model["w_x"]
    w_y = tps_model["w_y"]
    N = tps_model["N"]

    # Build coarse grid
    if grid_step > 1:
        y_coarse = np.arange(0, H_out, grid_step, dtype=np.float32)
        x_coarse = np.arange(0, W_out, grid_step, dtype=np.float32)
    else:
        y_coarse = np.arange(H_out, dtype=np.float32)
        x_coarse = np.arange(W_out, dtype=np.float32)

    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse)
    pts_coarse = np.stack([xx_coarse, yy_coarse], axis=-1).reshape(-1, 2)

    # Evaluate TPS at coarse grid
    def U(r_sq):
        eps = 1e-6
        return r_sq * np.log(r_sq + eps)

    src_coarse = np.zeros((pts_coarse.shape[0], 2), dtype=np.float32)

    for i, pt in enumerate(pts_coarse):
        # Affine part
        src_x = w_x[N] + w_x[N + 1] * pt[0] + w_x[N + 2] * pt[1]
        src_y = w_y[N] + w_y[N + 1] * pt[0] + w_y[N + 2] * pt[1]

        # Radial basis
        for j in range(N):
            r_sq = np.sum((pt - control_dst[j]) ** 2)
            src_x += w_x[j] * U(r_sq)
            src_y += w_y[j] * U(r_sq)

        src_coarse[i] = [src_x, src_y]

    # Reshape back to grid
    src_grid = src_coarse.reshape(len(y_coarse), len(x_coarse), 2)

    # Upsample to full resolution if needed
    if grid_step > 1:
        src_x = cv2.resize(src_grid[..., 0], (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        src_y = cv2.resize(src_grid[..., 1], (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        src_full = np.stack([src_x, src_y], axis=-1)
    else:
        src_full = src_grid

    # Build remap for cv2.remap
    map_x = src_full[..., 0].astype(np.float32)
    map_y = src_full[..., 1].astype(np.float32)

    # Clamp to image bounds
    map_x = np.clip(map_x, 0, img.shape[1] - 1)
    map_y = np.clip(map_y, 0, img.shape[0] - 1)

    # Apply remap
    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Compute displacement field
    # Correct coordinate grid: xx_full has x coords, yy_full has y coords
    xx_full, yy_full = np.meshgrid(np.arange(W_out), np.arange(H_out), indexing='xy')
    disp_field = np.stack([xx_full - map_x, yy_full - map_y], axis=-1).astype(np.float32)

    return warped.astype(np.uint8), disp_field.astype(np.float32)


# =====================================================================
# FOLDING DETECTION
# =====================================================================

def detect_folding(
    disp_field: np.ndarray,
    threshold: float = 0.0,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Detect folds/tears by examining Jacobian determinant of inverse mapping.

    Args:
        disp_field: Displacement field, shape (H, W, 2), float32
        threshold: Warn if fold fraction > this threshold
        verbose: Print diagnostics

    Returns:
        fold_mask: Binary mask where det(J) <= 0, shape (H, W), uint8
        fold_fraction: Fraction of pixels with folds
    """
    H, W = disp_field.shape[:2]

    # Compute Jacobian determinant via finite differences
    # J = d(disp)/d(x,y)
    # det(J) = J_xx * J_yy - J_xy * J_yx

    fold_mask = np.zeros((H, W), dtype=np.uint8)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            # Partial derivatives of displacement
            ddisp_dx = disp_field[y, x + 1] - disp_field[y, x - 1]
            ddisp_dy = disp_field[y + 1, x] - disp_field[y - 1, x]

            # Jacobian
            J_xx = 1.0 + ddisp_dx[0] / 2.0
            J_yy = 1.0 + ddisp_dy[1] / 2.0
            J_xy = ddisp_dx[1] / 2.0
            J_yx = ddisp_dy[0] / 2.0

            det_J = J_xx * J_yy - J_xy * J_yx

            if det_J <= threshold:
                fold_mask[y, x] = 1

    fold_fraction = float(fold_mask.sum()) / float((H - 2) * (W - 2))

    if verbose:
        print(f"[TPS Folding] {fold_fraction * 100:.2f}% pixels with folds (det_J <= {threshold})")

    return fold_mask, fold_fraction


# =====================================================================
# DEBUG VISUALIZATION
# =====================================================================

def draw_control_points_overlay(
    img_rgb: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_path: str,
    sel_idx: Optional[np.ndarray] = None,
    label_landmarks: bool = True
) -> None:
    """
    Draw control points before/after on image.

    Args:
        img_rgb: RGB image, shape (H, W, 3), uint8
        src_pts: Source control points, shape (N, 2), float32
        dst_pts: Target control points, shape (N, 2), float32
        out_path: Output file path
        sel_idx: Indices of landmark control points (vs boundary)
        label_landmarks: Label sanity-check indices
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    H, W = canvas.shape[:2]

    # Draw source points (cyan)
    for pt in src_pts:
        pt_int = tuple(np.round(pt).astype(np.int32))
        pt_int = (np.clip(pt_int[0], 0, W - 1), np.clip(pt_int[1], 0, H - 1))
        cv2.circle(canvas_bgr, pt_int, 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    # Draw destination points (magenta)
    for pt in dst_pts:
        pt_int = tuple(np.round(pt).astype(np.int32))
        pt_int = (np.clip(pt_int[0], 0, W - 1), np.clip(pt_int[1], 0, H - 1))
        cv2.circle(canvas_bgr, pt_int, 2, (255, 0, 255), -1, lineType=cv2.LINE_AA)

    # Draw arrows from src to dst for landmarks
    if sel_idx is not None:
        for i, idx in enumerate(sel_idx):
            if i < len(src_pts):
                src = tuple(np.round(src_pts[i]).astype(np.int32))
                dst = tuple(np.round(dst_pts[i]).astype(np.int32))
                src = (np.clip(src[0], 0, W - 1), np.clip(src[1], 0, H - 1))
                dst = (np.clip(dst[0], 0, W - 1), np.clip(dst[1], 0, H - 1))
                cv2.arrowedLine(canvas_bgr, src, dst, (100, 150, 255), 1, tipLength=0.2)

    cv2.putText(canvas_bgr, "Cyan: src  Magenta: dst", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas_bgr)


def draw_displacement_heatmap(
    img_rgb: np.ndarray,
    disp_field: np.ndarray,
    out_path: str
) -> None:
    """
    Draw displacement magnitude as heatmap.

    Args:
        img_rgb: RGB image, shape (H, W, 3), uint8
        disp_field: Displacement field, shape (H, W, 2), float32
        out_path: Output file path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute magnitude
    mag = np.linalg.norm(disp_field, axis=-1)

    # Normalize to 0-255
    mag_max = np.percentile(mag, 95)
    mag_norm = np.clip(mag / (mag_max + 1e-8), 0, 1) * 255

    # Apply colormap
    mag_uint8 = mag_norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)

    # Overlay on image
    alpha = 0.5
    overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1 - alpha,
                              heatmap, alpha, 0)

    cv2.imwrite(out_path, overlay)


def draw_grid_warp_preview(
    img_before: np.ndarray,
    img_after: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_path: str,
    grid_spacing: int = 50
) -> None:
    """
    Draw grid deformation preview.

    Args:
        img_before: Before warp image, shape (H, W, 3), uint8
        img_after: After warp image, shape (H, W, 3), uint8
        src_pts: Source control points
        dst_pts: Target control points
        out_path: Output file path
        grid_spacing: Grid line spacing in pixels
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    H, W = img_before.shape[:2]
    canvas = np.hstack([img_before, img_after])
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Draw grid on left side
    for y in range(0, H, grid_spacing):
        cv2.line(canvas_bgr, (0, y), (W, y), (100, 100, 100), 1)
    for x in range(0, W, grid_spacing):
        cv2.line(canvas_bgr, (x, 0), (x, H), (100, 100, 100), 1)

    # Draw control points
    for i in range(min(len(src_pts), len(dst_pts))):
        if i < len(src_pts) and src_pts[i, 0] >= 0:  # landmark points only
            src = tuple(np.round(src_pts[i]).astype(np.int32))
            dst = tuple(np.round(dst_pts[i]).astype(np.int32))
            src = (np.clip(src[0], 0, W - 1), np.clip(src[1], 0, H - 1))
            dst = (np.clip(dst[0] + W, 0, 2 * W - 1), np.clip(dst[1], 0, H - 1))
            cv2.circle(canvas_bgr, src, 2, (0, 255, 0), -1)
            cv2.circle(canvas_bgr, dst, 2, (0, 0, 255), -1)

    cv2.imwrite(out_path, canvas_bgr)


# =====================================================================
# MAIN WARP PIPELINE
# =====================================================================

def apply_facemesh_warp(
    img_target_after: np.ndarray,
    L_d: np.ndarray,
    delta_d: np.ndarray,
    weights: np.ndarray,
    L_out: np.ndarray,
    L_out_vis: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    reg: float = 1e-3,
    grid_step: int = 2,
    lock_boundary: bool = True,
    verbose: bool = True,
    output_dir: str = "outputs/diagnostics/facemesh_warp",
    guards: bool = False,
    guard_args: Optional[Dict[str, object]] = None,
    regions_config: Optional[Dict[str, List[int]]] = None,
    guard_output_dir: Optional[str] = None,
    save_debug: bool = False
) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
    """
    Apply FaceMesh-based warp to target_after image.

    Args:
        img_target_after: Output image from LivePortrait, shape (H, W, 3), uint8
        L_d: Donor landmarks, shape (468, 2), float32
        delta_d: Donor asymmetry delta, shape (468, 2), float32
        weights: Landmark weights, shape (468,), float32
        L_out: Output landmarks, shape (468, 2), float32
        L_out_vis: Output landmark visibility, shape (468,), float32 (optional)
        alpha: Warp strength (0-1 typical)
        reg: TPS regularization
        grid_step: TPS coarse-to-fine step
        lock_boundary: Lock boundary points
        verbose: Print diagnostics
        output_dir: Directory for debug outputs
        save_debug: Save debug outputs (before.png, after.png) even when alpha=0

    Returns:
        ok: Success flag
        warped_img: Warped image or None
        summary: Summary dictionary with statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "ok": False,
        "alpha": alpha,
        "reg": reg,
        "grid_step": grid_step,
        "lock_boundary": lock_boundary,
        "guards_enabled": guards,
    }

    # Early exit for alpha=0 (no deformation)
    if alpha == 0:
        if verbose:
            print("[FaceMesh Warp] Alpha=0, returning input image unchanged")
        summary["ok"] = True
        summary["skipped_alpha_zero"] = True

        # Save debug outputs if requested
        if save_debug:
            os.makedirs(output_dir, exist_ok=True)
            before_path = os.path.join(output_dir, "before.png")
            after_path = os.path.join(output_dir, "after.png")

            cv2.imwrite(
                before_path,
                cv2.cvtColor(img_target_after, cv2.COLOR_RGB2BGR)
            )
            # after.png is identical copy when alpha=0
            cv2.imwrite(
                after_path,
                cv2.cvtColor(img_target_after, cv2.COLOR_RGB2BGR)
            )

            if verbose:
                print(f"[FaceMesh Warp] Saved debug outputs: {before_path}, {after_path}")

        return True, img_target_after.copy(), summary

    try:
        H, W = img_target_after.shape[:2]
        guard_dir = guard_output_dir or os.path.join(output_dir, "guards")

        if verbose:
            print(f"[FaceMesh Warp] Starting warp pipeline (H={H}, W={W})")

        # Fit similarity transform: donor → output
        fit_idx = []
        fit_weights = []

        # Use face oval + anchors for stable alignment
        stable_indices = list(range(468))  # Use all available
        fit_idx = [i for i in stable_indices if weights[i] > 0]
        fit_weights = weights[fit_idx]

        if len(fit_idx) < 3:
            print("[FaceMesh Warp] Insufficient stable landmarks for alignment")
            return False, None, summary

        X_fit = L_d[fit_idx]
        Y_fit = L_out[fit_idx]

        sR, t = estimate_similarity_umeyama(X_fit, Y_fit, weights=fit_weights)

        if verbose:
            print(f"[FaceMesh Warp] Similarity fit: scale≈{np.linalg.norm(sR) / np.sqrt(2):.4f}, t={t}")

        # Align delta vectors
        delta_out = align_delta_to_output(delta_d, sR)

        # Store sR and aligned delta in summary for later validation
        summary["sR"] = sR
        summary["delta_out_aligned"] = delta_out.copy()

        # Optional guardrails
        weights_for_warp = weights.copy()
        guard_debug: Dict[str, object] = {}
        guard_cfg = dict(guard_args or {})
        guard_cfg.setdefault("facemesh_warp_alpha", alpha)
        if guards:
            if verbose:
                print("[FaceMesh Warp] Applying guardrails...")
            try:
                delta_guarded, weights_guarded, guard_debug = apply_guardrails(
                    L_out,
                    delta_out,
                    weights_for_warp,
                    (H, W),
                    regions_config,
                    guard_cfg,
                    debug_dir=guard_dir if guard_cfg.get("facemesh_guard_debug", False) else None
                )
                delta_out = delta_guarded
                weights_for_warp = weights_guarded
                summary["guard_effect_mask"] = guard_debug.get("effect_mask") is not None
                summary["guard_face_mask"] = guard_debug.get("face_mask") is not None
            except Exception as guard_e:
                if verbose:
                    print(f"[FaceMesh Warp] Guardrails failed: {guard_e}")
                summary["guard_error"] = str(guard_e)

        # Mouth-only mode: direct control point selection from lips
        mouth_only = guard_cfg.get("guard_mouth_only", False) if guards else False

        if mouth_only:
            # Direct selection from lip landmarks only
            lips_idx = regions_config.get("lips", FACEMESH_LIPS_IDX) if regions_config else FACEMESH_LIPS_IDX
            lips_idx = [i for i in lips_idx if 0 <= i < L_out.shape[0]]

            if len(lips_idx) < 10:
                print(f"[FaceMesh Warp] Mouth-only: lips_idx too small ({len(lips_idx)}), cannot warp")
                return False, None, summary

            # Debug: print delta magnitudes on lips
            lip_mag = np.linalg.norm(delta_out[lips_idx], axis=1)
            if verbose:
                print(f"[FaceMesh Warp DEBUG] lip |delta_out| mean/max: {float(lip_mag.mean()):.4f} / {float(lip_mag.max()):.4f} px")

            src_pts = L_out[lips_idx].astype(np.float32)
            dst_pts = (L_out[lips_idx] + alpha * delta_out[lips_idx]).astype(np.float32)
            sel_idx = np.array(lips_idx, dtype=np.int32)

            # Optional: add anchors with zero displacement for stability
            anchors = regions_config.get("anchors", FACEMESH_ANCHOR_IDX) if regions_config else FACEMESH_ANCHOR_IDX
            anchors = [i for i in anchors if 0 <= i < L_out.shape[0]]
            if len(anchors) > 0:
                src_pts = np.vstack([src_pts, L_out[anchors].astype(np.float32)])
                dst_pts = np.vstack([dst_pts, L_out[anchors].astype(np.float32)])  # zero delta

            if verbose:
                print(f"[FaceMesh Warp] Mouth-only: using {len(lips_idx)} lip control points + {len(anchors)} anchors")
        else:
            # Standard control point selection
            src_pts, dst_pts, sel_idx = select_control_points(
                L_out, delta_out, weights_for_warp,
                vis=L_out_vis,
                lock_boundary=lock_boundary,
                image_shape=(H, W),
                alpha=alpha,
                verbose=verbose
            )

        if len(src_pts) < 6:
            print("[FaceMesh Warp] Insufficient control points")
            return False, None, summary

        # Fit TPS (inverse mapping: dst → src)
        # First deduplicate to avoid TPS solver instability
        src_pts_dedup, dst_pts_dedup = deduplicate_control_points(
            src_pts, dst_pts, tolerance_px=0.05, verbose=verbose
        )

        if verbose:
            print(f"[FaceMesh Warp] Fitting TPS with {len(src_pts_dedup)} deduplicated control points...")

        tps_model = tps_fit(src_pts_dedup, dst_pts_dedup, reg=reg, verbose=verbose)

        # Apply warp
        if verbose:
            print("[FaceMesh Warp] Warping image...")

        warped_img, disp_field = tps_warp_image(
            img_target_after, tps_model, (H, W),
            grid_step=grid_step, verbose=verbose
        )

        # Displacement sanity checks
        if not np.isfinite(disp_field).all() or np.max(np.abs(disp_field)) > 1e6:
            print("[FaceMesh Warp] Invalid displacement field detected, skipping warp")
            summary["error"] = "invalid_disp_field"
            return False, None, summary

        # Detect folding
        fold_mask, fold_frac = detect_folding(disp_field, threshold=0.0, verbose=verbose)
        summary["fold_fraction"] = fold_frac
        summary["num_control_points"] = len(src_pts)
        summary["guard_warnings"] = guard_debug.get("warnings", [])

        # Face-only composite if requested
        effect_mask = guard_debug.get("effect_mask") if guards else None
        face_mask = guard_debug.get("face_mask") if guards else None
        if guards and guard_cfg.get("guard_warp_face_only", True) and face_mask is not None:
            warped_img = composite_face_only(img_target_after, warped_img, face_mask, effect_mask)
            summary["guard_face_only"] = True
        else:
            summary["guard_face_only"] = False

        # Save debug outputs (output_dir already created earlier)
        before_path = os.path.join(output_dir, "before.png")
        after_path = os.path.join(output_dir, "after.png")

        cv2.imwrite(
            before_path,
            cv2.cvtColor(img_target_after, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            after_path,
            cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
        )

        draw_control_points_overlay(
            img_target_after, src_pts, dst_pts,
            os.path.join(output_dir, "control_points_overlay.png"),
            sel_idx=sel_idx
        )

        draw_displacement_heatmap(
            img_target_after, disp_field,
            os.path.join(output_dir, "warp_magnitude_heatmap.png")
        )

        draw_grid_warp_preview(
            img_target_after, warped_img, src_pts, dst_pts,
            os.path.join(output_dir, "grid_warp_preview.png")
        )

        if fold_frac > 0.001:
            fold_img = (fold_mask * 255).astype(np.uint8)
            fold_img_color = cv2.applyColorMap(fold_img, cv2.COLORMAP_HOT)
            cv2.imwrite(
                os.path.join(output_dir, "fold_mask.png"),
                fold_img_color
            )

        summary["ok"] = True
        summary["warp_succeeded"] = True

        if verbose:
            print(f"[FaceMesh Warp] ✓ Warp complete. Folding: {fold_frac * 100:.2f}%")

        return True, warped_img, summary

    except Exception as e:
        print(f"[FaceMesh Warp] Error: {e}")
        import traceback
        traceback.print_exc()
        summary["error"] = str(e)
        return False, None, summary


def validate_warp(
    img_warped: np.ndarray,
    L_out_target: np.ndarray,
    sel_idx: np.ndarray,
    extractor,
    output_dir: str = "outputs/diagnostics/facemesh_warp",
    verbose: bool = True
) -> Dict:
    """
    Validate warp by running FaceMesh on warped image.

    Args:
        img_warped: Warped image, shape (H, W, 3), uint8
        L_out_target: Target landmark positions, shape (468, 2), float32
        sel_idx: Indices of selected landmarks
        extractor: FaceMeshLandmarkExtractor instance
        output_dir: Debug output directory
        verbose: Print diagnostics

    Returns:
        Validation summary dictionary
    """
    validation = {
        "success": False,
        "mean_error_px": None,
        "max_error_px": None,
    }

    try:
        ok_warp, L_warp, _ = extractor.extract(img_warped)

        if not ok_warp:
            print("[FaceMesh Validate] Failed to extract landmarks on warped image")
            return validation

        # Compute error on control landmarks
        errors = []
        for idx in sel_idx:
            if 0 <= idx < 468:
                err = np.linalg.norm(L_warp[idx] - L_out_target[idx])
                errors.append(err)

        if len(errors) > 0:
            errors = np.array(errors)
            validation["mean_error_px"] = float(errors.mean())
            validation["max_error_px"] = float(errors.max())
            validation["success"] = True

            if verbose:
                print(f"[FaceMesh Validate] Mean error: {validation['mean_error_px']:.2f} px, "
                      f"Max error: {validation['max_error_px']:.2f} px")

    except Exception as e:
        print(f"[FaceMesh Validate] Error: {e}")

    return validation
