"""Mouth TPS residual pass used after LivePortrait render."""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Optional

import cv2
import numpy as np

from facemesh_flip_align import align_flipped_landmarks_indexwise
from facemesh_utils import FaceMeshExtractor, default_facemesh_regions

default_regions = default_facemesh_regions

ANCHORS = [10, 152, 234, 454]
IDX_MOUTH_LEFT_CORNER = 61
IDX_MOUTH_RIGHT_CORNER = 291

# Mouth side gating v2 (mouth-local geometry, soft tanh, contralateral floor, bounded renorm).
# Without gating, donor mouth asymmetry leaks motion to the wrong hemiface; the legacy
# sigmoid about mean-x was overly suppressive. v2 keeps localization while preserving more
# useful signal: normalized lip horizontal coordinate, tanh same-side weights with a floor,
# and capped gain so mean vector norm is only partially restored after gating.
GATE_FLOOR = 0.18
GATE_SOFTNESS = 0.18
GATE_RENORM_TARGET_FRAC = 0.85
GATE_RENORM_MAX_GAIN = 1.45
GATE_RENORM_EPS = 1e-9


def remap_interpolation_flag(name: str) -> int:
    """Map interpolation name to OpenCV remap flag."""
    m = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return m.get(name.lower().strip(), cv2.INTER_CUBIC)


def _tps_U(r2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    r2 = np.maximum(r2, eps)
    return r2 * np.log(r2)


def _fit_tps_weights(
    ctrl_dst: np.ndarray, ctrl_src_1d: np.ndarray, reg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scalar TPS weights."""
    N = ctrl_dst.shape[0]
    if N < 3:
        raise ValueError("TPS needs at least 3 control points")
    diff = ctrl_dst[:, None, :] - ctrl_dst[None, :, :]
    r2 = np.sum(diff**2, axis=2) + 1e-12
    K = _tps_U(r2)
    K = K + reg * np.eye(N)
    P = np.concatenate([np.ones((N, 1)), ctrl_dst], axis=1)
    A = np.zeros((N + 3, N + 3), dtype=np.float64)
    A[:N, :N] = K
    A[:N, N:] = P
    A[N:, :N] = P.T
    b = np.zeros(N + 3, dtype=np.float64)
    b[:N] = ctrl_src_1d
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    w, a = sol[:N], sol[N:]
    return w, a


def _eval_tps_grid(
    yy: np.ndarray,
    xx: np.ndarray,
    ctrl_dst: np.ndarray,
    w: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Evaluate scalar TPS on a grid."""
    pts = np.stack([xx, yy], axis=-1)
    diff = pts[..., None, :] - ctrl_dst[None, None, :, :]
    r2 = np.sum(diff**2, axis=-1) + 1e-12
    U = _tps_U(r2)
    nonlin = np.tensordot(U, w, axes=([-1], [0]))
    return a[0] + a[1] * xx + a[2] * yy + nonlin


def _estimate_similarity_A(donor_xy: np.ndarray, out_xy: np.ndarray) -> np.ndarray:
    """Estimate 2x2 linear part of donor->output similarity."""
    M, _ = cv2.estimateAffinePartial2D(
        donor_xy.astype(np.float32), out_xy.astype(np.float32), method=cv2.LMEDS
    )
    if M is None:
        return np.eye(2, dtype=np.float64)
    A = M[:, :2].astype(np.float64)
    return A


def _write_tps_summary_json(*dirs: str | None, summary: dict[str, Any]) -> None:
    for debug_dir in dirs:
        if not debug_dir:
            continue
        os.makedirs(debug_dir, exist_ok=True)
        with open(osp.join(debug_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def _disp_norm_stats(dx: np.ndarray, dy: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    """Basic displacement stats inside ROI."""
    if not np.any(roi):
        return {
            "mean_abs_dx": 0.0,
            "max_abs_dx": 0.0,
            "mean_abs_dy": 0.0,
            "max_abs_dy": 0.0,
            "mean_norm": 0.0,
            "max_norm": 0.0,
        }
    adx = np.abs(dx[roi])
    ady = np.abs(dy[roi])
    n = np.sqrt(dx[roi] ** 2 + dy[roi] ** 2)
    return {
        "mean_abs_dx": float(np.mean(adx)),
        "max_abs_dx": float(np.max(adx)),
        "mean_abs_dy": float(np.mean(ady)),
        "max_abs_dy": float(np.max(ady)),
        "mean_norm": float(np.mean(n)),
        "max_norm": float(np.max(n)),
    }


def _lip_vec_norm_stats(v: np.ndarray) -> dict[str, float]:
    """Mean/max vector norm."""
    if v.size == 0:
        return {"mean_norm": 0.0, "max_norm": 0.0}
    m = np.linalg.norm(v, axis=1)
    return {"mean_norm": float(np.mean(m)), "max_norm": float(np.max(m))}


def apply_lip_lr_grouping(
    lip_xy: np.ndarray,
    delta_xy: np.ndarray,
    strength: float = 0.55,
    softness: float = 0.20,
    corner_boost: float = 1.25,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """
    Structural prior on lip displacement vectors before cap / scale / TPS.

    Soft left/right coherence (tanh memberships, weighted side means) — not multi-region
    or topology-residual models. Corners get slightly stronger influence via point_w when
    estimating side means and via local_strength. Runs after side gating v2 so it shapes the
    gated field that drives control points.
    """
    lip_xy = np.asarray(lip_xy, dtype=np.float64)
    delta_xy = np.asarray(delta_xy, dtype=np.float64)
    if lip_xy.shape != delta_xy.shape or lip_xy.ndim != 2 or lip_xy.shape[1] != 2:
        raise ValueError("lip_xy and delta_xy must be [N, 2] with matching shapes")

    n = lip_xy.shape[0]
    if n == 0:
        return (
            delta_xy.copy(),
            {
                "grouping_enabled": True,
                "grouping_mode": "left_right",
                "error": "empty_lip_points",
            },
            {},
        )

    x = lip_xy[:, 0]
    x_left = float(np.min(x))
    x_right = float(np.max(x))
    cx = 0.5 * (x_left + x_right)
    half_w = max(0.5 * (x_right - x_left), 1e-6)
    t = (x - cx) / half_w
    t = np.clip(t, -1.0, 1.0)

    s = max(float(softness), 1e-4)
    w_right = 0.5 * (1.0 + np.tanh(t / s))
    w_left = 1.0 - w_right

    cornerness = np.abs(t)
    point_w = 1.0 + float(corner_boost) * cornerness
    wl = w_left * point_w
    wr = w_right * point_w

    sw_l = float(np.sum(wl)) + 1e-12
    sw_r = float(np.sum(wr)) + 1e-12
    left_mean = np.sum(wl[:, np.newaxis] * delta_xy, axis=0) / sw_l
    right_mean = np.sum(wr[:, np.newaxis] * delta_xy, axis=0) / sw_r

    grouped = w_left[:, np.newaxis] * left_mean + w_right[:, np.newaxis] * right_mean

    local_strength = float(strength) * (0.70 + 0.30 * np.abs(t))
    local_strength = np.clip(local_strength, 0.0, 1.0)
    new_delta = (1.0 - local_strength[:, np.newaxis]) * delta_xy + local_strength[
        :, np.newaxis
    ] * grouped
    new_delta = np.nan_to_num(new_delta, nan=0.0, posinf=0.0, neginf=0.0)

    norms_before = np.linalg.norm(delta_xy, axis=1)
    norms_after = np.linalg.norm(new_delta, axis=1)
    change = np.linalg.norm(new_delta - delta_xy, axis=1)

    metrics: dict[str, Any] = {
        "grouping_enabled": True,
        "grouping_mode": "left_right",
        "grouping_strength": float(strength),
        "grouping_softness": float(softness),
        "grouping_corner_boost": float(corner_boost),
        "left_mean_dx": float(left_mean[0]),
        "left_mean_dy": float(left_mean[1]),
        "right_mean_dx": float(right_mean[0]),
        "right_mean_dy": float(right_mean[1]),
        "mean_delta_norm_before_grouping": float(np.mean(norms_before)),
        "mean_delta_norm_after_grouping": float(np.mean(norms_after)),
        "max_delta_norm_before_grouping": float(np.max(norms_before)),
        "max_delta_norm_after_grouping": float(np.max(norms_after)),
        "mean_grouping_change_norm": float(np.mean(change)),
        "left_membership_min": float(np.min(w_left)),
        "left_membership_max": float(np.max(w_left)),
        "right_membership_min": float(np.min(w_right)),
        "right_membership_max": float(np.max(w_right)),
    }
    arrays = {
        "w_left": w_left.astype(np.float64),
        "w_right": w_right.astype(np.float64),
        "t": t.astype(np.float64),
    }
    return new_delta.astype(np.float64), metrics, arrays


def apply_mouth_side_gating_v2(
    lip_xy: np.ndarray,
    delta_xy: np.ndarray,
    chosen_side: str,
    *,
    debug: bool = False,
    debug_dict: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """
    Soft hemispheric gating on lip displacement vectors using mouth-local x geometry.

    ``side_sigma`` (CLI) is not used here — weights come from normalized lip extent so
    gating tracks actual mouth geometry rather than a global horizontal sigmoid.
    """
    lip_xy = np.asarray(lip_xy, dtype=np.float64)
    delta_xy = np.asarray(delta_xy, dtype=np.float64)
    if lip_xy.shape != delta_xy.shape or lip_xy.ndim != 2 or lip_xy.shape[1] != 2:
        raise ValueError("lip_xy and delta_xy must be [N, 2] with matching shapes")
    if chosen_side not in ("left", "right"):
        raise ValueError('chosen_side must be "left" or "right"')

    n = lip_xy.shape[0]
    if n == 0:
        empty_w = np.array([], dtype=np.float64)
        empty_t = np.array([], dtype=np.float64)
        metrics: dict[str, Any] = {
            "error": "empty_lip_points",
            "mean_norm_before": 0.0,
            "mean_norm_after_pre_renorm": 0.0,
            "mean_norm_after_post_renorm": 0.0,
            "renorm_gain": 1.0,
        }
        return delta_xy.copy(), metrics, empty_w, empty_t

    x = lip_xy[:, 0]
    x_left = float(np.min(x))
    x_right = float(np.max(x))
    cx = 0.5 * (x_left + x_right)
    half_w = max(0.5 * (x_right - x_left), 1e-6)
    t = (x - cx) / half_w
    t = np.clip(t, -1.0, 1.0)

    s = max(float(GATE_SOFTNESS), 1e-6)
    base_right = 0.5 * (1.0 + np.tanh(t / s))
    base_left = 1.0 - base_right
    if chosen_side == "right":
        chosen_raw = base_right
    else:
        chosen_raw = base_left

    gate_w = GATE_FLOOR + (1.0 - GATE_FLOOR) * chosen_raw
    delta_gated = delta_xy * gate_w[:, np.newaxis]

    norms_before = np.linalg.norm(delta_xy, axis=1)
    norms_pre = np.linalg.norm(delta_gated, axis=1)
    mean_before = float(np.mean(norms_before))
    mean_after_pre = float(np.mean(norms_pre))

    renorm_gain = 1.0
    if mean_after_pre > GATE_RENORM_EPS:
        target_mean = float(GATE_RENORM_TARGET_FRAC) * mean_before
        renorm_gain = target_mean / mean_after_pre
        renorm_gain = float(np.clip(renorm_gain, 1.0, float(GATE_RENORM_MAX_GAIN)))
        delta_gated = delta_gated * renorm_gain

    norms_post = np.linalg.norm(delta_gated, axis=1)
    mean_after_post = float(np.mean(norms_post))

    nw = gate_w.size
    metrics = {
        "mean_norm_before": mean_before,
        "mean_norm_after_pre_renorm": mean_after_pre,
        "mean_norm_after_post_renorm": mean_after_post,
        "renorm_gain": renorm_gain,
        "mean_w": float(np.mean(gate_w)),
        "min_w": float(np.min(gate_w)),
        "p10_w": float(np.percentile(gate_w, 10.0)) if nw else 0.0,
        "max_w": float(np.max(gate_w)),
        "t_min": float(np.min(t)),
        "t_max": float(np.max(t)),
    }
    if debug and debug_dict is not None:
        debug_dict["mouth_side_gating_v2"] = metrics
    return delta_gated.astype(np.float64), metrics, gate_w.astype(np.float64), t.astype(np.float64)


def _apply_affine_partial_2d(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to xy points."""
    xy = np.asarray(xy, dtype=np.float64)
    ones = np.ones((xy.shape[0], 1), dtype=np.float64)
    hom = np.hstack([xy, ones])
    return (hom @ M.T)


def _draw_lip_points_bgr(
    bgr: np.ndarray, pts: np.ndarray, color_bgr: tuple[int, int, int], r: int = 3
) -> np.ndarray:
    vis = bgr.copy()
    for i in range(len(pts)):
        p = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        cv2.circle(vis, p, r, color_bgr, -1, lineType=cv2.LINE_AA)
    return vis


def _draw_lip_arrows_bgr(
    bgr: np.ndarray,
    src_pts: np.ndarray,
    delta: np.ndarray,
    arrow_scale: float,
    thickness: int = 1,
) -> np.ndarray:
    """Draw displacement arrows."""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    mag = np.linalg.norm(delta, axis=1)
    mv = float(mag.max() + 1e-9)
    for i in range(len(src_pts)):
        t = float(mag[i] / mv)
        col = (int(80 * t), int(255 * t), int(255 * (1 - t)))
        x0 = int(round(src_pts[i, 0]))
        y0 = int(round(src_pts[i, 1]))
        x1 = int(round(src_pts[i, 0] + arrow_scale * delta[i, 0]))
        y1 = int(round(src_pts[i, 1] + arrow_scale * delta[i, 1]))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), col, thickness, tipLength=0.25, line_type=cv2.LINE_AA)
    return vis


def _save_dense_disp_viz(
    out_dir: str,
    base_rgb: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    roi_mask: np.ndarray,
    quiver_step: int = 8,
    quiver_mult: float = 12.0,
) -> None:
    """Save heatmap and quiver debug views."""
    H, W = dx.shape
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    mag = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)
    m = roi_mask > 0.05
    if np.any(m):
        p98 = float(np.percentile(mag[m], 98.0))
        p98 = max(p98, 1e-6)
    else:
        p98 = float(mag.max() + 1e-6)
    mag_n = np.clip(mag / p98, 0.0, 1.0)
    hm_u8 = (mag_n * 255.0).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)
    roi_u8 = (roi_mask > 0.05).astype(np.uint8) * 255
    roi_u8 = cv2.GaussianBlur(roi_u8, (0, 0), 2.0)
    blend = roi_u8.astype(np.float32) / 255.0
    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    comp = hm_color.astype(np.float32) * blend[..., None] + gray3 * (1.0 - blend[..., None])
    cv2.imwrite(osp.join(out_dir, "mouth_disp_heatmap.png"), np.clip(comp, 0, 255).astype(np.uint8))

    vis = base_bgr.copy()
    for y in range(0, H, quiver_step):
        for x in range(0, W, quiver_step):
            if not (roi_mask[y, x] > 0.08):
                continue
            x0, y0 = float(x), float(y)
            x1 = x0 + quiver_mult * float(dx[y, x])
            y1 = y0 + quiver_mult * float(dy[y, x])
            cv2.arrowedLine(
                vis,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                (255, 220, 0),
                1,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )
    cv2.imwrite(osp.join(out_dir, "mouth_disp_quiver.png"), vis)


def _convex_hull_mask(
    pts_xy: np.ndarray, shape_hw: tuple[int, int], scale: float
) -> np.ndarray:
    """Binary mask from scaled convex hull."""
    h, w = shape_hw
    if len(pts_xy) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    c = np.mean(pts_xy, axis=0)
    scaled = c + scale * (pts_xy - c)
    hull = cv2.convexHull(scaled.astype(np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


def _build_mouth_blend_masks(
    src_pts: np.ndarray,
    shape_hw: tuple[int, int],
    mask_scale: float,
    erode_iter: int,
    feather_sigma: float,
    legacy_mask_blur_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw + feathered blend masks."""
    h, w = shape_hw
    raw_u8 = _convex_hull_mask(src_pts, shape_hw, mask_scale)
    raw = raw_u8.astype(np.float32) / 255.0

    if legacy_mask_blur_kernel > 0:
        kb = max(3, int(legacy_mask_blur_kernel) | 1)
        feathered = cv2.GaussianBlur(raw_u8.astype(np.float32), (kb, kb), 0) / 255.0
        feathered = np.clip(feathered, 0.0, 1.0)
        return raw, feathered

    tight = raw_u8
    if erode_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(int(erode_iter)):
            tight = cv2.erode(tight, k)
    tight_f = tight.astype(np.float32) / 255.0
    sg = max(float(feather_sigma), 0.5)
    feathered = cv2.GaussianBlur(tight_f, (0, 0), sg)
    feathered = np.clip(feathered, 0.0, 1.0)
    return raw, feathered


def _inner_lip_region_mask(src_pts: np.ndarray, shape_hw: tuple[int, int], radial: float) -> np.ndarray:
    """Smaller hull inside the full mouth mask."""
    h, w = shape_hw
    if len(src_pts) < 3:
        return np.zeros((h, w), dtype=np.float32)
    c = np.mean(src_pts, axis=0)
    pts = c + float(radial) * (src_pts - c)
    u8 = _convex_hull_mask(pts, shape_hw, 1.0)
    return (u8.astype(np.float32) / 255.0)


def _apply_lip_unsharp(
    rgb: np.ndarray,
    lip_weight: np.ndarray,
    amount: float,
    sigma: float,
) -> np.ndarray:
    """Unsharp mask blended only inside lip_weight."""
    lip_weight = np.clip(lip_weight.astype(np.float32), 0.0, 1.0)
    if amount <= 0 or sigma <= 0:
        return rgb
    base = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(rgb, (0, 0), float(sigma)).astype(np.float32)
    sharp = np.clip(base + float(amount) * (base - blur), 0, 255)
    m = lip_weight[..., None]
    out = sharp * m + base * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _lip_edge_viz(rgb: np.ndarray, lip_mask01: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(gray, 60, 160)
    m = (lip_mask01 > 0.15).astype(np.uint8) * 255
    e = cv2.bitwise_and(e, e, mask=m)
    return np.stack([e, e, e], axis=-1)


def apply_mouth_tps_residual(
    donor_rgb: np.ndarray,
    out_rgb: np.ndarray,
    facemesh_extractor: Optional[FaceMeshExtractor],
    regions: dict,
    alpha: float,
    reg: float,
    cap_px: float,
    side_mode: str,
    side_sigma: float,  # retained for CLI/API; mouth side gating v2 uses mouth-local x, not this
    mask_scale: float,
    mask_blur: int,
    debug_dir: str | None,
    *,
    mouth_remap_interp: str = "cubic",
    mouth_residual_scale: float = 0.8,
    mouth_mask_erode_iter: int = 1,
    mouth_mask_feather_sigma: float = 2.5,
    lip_sharpen: bool = False,
    lip_sharpen_amount: float = 0.28,
    lip_sharpen_sigma: float = 1.0,
    lip_sharpen_radial: float = 0.90,
    artifact_dir: str | None = None,
    mouth_disable_side_gating: bool = False,
    mouth_force_side: str | None = None,
    mouth_debug_arrow_scale: float = 1.0,
    facemesh_group_lr: bool = False,
    facemesh_group_strength: float = 0.55,
    facemesh_group_softness: float = 0.20,
    facemesh_group_corner_boost: float = 1.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply mouth-only TPS residual warp and blend it back."""
    interp_flag = remap_interpolation_flag(mouth_remap_interp)
    legacy_kernel = int(mask_blur) if int(mask_blur) > 0 else 0

    summary: dict[str, Any] = {
        "ok": False,
        "alpha": alpha,
        "reg": reg,
        "cap_px": cap_px,
        "side_mode": side_mode,
        "chosen_side": None,
        "side_sigma": side_sigma,
        "mouth_remap_interp": mouth_remap_interp,
        "mouth_residual_scale": mouth_residual_scale,
        "mouth_mask_erode_iter": mouth_mask_erode_iter,
        "mouth_mask_feather_sigma": mouth_mask_feather_sigma,
        "mouth_mask_blur_legacy_kernel": legacy_kernel,
        "lip_sharpen": lip_sharpen,
        "Nlip": 0,
        "n_capped": 0,
        "max_before": 0.0,
        "max_after": 0.0,
        "fold_fraction": None,
        "oob_fraction": None,
        "mean_diff_in_mask": None,
        "max_diff_in_mask": None,
        "error": None,
        "diagnostics": {},
        "mouth_disable_side_gating": mouth_disable_side_gating,
        "mouth_force_side": mouth_force_side,
        "mouth_debug_arrow_scale": mouth_debug_arrow_scale,
        "facemesh_group_lr": facemesh_group_lr,
        "facemesh_group_strength": facemesh_group_strength,
        "facemesh_group_softness": facemesh_group_softness,
        "facemesh_group_corner_boost": facemesh_group_corner_boost,
    }

    lips_idx = regions.get("FACEMESH_LIPS_IDX")
    if lips_idx is None:
        lips_idx = default_facemesh_regions()["FACEMESH_LIPS_IDX"]

    if out_rgb.dtype != np.uint8 or donor_rgb.dtype != np.uint8:
        summary["error"] = "expected_uint8_rgb"
        _write_tps_summary_json(debug_dir, artifact_dir, summary=summary)
        return out_rgb.copy(), summary

    H, W = out_rgb.shape[:2]

    close_after = False
    ex = facemesh_extractor
    if ex is None:
        ex = FaceMeshExtractor()
        close_after = True

    try:
        Ld = ex.process_rgb(donor_rgb)
        if Ld is None:
            summary["error"] = "donor_facemesh_failed"
            _write_tps_summary_json(debug_dir, artifact_dir, summary=summary)
            return out_rgb.copy(), summary

        donor_flip = donor_rgb[:, ::-1, :].copy()
        Lf = ex.process_rgb(donor_flip)
        if Lf is None:
            summary["error"] = "donor_flip_facemesh_failed"
            _write_tps_summary_json(debug_dir, artifact_dir, summary=summary)
            return out_rgb.copy(), summary

        Lout = ex.process_rgb(out_rgb)
        if Lout is None:
            summary["error"] = "out_facemesh_failed"
            _write_tps_summary_json(debug_dir, artifact_dir, summary=summary)
            return out_rgb.copy(), summary
    except Exception as e:
        summary["error"] = f"facemesh_error:{e!r}"
        _write_tps_summary_json(debug_dir, artifact_dir, summary=summary)
        return out_rgb.copy(), summary
    finally:
        if close_after:
            ex.close()

    Wd = donor_rgb.shape[1]
    Lf_back = np.empty_like(Lf)
    Lf_back[:, 0] = (Wd - 1.0) - Lf[:, 0]
    Lf_back[:, 1] = Lf[:, 1]

    lips_idx_arr = np.array(lips_idx, dtype=np.int64)
    Lf_aligned = align_flipped_landmarks_indexwise(Lf_back)
    delta_lips_raw_donor = Ld[lips_idx_arr] - Lf_aligned[lips_idx_arr]

    delta_donor = Ld - Lf_aligned
    bias = np.mean(delta_donor[ANCHORS], axis=0)
    delta_donor = delta_donor - bias

    # Delta vectors only use the 2x2 linear part
    A = _estimate_similarity_A(Ld[ANCHORS], Lout[ANCHORS])
    delta_out = (A @ delta_donor.T).T
    # Raw asymmetry (no anchor bias) mapped to output frame for debug checks
    delta_raw_out = (A @ (Ld[lips_idx_arr] - Lf_aligned[lips_idx_arr]).T).T

    M_donor_to_out, _ = cv2.estimateAffinePartial2D(
        Ld[ANCHORS].astype(np.float32), Lout[ANCHORS].astype(np.float32), method=cv2.LMEDS
    )
    if M_donor_to_out is None:
        M_donor_to_out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    xy_lip = Lout[lips_idx_arr]

    auto_chosen: str | None = None
    if side_mode == "auto":
        dy_l = delta_out[IDX_MOUTH_LEFT_CORNER, 1]
        dy_r = delta_out[IDX_MOUTH_RIGHT_CORNER, 1]
        auto_chosen = "right" if abs(float(dy_r)) >= abs(float(dy_l)) else "left"

    chosen = side_mode if side_mode != "auto" else (auto_chosen or "bilateral")
    if mouth_force_side in ("left", "right"):
        chosen = str(mouth_force_side)

    summary["chosen_side"] = chosen
    summary["auto_chosen_side"] = auto_chosen

    delta_aligned_lips = delta_out[lips_idx_arr].copy()

    w: np.ndarray
    t_side: np.ndarray | None = None
    side_gating_v2_block: dict[str, Any]

    if mouth_disable_side_gating or chosen == "bilateral":
        w = np.ones(len(lips_idx_arr), dtype=np.float64)
        delta_gated_pre_cap = delta_aligned_lips.copy()
        side_gating_v2_block = {"enabled": False}
    elif chosen in ("left", "right"):
        delta_gated_pre_cap, v2m, w, t_side = apply_mouth_side_gating_v2(
            xy_lip,
            delta_aligned_lips,
            chosen,
        )
        side_gating_v2_block = {
            "enabled": True,
            "version": "mouth_local_floor_renorm_v2",
            "chosen_side": chosen,
            "gate_floor": GATE_FLOOR,
            "gate_softness": GATE_SOFTNESS,
            "gate_renorm_target_frac": GATE_RENORM_TARGET_FRAC,
            "gate_renorm_max_gain": GATE_RENORM_MAX_GAIN,
            **v2m,
        }
    else:
        w = np.ones(len(lips_idx_arr), dtype=np.float64)
        delta_gated_pre_cap = delta_aligned_lips.copy()
        side_gating_v2_block = {"enabled": False}

    delta_lips = delta_gated_pre_cap.copy()
    lip_lr_group_arrays: dict[str, np.ndarray] = {}
    lip_lr_metrics: Optional[dict[str, Any]] = None
    # --facemesh-group-lr: soft left/right grouping only (after side gating v2, before cap).
    if facemesh_group_lr:
        delta_lips, lip_lr_metrics, lip_lr_group_arrays = apply_lip_lr_grouping(
            xy_lip,
            delta_lips,
            strength=facemesh_group_strength,
            softness=facemesh_group_softness,
            corner_boost=facemesh_group_corner_boost,
        )

    st_after_lr = _lip_vec_norm_stats(delta_lips)
    delta_after_lr_pre_cap = delta_lips.copy()

    mag = np.linalg.norm(delta_lips, axis=1)
    max_before = float(mag.max()) if mag.size else 0.0
    n_capped = int(np.sum(mag > cap_px))
    scale_cap = np.ones_like(mag)
    mcap = mag > cap_px
    scale_cap[mcap] = cap_px / (mag[mcap] + 1e-12)
    delta_lips = delta_lips * scale_cap[:, None]
    max_after = float(np.linalg.norm(delta_lips, axis=1).max()) if mag.size else 0.0

    summary["Nlip"] = int(len(lips_idx_arr))
    summary["n_capped"] = n_capped
    summary["max_before"] = max_before
    summary["max_after"] = max_after

    eff = float(mouth_residual_scale) * float(alpha)
    st_raw = _lip_vec_norm_stats(delta_lips_raw_donor)
    st_bias = _lip_vec_norm_stats(delta_donor[lips_idx_arr])
    st_aligned = _lip_vec_norm_stats(delta_aligned_lips)
    st_gated = _lip_vec_norm_stats(delta_gated_pre_cap)
    st_cap = _lip_vec_norm_stats(delta_lips)
    st_scaled = _lip_vec_norm_stats(eff * delta_lips)
    raw_out_stats = _lip_vec_norm_stats(delta_raw_out)

    _, s_svd, _ = np.linalg.svd(A)
    mean_ratio_aligned_over_donor = float(
        st_aligned["mean_norm"] / (st_bias["mean_norm"] + 1e-12)
    )
    diag: dict[str, Any] = {
        "delta_lips_raw_donor": st_raw,
        "delta_lips_after_anchor_bias_donor": st_bias,
        "delta_raw_aligned_output_no_bias": raw_out_stats,
        "delta_aligned_output_after_bias": st_aligned,
        "delta_after_side_gating_pre_cap": st_gated,
        "delta_after_cap": st_cap,
        "delta_after_mouth_residual_scale": st_scaled,
        "eff_mouth": eff,
        "similarity_A_singular_values": [float(s_svd[0]), float(s_svd[1])],
        "similarity_A_det": float(np.linalg.det(A)),
        "mean_norm_ratio_aligned_over_donor_lip_delta": mean_ratio_aligned_over_donor,
        "side_gating_weights": {
            "mean_w": float(np.mean(w)),
            "min_w": float(np.min(w)),
            "p10_w": float(np.percentile(w, 10.0)) if w.size else 0.0,
            "max_w": float(np.max(w)),
        },
        "side_gating_v2": side_gating_v2_block,
        "checks": {
            "gating_weights_mostly_near_zero": bool(
                float(np.mean(w)) < 0.08 and not side_gating_v2_block.get("enabled", False)
            ),
            "auto_differs_from_forced": bool(
                mouth_force_side in ("left", "right")
                and auto_chosen is not None
                and auto_chosen != mouth_force_side
            ),
            "donor_raw_asymmetry_tiny": bool(st_raw["max_norm"] < 0.35),
            "similarity_reduces_lip_delta_norm_strongly": bool(mean_ratio_aligned_over_donor < 0.4),
            "scaled_control_displacement_subpixel": bool(st_scaled["max_norm"] < 0.5),
        },
    }
    if facemesh_group_lr and lip_lr_metrics is not None:
        diag["delta_after_lr_grouping"] = st_after_lr
        diag["lip_lr_grouping"] = lip_lr_metrics
    else:
        diag["lip_lr_grouping"] = {"grouping_enabled": False}
    summary["diagnostics"] = diag

    # Build source/destination mouth control points.
    src_pts = Lout[lips_idx_arr].copy()
    dst_pts = src_pts + eff * delta_lips
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0.0, W - 1.0)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0.0, H - 1.0)

    mask_raw01, mask_feather01 = _build_mouth_blend_masks(
        src_pts,
        (H, W),
        mask_scale,
        mouth_mask_erode_iter,
        mouth_mask_feather_sigma,
        legacy_kernel,
    )

    # Inverse map g(dst)=src for cv2.remap.
    ctrl_dst = dst_pts
    ctrl_src_x = src_pts[:, 0]
    ctrl_src_y = src_pts[:, 1]
    wx, ax = _fit_tps_weights(ctrl_dst, ctrl_src_x, reg)
    wy, ay = _fit_tps_weights(ctrl_dst, ctrl_src_y, reg)

    grid_step = 2
    ys = np.arange(0, H, grid_step, dtype=np.float64)
    xs = np.arange(0, W, grid_step, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    map_x_c = _eval_tps_grid(yy, xx, ctrl_dst, wx, ax).astype(np.float32)
    map_y_c = _eval_tps_grid(yy, xx, ctrl_dst, wy, ay).astype(np.float32)

    map_x = cv2.resize(map_x_c, (W, H), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_c, (W, H), interpolation=cv2.INTER_LINEAR)

    yy_full, xx_full = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    dx_field = map_x.astype(np.float64) - xx_full
    dy_field = map_y.astype(np.float64) - yy_full
    roi_mouth = mask_feather01 > 0.05
    dense_stats = _disp_norm_stats(dx_field, dy_field, roi_mouth)
    summary["diagnostics"]["dense_field_in_mouth_roi"] = dense_stats
    summary["diagnostics"]["mouth_roi_area_fraction"] = float(np.mean(roi_mouth))
    mag_field = np.sqrt(dx_field**2 + dy_field**2)
    mx = float(np.max(mag_field[roi_mouth])) if np.any(roi_mouth) else 0.0
    summary["diagnostics"]["checks"]["dense_max_norm_very_small_in_roi"] = bool(mx < 0.25)

    viz_targets: list[str] = []
    if artifact_dir:
        viz_targets.append(artifact_dir)
    if debug_dir and debug_dir not in viz_targets:
        viz_targets.append(debug_dir)
    if viz_targets:
        ars = float(mouth_debug_arrow_scale)
        base_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        mirrored_lips_out = _apply_affine_partial_2d(M_donor_to_out, Lf_aligned[lips_idx_arr])
        for viz_dir in viz_targets:
            os.makedirs(viz_dir, exist_ok=True)
            cv2.imwrite(
                osp.join(viz_dir, "lip_ctrl_src_overlay.png"),
                _draw_lip_points_bgr(base_bgr, src_pts, (0, 255, 255), 4),
            )
            cv2.imwrite(
                osp.join(viz_dir, "lip_ctrl_mirrored_overlay.png"),
                _draw_lip_points_bgr(base_bgr, mirrored_lips_out, (0, 128, 255), 4),
            )
            cv2.imwrite(
                osp.join(viz_dir, "lip_ctrl_delta_overlay.png"),
                _draw_lip_arrows_bgr(base_bgr, src_pts, delta_raw_out, ars),
            )
            cv2.imwrite(
                osp.join(viz_dir, "lip_ctrl_aligned_delta_overlay.png"),
                _draw_lip_arrows_bgr(base_bgr, src_pts, delta_aligned_lips, ars),
            )
            cv2.imwrite(
                osp.join(viz_dir, "lip_ctrl_gated_delta_overlay.png"),
                _draw_lip_arrows_bgr(base_bgr, src_pts, delta_gated_pre_cap, ars),
            )
            cv2.imwrite(
                osp.join(viz_dir, "lip_vectors_after_side_gating.png"),
                _draw_lip_arrows_bgr(base_bgr, src_pts, delta_gated_pre_cap, ars),
            )
            if side_gating_v2_block.get("enabled"):
                np.save(
                    osp.join(viz_dir, "lip_side_gate_weights.npy"),
                    np.asarray(w, dtype=np.float64),
                )
                if t_side is not None and t_side.size:
                    np.save(
                        osp.join(viz_dir, "lip_side_gate_tcoord.npy"),
                        np.asarray(t_side, dtype=np.float64),
                    )
            if facemesh_group_lr:
                cv2.imwrite(
                    osp.join(viz_dir, "lip_vectors_before_grouping.png"),
                    _draw_lip_arrows_bgr(base_bgr, src_pts, delta_gated_pre_cap, ars),
                )
                cv2.imwrite(
                    osp.join(viz_dir, "lip_vectors_after_grouping.png"),
                    _draw_lip_arrows_bgr(base_bgr, src_pts, delta_after_lr_pre_cap, ars),
                )
                np.save(
                    osp.join(viz_dir, "lip_delta_before_grouping.npy"),
                    np.asarray(delta_gated_pre_cap, dtype=np.float64),
                )
                np.save(
                    osp.join(viz_dir, "lip_delta_after_grouping.npy"),
                    np.asarray(delta_after_lr_pre_cap, dtype=np.float64),
                )
                wl_a = lip_lr_group_arrays.get("w_left")
                wr_a = lip_lr_group_arrays.get("w_right")
                if wl_a is not None and wr_a is not None:
                    np.save(
                        osp.join(viz_dir, "lip_lr_membership.npy"),
                        np.column_stack(
                            [np.asarray(wl_a, dtype=np.float64), np.asarray(wr_a, dtype=np.float64)]
                        ),
                    )
                t_a = lip_lr_group_arrays.get("t")
                if t_a is not None:
                    np.save(
                        osp.join(viz_dir, "lip_lr_tcoord.npy"),
                        np.asarray(t_a, dtype=np.float64),
                    )
            _save_dense_disp_viz(
                viz_dir,
                out_rgb,
                dx_field.astype(np.float32),
                dy_field.astype(np.float32),
                mask_feather01,
            )
        print(
            "[mouth_tps_residual] diagnostics:",
            "raw_donor max|d|=%.4f mean|d|=%.4f;"
            % (st_raw["max_norm"], st_raw["mean_norm"]),
            "after_bias_donor max|d|=%.4f;"
            % (st_bias["max_norm"],),
            "aligned max|d|=%.4f gated max|d|=%.4f eff*delta max|d|=%.4f;"
            % (st_aligned["max_norm"], st_gated["max_norm"], st_scaled["max_norm"]),
            "dense_ROI mean|d|=%.4f max|d|=%.4f;"
            % (dense_stats["mean_norm"], dense_stats["max_norm"]),
            "w_mean=%.4f w_min=%.4f chosen=%s auto=%s;"
            % (float(np.mean(w)), float(np.min(w)), chosen, auto_chosen),
            "A_sv=[%.4f,%.4f] roi_frac=%.4f"
            % (
                float(s_svd[0]),
                float(s_svd[1]),
                float(np.mean(roi_mouth)),
            ),
        )

    mask_bl = np.clip(mask_feather01[..., None], 0.0, 1.0)
    mouth_2d = mask_bl[..., 0].astype(np.float32)
    m_fold = mouth_2d > 0.5

    dXdy, dXdx = np.gradient(map_x)
    dYdy, dYdx = np.gradient(map_y)
    detJ = dXdx * dYdy - dXdy * dYdx
    if np.any(m_fold):
        med_det = float(np.median(detJ[m_fold]))
    else:
        med_det = float(np.median(detJ))
    if med_det < 0.0:
        detJ = -detJ

    if np.any(m_fold):
        fold_fraction = float(np.mean(detJ[m_fold] <= 0.0))
    else:
        fold_fraction = float(np.mean(detJ <= 0.0)) if detJ.size else 0.0
    summary["fold_fraction"] = fold_fraction

    oob = (
        (map_x < 0.0)
        | (map_x > float(W - 1))
        | (map_y < 0.0)
        | (map_y > float(H - 1))
    )
    if np.any(m_fold):
        oob_fraction = float(np.mean(oob[m_fold]))
    else:
        oob_fraction = float(np.mean(oob)) if oob.size else 0.0
    summary["oob_fraction"] = oob_fraction

    if artifact_dir:
        assert out_rgb.shape[:2] == (H, W), (out_rgb.shape, H, W)
        assert map_x.shape == (H, W), (map_x.shape, (H, W))
        assert map_y.shape == (H, W), (map_y.shape, (H, W))
        print(
            "[mouth_tps_residual] remap debug:",
            "base_img.shape", out_rgb.shape,
            "map_x.shape", map_x.shape,
            "map_y.shape", map_y.shape,
            "map_x min/max", float(np.min(map_x)), float(np.max(map_x)),
            "map_y min/max", float(np.min(map_y)), float(np.max(map_y)),
        )

    warped = cv2.remap(
        out_rgb,
        map_x,
        map_y,
        interpolation=interp_flag,
        borderMode=cv2.BORDER_REPLICATE,
    )

    comp = warped.astype(np.float64) * mask_bl + out_rgb.astype(np.float64) * (1.0 - mask_bl)
    final = np.clip(np.round(comp), 0, 255).astype(np.uint8)

    if lip_sharpen:
        lip_w = _inner_lip_region_mask(src_pts, (H, W), lip_sharpen_radial)
        lip_w = cv2.GaussianBlur(lip_w, (0, 0), 1.2)
        lip_w = np.clip(lip_w, 0.0, 1.0)
        final = _apply_lip_unsharp(final, lip_w, lip_sharpen_amount, lip_sharpen_sigma)

    diff = np.mean(np.abs(final.astype(np.float32) - out_rgb.astype(np.float32)), axis=2)
    m = mask_bl[..., 0] > 0.05
    if np.any(m):
        summary["mean_diff_in_mask"] = float(np.mean(diff[m]))
        summary["max_diff_in_mask"] = float(np.max(diff[m]))
    summary["ok"] = True

    # Optional run artifacts (A/B interpolation, feathering, sharpen). Metrics above are unchanged.
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        cv2.imwrite(
            osp.join(artifact_dir, "liveportrait_base.png"),
            cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            osp.join(artifact_dir, "mouth_warped_full.png"),
            cv2.cvtColor(warped, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            osp.join(artifact_dir, "mouth_mask_raw.png"),
            (np.clip(mask_raw01, 0, 1) * 255).astype(np.uint8),
        )
        cv2.imwrite(
            osp.join(artifact_dir, "mouth_mask_feathered.png"),
            (np.clip(mask_feather01, 0, 1) * 255).astype(np.uint8),
        )
        cv2.imwrite(
            osp.join(artifact_dir, "final_composite.png"),
            cv2.cvtColor(final, cv2.COLOR_RGB2BGR),
        )
        le_b = _lip_edge_viz(out_rgb, mask_feather01)
        le_a = _lip_edge_viz(final, mask_feather01)
        cv2.imwrite(osp.join(artifact_dir, "lip_edges_before.png"), cv2.cvtColor(le_b, cv2.COLOR_RGB2BGR))
        cv2.imwrite(osp.join(artifact_dir, "lip_edges_after.png"), cv2.cvtColor(le_a, cv2.COLOR_RGB2BGR))

    # debug
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(osp.join(debug_dir, "before.png"), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(osp.join(debug_dir, "after.png"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        vis = out_rgb.copy()
        for i in range(len(src_pts)):
            p0 = (int(round(src_pts[i, 0])), int(round(src_pts[i, 1])))
            p1 = (int(round(dst_pts[i, 0])), int(round(dst_pts[i, 1])))
            cv2.circle(vis, p0, 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, p1, 3, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.imwrite(osp.join(debug_dir, "lip_control_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        v2 = out_rgb.copy()
        magv = np.linalg.norm(eff * delta_lips, axis=1)
        mv = float(magv.max() + 1e-6)
        for i in range(len(src_pts)):
            t = float(magv[i] / mv)
            col = (int(255 * (1 - t)), int(255 * t), 128)
            x0, y0 = int(round(src_pts[i, 0])), int(round(src_pts[i, 1]))
            x1 = int(round(src_pts[i, 0] + eff * delta_lips[i, 0]))
            y1 = int(round(src_pts[i, 1] + eff * delta_lips[i, 1]))
            cv2.arrowedLine(v2, (x0, y0), (x1, y1), col, 1, tipLength=0.3)
        cv2.imwrite(osp.join(debug_dir, "lip_vectors.png"), cv2.cvtColor(v2, cv2.COLOR_RGB2BGR))

        cv2.imwrite(osp.join(debug_dir, "mouth_mask.png"), (mask_bl[..., 0] * 255).astype(np.uint8))
        cv2.imwrite(osp.join(debug_dir, "mouth_mask_raw.png"), (mask_raw01 * 255).astype(np.uint8))

        if np.any(m_fold):
            v = detJ[m_fold]
            lo, hi = float(np.percentile(v, 2.0)), float(np.percentile(v, 98.0))
            det_norm = np.clip((detJ - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        else:
            det_norm = np.zeros_like(detJ)
        hm_u8 = (det_norm * 255.0).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        cv2.imwrite(osp.join(debug_dir, "detJ_heatmap.png"), hm_color)

        fold_bin = (((detJ <= 0.0) & m_fold).astype(np.uint8)) * 255
        cv2.imwrite(osp.join(debug_dir, "fold_mask.png"), fold_bin)

        dh = np.mean(np.abs(final.astype(np.float32) - out_rgb.astype(np.float32)), axis=2)
        dh_n = (dh / (dh.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(debug_dir, "diff_heatmap.png"), dh_n)

    _write_tps_summary_json(debug_dir, artifact_dir, summary=summary)
    return final, summary
