"""
Lightweight asymmetry metrics for fast iteration (Phase 7).
All metrics operate on FaceMesh pixel coordinates (H-down, W-right).
"""

import math
from typing import Dict, Optional

import numpy as np


def mouth_corner_droop(L: np.ndarray, left_idx: int = 61, right_idx: int = 291) -> float:
    """Return y_right - y_left; positive means right corner is lower (y grows downward)."""
    if L is None or L.shape[0] <= max(left_idx, right_idx):
        return 0.0
    y_left = float(L[left_idx, 1])
    y_right = float(L[right_idx, 1])
    return y_right - y_left


def lip_centerline_tilt(L: np.ndarray, top_mid_idx: int = 0, bot_mid_idx: int = 17) -> Dict[str, float]:
    """
    Tilt of lip centerline. Positive angle (deg) means clockwise tilt (x-right, y-down).
    Returns dict with angle_deg and dx.
    """
    if L is None or L.shape[0] <= max(top_mid_idx, bot_mid_idx):
        return {"angle_deg": 0.0, "dx": 0.0}
    top = L[top_mid_idx]
    bot = L[bot_mid_idx]
    vec = bot - top  # (dx, dy)
    angle_rad = math.atan2(vec[0], vec[1] + 1e-6)
    angle_deg = math.degrees(angle_rad)
    return {"angle_deg": angle_deg, "dx": float(vec[0])}


def cheek_height_diff(L: np.ndarray, left_cheek: int = 234, right_cheek: int = 454) -> float:
    """Return y_right - y_left; positive means right cheek is lower."""
    if L is None or L.shape[0] <= max(left_cheek, right_cheek):
        return 0.0
    return float(L[right_cheek, 1] - L[left_cheek, 1])


def jawline_sag_index(L: np.ndarray, chin_idx: int = 152, jaw_left: int = 234, jaw_right: int = 454) -> float:
    """Chin depth relative to jaw anchors. Larger means chin lower."""
    if L is None or L.shape[0] <= max(chin_idx, jaw_left, jaw_right):
        return 0.0
    chin_y = float(L[chin_idx, 1])
    jaw_mean = float((L[jaw_left, 1] + L[jaw_right, 1]) / 2.0)
    return chin_y - jaw_mean


def asymmetry_score(metrics: Dict[str, float]) -> float:
    """
    Weighted sum of absolute metrics to provide a single scalar.
    score = 0.35*|droop| + 0.25*|tilt_deg| + 0.25*|cheek_diff| + 0.15*|sag|
    """
    droop = abs(float(metrics.get("droop", 0.0)))
    tilt = abs(float(metrics.get("tilt_deg", 0.0)))
    cheek = abs(float(metrics.get("cheek_diff", 0.0)))
    sag = abs(float(metrics.get("sag", 0.0)))
    return 0.35 * droop + 0.25 * tilt + 0.25 * cheek + 0.15 * sag


def compute_metrics(L: Optional[np.ndarray]) -> Dict[str, float]:
    """Convenience helper to compute all metrics and aggregate score."""
    if L is None:
        return {
            "droop": 0.0,
            "tilt_deg": 0.0,
            "tilt_dx": 0.0,
            "cheek_diff": 0.0,
            "sag": 0.0,
            "score": 0.0,
        }
    droop = mouth_corner_droop(L)
    tilt = lip_centerline_tilt(L)
    cheek = cheek_height_diff(L)
    sag = jawline_sag_index(L)
    metrics = {
        "droop": droop,
        "tilt_deg": tilt["angle_deg"],
        "tilt_dx": tilt["dx"],
        "cheek_diff": cheek,
        "sag": sag,
    }
    metrics["score"] = asymmetry_score(metrics)
    return metrics
