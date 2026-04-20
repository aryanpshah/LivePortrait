import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import facemesh_guards as fg


def _fake_landmarks(H: int = 100, W: int = 100) -> np.ndarray:
    """Generate a simple circular landmark layout for testing."""
    center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    radius = min(H, W) * 0.3
    coords = []
    for i in range(468):
        theta = 2.0 * np.pi * (i / 468.0)
        offset = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * radius
        coords.append(center + offset)
    return np.stack(coords, axis=0).astype(np.float32)


def test_cap_delta_magnitude_caps_spikes():
    delta = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 12.0]], dtype=np.float32)
    weights = np.ones(3, dtype=np.float32)
    capped, cap_value, stats = fg.cap_delta_magnitude(delta, weights=weights, max_px=5.0)
    assert np.all(np.linalg.norm(capped, axis=1) <= 5.0001)
    assert cap_value == 5.0
    assert stats["after_max"] <= 5.0001


def test_smooth_delta_knn_reduces_peak():
    L = np.stack([np.array([i * 5.0, 0.0], dtype=np.float32) for i in range(5)], axis=0)
    delta = np.zeros((5, 2), dtype=np.float32)
    delta[2] = np.array([10.0, 0.0], dtype=np.float32)
    smoothed, rms_hist = fg.smooth_delta_knn(L, delta, k=2, iters=2, lam=0.5)
    assert smoothed[2, 0] < 10.0  # peak should attenuate
    assert len(rms_hist) == 2


def test_apply_anchor_zeroing_scales_down():
    delta = np.ones((4, 2), dtype=np.float32)
    anchored = fg.apply_anchor_zeroing(delta, anchor_idx=[0, 1], strength=0.5)
    assert np.allclose(anchored[0], [0.5, 0.5])
    assert np.allclose(anchored[1], [0.5, 0.5])
    assert np.allclose(anchored[2], [1.0, 1.0])


def test_face_masks_are_bounded():
    L = _fake_landmarks()
    mask = fg.build_face_mask_from_hull((100, 100), L, dilate=4, erode=0, blur=5)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask.mean() > 0.0


def test_mouth_only_mask_focuses_center():
    L = _fake_landmarks()
    mask = fg.make_mouth_only_mask((100, 100), L, radius_px=20, sigma_px=5)
    center_val = mask[50, 50]
    corner_val = mask[0, 0]
    assert center_val > 0.5
    assert corner_val < 0.05


def test_composite_face_only_blends():
    H, W = 50, 50
    original = np.zeros((H, W, 3), dtype=np.uint8)
    warped = np.full((H, W, 3), 255, dtype=np.uint8)
    face_mask = np.ones((H, W), dtype=np.float32)
    blended = fg.composite_face_only(original, warped, face_mask)
    assert blended.mean() > 250.0
