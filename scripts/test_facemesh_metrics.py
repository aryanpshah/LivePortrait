import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import facemesh_metrics as fm


def _base_landmarks() -> np.ndarray:
    L = np.zeros((468, 2), dtype=np.float32)
    L[61] = np.array([10.0, 10.0], dtype=np.float32)   # left mouth corner
    L[291] = np.array([12.0, 20.0], dtype=np.float32)  # right mouth corner lower
    L[0] = np.array([10.0, 0.0], dtype=np.float32)     # top lip center
    L[17] = np.array([15.0, 10.0], dtype=np.float32)   # bottom lip center offset right
    L[234] = np.array([20.0, 8.0], dtype=np.float32)   # left cheek higher
    L[454] = np.array([25.0, 12.0], dtype=np.float32)  # right cheek lower
    L[152] = np.array([22.0, 30.0], dtype=np.float32)  # chin lower
    return L


def test_mouth_corner_droop_sign():
    L = _base_landmarks()
    droop = fm.mouth_corner_droop(L)
    assert droop == 10.0


def test_lip_centerline_tilt_angle():
    L = _base_landmarks()
    tilt = fm.lip_centerline_tilt(L)
    assert tilt["angle_deg"] > 25.0 and tilt["angle_deg"] < 28.0
    assert tilt["dx"] == 5.0


def test_cheek_height_diff():
    L = _base_landmarks()
    diff = fm.cheek_height_diff(L)
    assert diff == 4.0


def test_jawline_sag_index():
    L = _base_landmarks()
    sag = fm.jawline_sag_index(L)
    # chin_y=30.0, jaw_mean=(8.0+12.0)/2=10.0 -> sag=20.0
    assert sag == 20.0


def test_compute_metrics_bundle():
    L = _base_landmarks()
    metrics = fm.compute_metrics(L)
    assert metrics["droop"] == 10.0
    assert metrics["tilt_dx"] == 5.0
    assert metrics["cheek_diff"] == 4.0
    assert metrics["sag"] == 20.0
    assert metrics["score"] > 0.0
