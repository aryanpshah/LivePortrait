"""Align flipped FaceMesh landmarks back to canonical index order."""

from __future__ import annotations

import os.path as osp

import numpy as np

_PERM_PATH = osp.join(osp.dirname(__file__), "facemesh_flip_perm_468.npy")


def load_flip_permutation_468() -> np.ndarray:
    """Load the 468-index flip permutation."""
    if not osp.isfile(_PERM_PATH):
        raise FileNotFoundError(
            f"Missing {_PERM_PATH}; run facemesh_flip_align setup or restore from repo."
        )
    return np.load(_PERM_PATH)


def align_flipped_landmarks_indexwise(lf_back_xy: np.ndarray) -> np.ndarray:
    """Reindex flip-backed landmarks to match non-flipped anatomy."""
    perm = load_flip_permutation_468()
    if lf_back_xy.shape[0] != 468:
        raise ValueError(f"Expected 468 landmarks, got {lf_back_xy.shape[0]}")
    return lf_back_xy[perm].copy()
