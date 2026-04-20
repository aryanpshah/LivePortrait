"""
MediaPipe FaceMesh Landmark Extractor and Asymmetry Analysis

This module provides utilities to extract 468 facial landmarks using MediaPipe FaceMesh
in static image mode. It computes donor asymmetry signals by comparing a donor image
with its horizontally flipped version (donor-mirrored), producing clean delta fields
with robust weighting and comprehensive debug outputs.

Key concept:
- Δ (delta) = donor landmarks - flipped-and-mirrored donor landmarks
  This represents the inherent left-right asymmetry in the donor's facial geometry.

Coordinate system:
- MediaPipe gives normalized coordinates [0, 1] in (x, y) format.
- Conversion to pixel: x_px = x_norm * (W - 1), y_px = y_norm * (H - 1)
- Clipped to [0, W-1] and [0, H-1] respectively.
- Horizontal flip: x_flipped = W - 1 - x
"""

import os
import json
import math
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# LANDMARK INDEX DEFINITIONS
# =====================================================================

# Lips: outer ring + inner mouth cavity (outer + inner combined)
FACEMESH_LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
    310, 311, 312, 13, 82, 81, 42, 183, 78
]

# Lip corner anchors (left and right mouth corners for lateral mouth stability)
FACEMESH_LIP_CORNER_IDX = [61, 291]  # left (61) and right (291) mouth corners

# Lip midline hints (upper and lower lip center hints)
FACEMESH_LIP_MID_TOP_IDX = [0, 13]   # upper lip midline
FACEMESH_LIP_MID_BOT_IDX = [17, 14]  # lower lip midline

# Face oval / jawline contour scaffold (main stability scaffold for lower-face registration)
FACEMESH_FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Jaw region (initialized as full face oval, can be refined)
FACEMESH_JAW_IDX = FACEMESH_FACE_OVAL_IDX

# Anchor indices for bias removal and stabilization
# These points are used to compute and remove global translation bias from asymmetry delta
FACEMESH_ANCHOR_IDX = [10, 152, 234, 454]
# 10: forehead-ish (top center)
# 152: chin (bottom center)
# 234: left cheek anchor
# 454: right cheek anchor

# Cheek seeds for dynamic patch generation
CHEEK_SEEDS = [234, 454]  # left and right cheek seed points


# =====================================================================
# FACEMESH EXTRACTOR CLASS
# =====================================================================

class FaceMeshLandmarkExtractor:
    """
    Lazy wrapper around MediaPipe FaceMesh for static image mode.

    Features:
    - Deterministic initialization
    - Lazy model loading
    - Returns ok (bool), lm_px (468,2), lm_vis (468,)
    - Safe failure handling (no crashes on missing face)
    """

    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1,
                 model_path: str = "pretrained_weights/liveportrait/face_landmarker.task",
                 verbose: bool = True, refine_landmarks: bool = False, debug: bool = False):
        """
        Initialize FaceMesh extractor.

        Args:
            static_image_mode: If True, use static image mode (video mode not used).
            max_num_faces: Maximum number of faces to detect.
            model_path: Path to the mediapipe .task model file.
            verbose: Print warnings on extraction failures.
            refine_landmarks: Keep refined (478) landmarks when available. Default False (468).
            debug: Emit additional debug prints when True.
        """
        self.running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        self.max_num_faces = max_num_faces
        self.model_path = model_path
        self.verbose = verbose
        self.refine_landmarks = refine_landmarks
        self.debug = debug
        self._face_landmarker = None

    def _init_model(self):
        """Lazy initialization of MediaPipe FaceLandmarker."""
        if self._face_landmarker is None:
            if not os.path.exists(self.model_path):
                print(f"[FaceMesh] FATAL: MediaPipe model file not found at '{self.model_path}'")
                print("[FaceMesh] Please download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
                # Raising an exception is better than letting it fail later
                raise FileNotFoundError(f"MediaPipe model file not found: {self.model_path}")

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=self.running_mode,
                num_faces=self.max_num_faces,
                output_face_blendshapes=False, # Not needed for this task
                output_facial_transformation_matrixes=False # Not needed for this task
            )
            self._face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, img_rgb: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract FaceMesh landmarks from RGB image.

        Args:
            img_rgb: RGB image as numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            Tuple of:
            - ok (bool): True if face detected and landmarks extracted successfully.
            - lm_px (ndarray): Shape (468, 2), float32, pixel coordinates (x, y).
                               None if ok=False.
            - lm_vis (ndarray): Shape (468,), float32, visibility/confidence scores.
                                Filled with ones if not available. None if ok=False.
        """
        try:
            self._init_model()
        except FileNotFoundError as e:
            if self.verbose:
                # The _init_model method already prints a detailed message.
                # We just print a summary here and return failure.
                print(f"[FaceMesh] Extraction failed due to missing model file.")
            return False, None, None

        try:
            img_rgb = np.asarray(img_rgb)
            if img_rgb.dtype != np.uint8:
                img_rgb = img_rgb.astype(np.uint8)

            if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
                raise ValueError(f"Expected RGB image (H,W,3), got {img_rgb.shape}")

            # MediaPipe is strict about contiguity; enforce C-contiguous layout.
            img_rgb = np.ascontiguousarray(img_rgb)

            if self.debug:
                print(
                    "[FaceMesh DEBUG] shape",
                    img_rgb.shape,
                    "dtype",
                    img_rgb.dtype,
                    "contig",
                    img_rgb.flags["C_CONTIGUOUS"],
                    "strides",
                    img_rgb.strides,
                )

            H, W = img_rgb.shape[:2]
            if H <= 0 or W <= 0:
                if self.verbose:
                    print(f"[FaceMesh] Invalid image shape: {img_rgb.shape}")
                return False, None, None

            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Run FaceLandmarker inference
            results = self._face_landmarker.detect(mp_image)

            if not results.face_landmarks or len(results.face_landmarks) == 0:
                if self.verbose:
                    print("[FaceMesh] No face detected in image")
                return False, None, None

            # Extract first face's landmarks
            face_landmarks = results.face_landmarks[0]
            expected_landmarks = 478 if self.refine_landmarks else 468
            num_landmarks = len(face_landmarks)

            landmarks_iter = face_landmarks
            if num_landmarks == 478 and not self.refine_landmarks:
                if self.verbose:
                    print("[FaceMesh] Warning: Expected 468 landmarks, got 478; truncating to 468 (iris/refine output)")
                landmarks_iter = face_landmarks[:468]
                num_landmarks = 468
            elif num_landmarks != expected_landmarks and self.verbose:
                print(f"[FaceMesh] Warning: Expected {expected_landmarks} landmarks, but found {num_landmarks}. This may be due to a model mismatch.")

            # Convert normalized [0, 1] coords to pixel coordinates
            lm_px = []
            lm_vis = []

            for lm in landmarks_iter:
                x_norm = lm.x
                y_norm = lm.y

                # Convert to pixel coordinates
                x_px = x_norm * (W - 1)
                y_px = y_norm * (H - 1)

                # Clip to valid range
                x_px = np.clip(x_px, 0, W - 1)
                y_px = np.clip(y_px, 0, H - 1)

                lm_px.append([x_px, y_px])

                # Extract visibility if available, otherwise fallback
                vis = getattr(lm, 'visibility', getattr(lm, 'presence', 1.0))
                lm_vis.append(vis)

            lm_px = np.array(lm_px, dtype=np.float32)
            lm_vis = np.array(lm_vis, dtype=np.float32)

            # The new API guarantees 478 landmarks if blendshapes are on, 468 if off.
            # We have blendshapes off, so we expect 468.
            if lm_px.shape[0] != expected_landmarks and self.verbose:
                print(f"[FaceMesh] Warning: Landmarks count after processing is {lm_px.shape[0]}, expected {expected_landmarks}.")

            assert lm_px.shape[1] == 2, f"Expected shape (*, 2), got {lm_px.shape}"
            assert lm_vis.shape[0] == lm_px.shape[0], f"Landmark and visibility count mismatch: {lm_px.shape[0]} vs {lm_vis.shape[0]}"

            return True, lm_px, lm_vis

        except Exception as e:
            if self.verbose:
                import traceback
                print(f"[FaceMesh] Extraction failed: {e}")
                # traceback.print_exc() # Uncomment for full debug trace
            return False, None, None

    def cleanup(self):
        """Release resources."""
        if self._face_landmarker is not None:
            self._face_landmarker.close()
            self._face_landmarker = None

def draw_facemesh_overlay(
    img_rgb: np.ndarray,
    lm_px: np.ndarray,
    groups_dict: Dict[str, List[int]],
    out_path: str,
    label_some: bool = True
) -> None:
    """
    Draw FaceMesh landmarks with group-based coloring.

    Args:
        img_rgb: RGB image, shape (H, W, 3), dtype uint8.
        lm_px: Landmark pixels, shape (468, 2), float32.
        groups_dict: Dict mapping group names to landmark index lists.
        out_path: Output file path for saving the overlay.
        label_some: If True, label sanity-check indices.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Color palette for different groups
    colors = {
        "lips": (50, 220, 255),         # orange-yellow
        "face_oval": (0, 255, 0),       # green
        "cheek_patch": (255, 128, 0),   # cyan
        "anchors": (255, 0, 0),         # red
        "all": (200, 200, 200)          # gray
    }

    # Draw all landmarks first (gray)
    for i in range(468):
        pt = tuple(np.round(lm_px[i]).astype(np.int32))
        pt = (np.clip(pt[0], 0, canvas.shape[1]-1),
              np.clip(pt[1], 0, canvas.shape[0]-1))
        cv2.circle(canvas_bgr, pt, 2, colors["all"], -1, lineType=cv2.LINE_AA)

    # Draw groups on top with their specific colors
    for group_name, idx_list in groups_dict.items():
        color = colors.get(group_name, (200, 200, 200))
        for idx in idx_list:
            if 0 <= idx < 468:
                pt = tuple(np.round(lm_px[idx]).astype(np.int32))
                pt = (np.clip(pt[0], 0, canvas.shape[1]-1),
                      np.clip(pt[1], 0, canvas.shape[0]-1))
                cv2.circle(canvas_bgr, pt, 3, color, -1, lineType=cv2.LINE_AA)

    # Label sanity-check indices
    if label_some:
        sanity_indices = [10, 152, 234, 454, 61, 291, 0, 17]
        for idx in sanity_indices:
            if 0 <= idx < 468:
                pt = tuple(np.round(lm_px[idx]).astype(np.int32))
                pt = (np.clip(pt[0], 0, canvas.shape[1]-1),
                      np.clip(pt[1], 0, canvas.shape[0]-1))
                cv2.putText(
                    canvas_bgr, str(idx),
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA
                )

    cv2.imwrite(out_path, canvas_bgr)


def draw_delta_heatmap(
    img_rgb: np.ndarray,
    lm_px: np.ndarray,
    delta: np.ndarray,
    roi_indices: List[int],
    out_path: str,
    scale_px: float = 360.0,
    min_arrow_px: float = 4.0
) -> None:
    """
    Draw delta vectors as arrows on the image.

    Args:
        img_rgb: RGB image, shape (H, W, 3), dtype uint8.
        lm_px: Landmark pixels, shape (468, 2), float32.
        delta: Delta vectors, shape (468, 2), float32.
        roi_indices: List of landmark indices to draw deltas for.
        out_path: Output file path.
        scale_px: Scaling factor for delta vector display.
        min_arrow_px: Minimum arrow length in pixels.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Compute magnitude for coloring
    mag = np.linalg.norm(delta, axis=1)
    mag_max = np.percentile(mag[roi_indices] if roi_indices else mag, 95) + 1e-8

    for idx in roi_indices:
        if 0 <= idx < 468:
            u, v = lm_px[idx]
            du = delta[idx, 0] * scale_px
            dv = delta[idx, 1] * scale_px

            arrow_mag = math.hypot(du, dv)
            if arrow_mag < 1e-6:
                continue

            if arrow_mag < min_arrow_px:
                scale_f = min_arrow_px / (arrow_mag + 1e-8)
                du *= scale_f
                dv *= scale_f

            start = (int(round(u)), int(round(v)))
            end = (int(round(u + du)), int(round(v + dv)))

            # Color based on magnitude
            norm_mag = min(1.0, mag[idx] / mag_max)
            color = (int(255 * norm_mag), int(100 * (1 - norm_mag)), int(200))

            cv2.arrowedLine(canvas_bgr, start, end, color, 1, tipLength=0.3)

    cv2.imwrite(out_path, canvas_bgr)


def draw_comparison_overlay(
    img_rgb: np.ndarray,
    lm_1: np.ndarray,
    lm_2: np.ndarray,
    out_path: str,
    label_1: str = "original",
    label_2: str = "flipped"
) -> None:
    """
    Draw two sets of landmarks on the same image with different colors for comparison.

    Args:
        img_rgb: RGB image, shape (H, W, 3), dtype uint8.
        lm_1: First landmark set, shape (468, 2), float32.
        lm_2: Second landmark set, shape (468, 2), float32.
        out_path: Output file path.
        label_1: Name/label for first landmark set.
        label_2: Name/label for second landmark set.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    H, W = canvas.shape[:2]

    # Draw first set (cyan)
    for i in range(468):
        pt = tuple(np.round(lm_1[i]).astype(np.int32))
        pt = (np.clip(pt[0], 0, W-1), np.clip(pt[1], 0, H-1))
        cv2.circle(canvas_bgr, pt, 2, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    # Draw second set (magenta)
    for i in range(468):
        pt = tuple(np.round(lm_2[i]).astype(np.int32))
        pt = (np.clip(pt[0], 0, W-1), np.clip(pt[1], 0, H-1))
        cv2.circle(canvas_bgr, pt, 2, (255, 0, 255), -1, lineType=cv2.LINE_AA)

    cv2.putText(canvas_bgr, f"{label_1} (cyan)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"{label_2} (magenta)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas_bgr)


def draw_flipback_indexed_lines(
    img_rgb: np.ndarray,
    L_d: np.ndarray,
    L_f_back_perm: np.ndarray,
    num_indices: int = 50,
    out_path: str = "outputs/diagnostics/facemesh/donor_flip_back_indexed_lines.png"
) -> None:
    """
    Draw lines connecting aligned donor and flip-back landmarks.

    For a subset of evenly-spaced landmark indices, draw a line from L_d[i] to
    L_f_back_perm[i]. After alignment, these lines should be short (well-matched pairs).

    Args:
        img_rgb: RGB image, shape (H, W, 3), dtype uint8.
        L_d: Donor landmarks, shape (N, 2), float32.
        L_f_back_perm: Permuted flip-back landmarks, shape (N, 2), float32.
        num_indices: Number of indices to draw (evenly spaced).
        out_path: Output file path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    H, W = canvas.shape[:2]
    N = L_d.shape[0]

    # Select evenly-spaced indices
    if num_indices >= N:
        selected_indices = list(range(N))
    else:
        selected_indices = np.linspace(0, N - 1, num_indices, dtype=np.int32).tolist()

    # Draw lines from L_d[i] to L_f_back_perm[i]
    for i in selected_indices:
        pt1 = tuple(np.round(L_d[i]).astype(np.int32))
        pt1 = (np.clip(pt1[0], 0, W - 1), np.clip(pt1[1], 0, H - 1))

        pt2 = tuple(np.round(L_f_back_perm[i]).astype(np.int32))
        pt2 = (np.clip(pt2[0], 0, W - 1), np.clip(pt2[1], 0, H - 1))

        # Draw line (green for good alignment)
        cv2.line(canvas_bgr, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        # Draw points
        cv2.circle(canvas_bgr, pt1, 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)  # Blue = donor
        cv2.circle(canvas_bgr, pt2, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)  # Red = flip-back

    cv2.putText(canvas_bgr, "Blue = Donor | Red = Flip-Back | Green = Match", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas_bgr)


# =====================================================================
# REGION CONFIGURATION UTILITIES
# =====================================================================

def load_facemesh_regions_config(config_path: Optional[str] = None) -> Dict[str, List[int]]:
    """
    Load landmark region indices from JSON config.

    If config_path is None or doesn't exist, returns default indices.

    Expected JSON format:
    {
        "lips": [...],
        "face_oval": [...],
        "anchors": [...],
        "cheek_seeds": [234, 454]
    }

    Args:
        config_path: Path to JSON config file.

    Returns:
        Dictionary with region indices.
    """
    default_config = {
        "lips": FACEMESH_LIPS_IDX,
        "face_oval": FACEMESH_FACE_OVAL_IDX,
        "anchors": FACEMESH_ANCHOR_IDX,
        "cheek_seeds": CHEEK_SEEDS,
    }

    if config_path is None or not os.path.exists(config_path):
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Merge with defaults, allowing overrides
        config = {**default_config, **config}
        return config
    except Exception as e:
        print(f"[FaceMesh] Failed to load config from {config_path}: {e}, using defaults")
        return default_config


def compute_dynamic_cheek_patch(
    lm_px: np.ndarray,
    cheek_seeds: List[int] = CHEEK_SEEDS,
    radius_fraction: float = 0.15,
    radius_min_px: float = 40.0,
    radius_max_px: float = 80.0
) -> List[int]:
    """
    Compute dynamic cheek patch indices based on landmark geometry.

    Cheek patch is all landmarks within a radius of cheek seed points.

    Args:
        lm_px: Landmark pixels, shape (468, 2), float32.
        cheek_seeds: Indices of cheek seed points.
        radius_fraction: Radius as fraction of cheek-to-cheek distance.
        radius_min_px: Minimum radius in pixels.
        radius_max_px: Maximum radius in pixels.

    Returns:
        List of landmark indices in the cheek patch.
    """
    if len(cheek_seeds) < 2 or any(i >= 468 for i in cheek_seeds):
        return []

    # Compute cheek-to-cheek distance (face scale)
    cheek_1 = lm_px[cheek_seeds[0]]
    cheek_2 = lm_px[cheek_seeds[1]]
    face_scale = np.linalg.norm(cheek_2 - cheek_1)

    # Compute radius
    radius_px = radius_fraction * face_scale
    radius_px = np.clip(radius_px, radius_min_px, radius_max_px)

    # Find all landmarks within radius of either cheek seed
    cheek_patch = set()
    for i in range(468):
        dist_to_seed_1 = np.linalg.norm(lm_px[i] - cheek_1)
        dist_to_seed_2 = np.linalg.norm(lm_px[i] - cheek_2)

        if dist_to_seed_1 <= radius_px or dist_to_seed_2 <= radius_px:
            cheek_patch.add(i)

    return sorted(list(cheek_patch))


# =====================================================================
# INDEX ALIGNMENT FOR FLIP-BACK LANDMARKS
# =====================================================================

def align_flipback_indices(
    L_d: np.ndarray,
    L_f_back: np.ndarray,
    k: int = 3
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Align flip-back landmarks to donor landmarks by finding optimal index correspondence.

    After horizontal flip and mirror-back, landmark indices may not align because the flip
    operation reorders them (left/right swap). This function computes a permutation that
    maps flip-back landmarks to their best matches in the donor frame.

    Algorithm:
    1. Compute distance matrix D[i,j] = ||L_d[i] - L_f_back[j]||
    2. For each donor point i, collect top-k nearest flip-back candidates j
    3. Build candidate edges (cost, i, j) and sort by cost
    4. Greedy assignment: if both i and j are unassigned, assign perm[i]=j
    5. For remaining unassigned points, assign to nearest available match
    6. Verify permutation is valid (bijection)

    Args:
        L_d: Donor landmarks, shape (N, 2)
        L_f_back: Flip-back landmarks, shape (N, 2)
        k: Number of top-k nearest neighbors to consider per point (default 3)

    Returns:
        L_f_back_perm: Permuted flip-back landmarks, shape (N, 2)
        perm: Permutation array, shape (N,) - perm[i] = j means L_f_back[j] -> L_d[i]
        stats: Dictionary with alignment statistics:
            - before_dist: distances before alignment (mean, median, p95, max)
            - after_dist: distances after alignment (mean, median, p95, max)
            - before_mean: mean distance before
            - after_mean: mean distance after
            - fractions: fraction of points within 2/5/10 pixels

    Raises:
        RuntimeError: If permutation is not valid (not a bijection)
    """
    N = L_d.shape[0]
    if L_f_back.shape[0] != N:
        raise ValueError(
            f"Shape mismatch: L_d has {N} points, L_f_back has {L_f_back.shape[0]}"
        )

    # Compute distance matrix: D[i,j] = ||L_d[i] - L_f_back[j]||
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            diff = L_d[i] - L_f_back[j]
            D[i, j] = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

    # Build candidate edges: (cost, i, j) for each i's top-k neighbors
    candidates = []
    for i in range(N):
        # Get indices of k smallest distances for this donor point
        top_k_indices = np.argsort(D[i])[:k]
        for j in top_k_indices:
            candidates.append((D[i, j], i, j))

    # Sort candidates by cost (ascending), then by i, then by j for determinism
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    # Greedy assignment
    perm = np.full(N, -1, dtype=np.int32)
    used_j = set()

    for cost, i, j in candidates:
        if perm[i] == -1 and j not in used_j:
            perm[i] = j
            used_j.add(j)

    # Assign remaining unassigned points to nearest available
    for i in range(N):
        if perm[i] == -1:
            # Find nearest unused j
            distances_for_i = D[i].copy()
            distances_for_i[list(used_j)] = np.inf
            nearest_j = np.argmin(distances_for_i)
            perm[i] = nearest_j
            used_j.add(nearest_j)

    # Verify permutation is a bijection
    if len(used_j) != N or len(set(perm)) != N:
        raise RuntimeError(
            f"Alignment failed: permutation is not a bijection. "
            f"Unique values: {len(set(perm))}, expected {N}"
        )

    # Compute aligned landmarks
    L_f_back_perm = L_f_back[perm]

    # Compute statistics
    before_dist = np.linalg.norm(L_d - L_f_back, axis=1)
    after_dist = np.linalg.norm(L_d - L_f_back_perm, axis=1)

    stats = {
        "before_dist": {
            "mean": float(np.mean(before_dist)),
            "median": float(np.median(before_dist)),
            "p95": float(np.percentile(before_dist, 95)),
            "max": float(np.max(before_dist)),
        },
        "after_dist": {
            "mean": float(np.mean(after_dist)),
            "median": float(np.median(after_dist)),
            "p95": float(np.percentile(after_dist, 95)),
            "max": float(np.max(after_dist)),
        },
        "before_mean": float(np.mean(before_dist)),
        "after_mean": float(np.mean(after_dist)),
        "improvement_factor": float(np.mean(before_dist) / (np.mean(after_dist) + 1e-8)),
        "fractions": {
            "lt_2px": float(np.mean(after_dist <= 2.0)),
            "lt_5px": float(np.mean(after_dist <= 5.0)),
            "lt_10px": float(np.mean(after_dist <= 10.0)),
        },
    }

    return L_f_back_perm, perm, stats


# =====================================================================
# DONOR ASYMMETRY COMPUTATION
# =====================================================================

def compute_donor_asymmetry_delta(
    donor_rgb: np.ndarray,
    extractor: FaceMeshLandmarkExtractor,
    regions_config: Dict[str, List[int]],
    apply_bias_removal: bool = True,
    max_delta_px: Optional[float] = None,
    clamp_percentile: float = 98.0,
    cheek_radius_px: Optional[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute donor asymmetry delta signal.

    This function:
    1. Extracts landmarks on donor image
    2. Creates horizontally flipped donor
    3. Extracts landmarks on flipped donor
    4. Mirrors flipped landmarks back into donor coordinate frame
    5. Computes delta (L_d - L_f_back)
    6. Optionally removes global bias (via anchor points)
    7. Optionally clamps extreme values

    Args:
        donor_rgb: Donor RGB image, shape (H, W, 3), dtype uint8.
        extractor: FaceMeshLandmarkExtractor instance.
        regions_config: Region indices config (from load_facemesh_regions_config).
        apply_bias_removal: If True, remove global translation bias via anchors.
        max_delta_px: Absolute maximum delta magnitude in pixels. None = auto (percentile).
        clamp_percentile: Percentile for auto clamping (default 98).
        cheek_radius_px: Override radius for dynamic cheek patch (default auto-computed).
        verbose: Print diagnostic info.

    Returns:
        Dictionary with keys:
        - 'ok' (bool): Success flag
        - 'L_d' (ndarray): Donor landmarks, shape (468, 2), float32
        - 'L_f' (ndarray): Flipped donor landmarks (in flipped frame), shape (468, 2)
        - 'L_f_back' (ndarray): Flipped landmarks mirrored back, shape (468, 2)
        - 'delta' (ndarray): Asymmetry delta, shape (468, 2), float32
        - 'weights' (ndarray): Region weights, shape (468,), float32
        - 'magnitude' (ndarray): Delta magnitudes, shape (468,), float32
        - 'regions' (dict): Per-region statistics
        - 'bias' (ndarray): Global bias removed (if applied), shape (2,)
        - 'clamp_params' (dict): Clamping parameters used
        - 'summary' (dict): Summary statistics
    """

    os.makedirs("outputs/diagnostics/facemesh", exist_ok=True)

    result = {"ok": False}

    try:
        H, W = donor_rgb.shape[:2]

        # Extract landmarks on donor
        ok_d, L_d, vis_d = extractor.extract(donor_rgb)
        if not ok_d or L_d is None:
            print("[FaceMesh] Failed to extract landmarks on donor image")
            return result

        if verbose:
            print(f"[FaceMesh] Extracted {L_d.shape[0]} landmarks on donor")

        # Create flipped donor
        donor_flip = donor_rgb[:, ::-1, :].copy()

        # Extract landmarks on flipped donor
        ok_f, L_f, vis_f = extractor.extract(donor_flip)
        if not ok_f or L_f is None:
            print("[FaceMesh] Failed to extract landmarks on flipped donor image")
            return result

        if verbose:
            print(f"[FaceMesh] Extracted {L_f.shape[0]} landmarks on flipped donor")

        # Mirror flipped landmarks back into donor frame
        L_f_back = L_f.copy()
        L_f_back[:, 0] = (W - 1) - L_f[:, 0]  # Mirror x coordinate
        L_f_back[:, 1] = L_f[:, 1]  # Keep y coordinate

        # Align flip-back landmarks to donor landmarks (fix index swap from flip)
        flipback_alignment_error_before = None
        flipback_alignment_error_after = None
        flipback_alignment_quality = None
        flipback_perm = None
        try:
            L_f_back_perm, perm, alignment_stats = align_flipback_indices(L_d, L_f_back, k=3)
            L_f_back = L_f_back_perm
            flipback_perm = perm
            flipback_alignment_error_before = alignment_stats["before_dist"]
            flipback_alignment_error_after = alignment_stats["after_dist"]
            flipback_alignment_quality = alignment_stats["fractions"]
            if verbose:
                print(
                    f"[FaceMesh] Aligned flip-back landmarks: "
                    f"error before={alignment_stats['before_mean']:.3f}px, "
                    f"after={alignment_stats['after_mean']:.3f}px, "
                    f"improvement={alignment_stats['improvement_factor']:.2f}x"
                )
        except Exception as e:
            print(f"[FaceMesh] WARNING: Could not align flip-back landmarks: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

        # Compute raw delta (now with aligned flip-back)
        delta_raw = L_d - L_f_back

        # Apply bias removal (via anchor points)
        bias = np.zeros(2, dtype=np.float32)
        if apply_bias_removal and len(regions_config["anchors"]) > 0:
            anchor_indices = regions_config["anchors"]
            anchor_indices = [i for i in anchor_indices if 0 <= i < 468]
            if len(anchor_indices) > 0:
                bias = delta_raw[anchor_indices].mean(axis=0).astype(np.float32)
                delta = delta_raw - bias
                if verbose:
                    print(f"[FaceMesh] Applied bias removal: {bias}")
            else:
                delta = delta_raw.copy()
        else:
            delta = delta_raw.copy()

        # Compute dynamic cheek patch
        if cheek_radius_px is not None:
            cheek_patch = compute_dynamic_cheek_patch(
                L_d, regions_config["cheek_seeds"],
                radius_fraction=0.15,  # dummy; will be overridden by radius_min/max
                radius_min_px=cheek_radius_px,
                radius_max_px=cheek_radius_px
            )
        else:
            cheek_patch = compute_dynamic_cheek_patch(
                L_d, regions_config["cheek_seeds"]
            )

        # Magnitude computation
        magnitude = np.linalg.norm(delta, axis=1).astype(np.float32)

        # Determine clamping limit
        roi_union = set(
            regions_config["lips"] +
            regions_config["face_oval"] +
            cheek_patch
        )
        roi_union = [i for i in roi_union if 0 <= i < 468]

        if max_delta_px is not None and max_delta_px > 0:
            clamp_max = max_delta_px
            clamp_method = "absolute"
        else:
            if len(roi_union) > 0:
                clamp_max = np.percentile(magnitude[roi_union], clamp_percentile)
            else:
                clamp_max = np.percentile(magnitude, clamp_percentile)
            clamp_method = f"percentile_{clamp_percentile}"

        # Apply magnitude clamping
        for i in range(468):
            if magnitude[i] > clamp_max:
                scale = clamp_max / (magnitude[i] + 1e-8)
                delta[i] *= scale
                magnitude[i] = clamp_max

        if verbose:
            print(f"[FaceMesh] Applied clamping: {clamp_method} = {clamp_max:.4f} px")

        # Compute weighting scheme (use actual number of points N)
        N = L_d.shape[0]
        weights = np.zeros(N, dtype=np.float32)
        weights[regions_config["lips"]] = 1.0
        weights[regions_config["face_oval"]] = 0.7
        weights[cheek_patch] = 0.5
        weights[regions_config["anchors"]] = 0.9

        # Compute per-region statistics
        N = L_d.shape[0]
        regions_stats = {}
        for region_name, region_indices in [
            ("lips", regions_config["lips"]),
            ("face_oval", regions_config["face_oval"]),
            ("cheek_patch", cheek_patch),
            ("anchors", regions_config["anchors"]),
        ]:
            region_indices = [i for i in region_indices if 0 <= i < N]
            if len(region_indices) > 0:
                region_mags = magnitude[region_indices]
                regions_stats[region_name] = {
                    "mean_mag": float(region_mags.mean()),
                    "max_mag": float(region_mags.max()),
                    "count": len(region_indices),
                }
            else:
                regions_stats[region_name] = {
                    "mean_mag": 0.0,
                    "max_mag": 0.0,
                    "count": 0,
                }

        # Summary statistics
        summary = {
            "bias_removed": apply_bias_removal,
            "bias_vector": bias.tolist() if apply_bias_removal else None,
            "clamp_method": clamp_method,
            "clamp_max_px": float(clamp_max),
            "overall_mean_mag": float(magnitude.mean()),
            "overall_max_mag": float(magnitude.max()),
            "overall_min_mag": float(magnitude.min()),
            "donor_image_shape": (H, W),
            "cheek_radius_px": float(cheek_radius_px) if cheek_radius_px else "auto",
            "cheek_seeds": regions_config["cheek_seeds"],
            "flipback_alignment_error_before": flipback_alignment_error_before,
            "flipback_alignment_error_after": flipback_alignment_error_after,
            "flipback_alignment_quality": flipback_alignment_quality,
        }

        result = {
            "ok": True,
            "L_d": L_d.astype(np.float32),
            "L_f": L_f.astype(np.float32),
            "L_f_back": L_f_back.astype(np.float32),
            "delta": delta.astype(np.float32),
            "weights": weights,
            "magnitude": magnitude,
            "vis_d": vis_d,
            "vis_f": vis_f,
            "regions": regions_stats,
            "bias": bias,
            "cheek_patch": cheek_patch,
            "flipback_perm": flipback_perm,
            "clamp_params": {
                "method": clamp_method,
                "max_px": clamp_max,
                "percentile": clamp_percentile,
            },
            "summary": summary,
        }

        if verbose:
            print("[FaceMesh] Asymmetry computation complete:")
            print(f"  Overall |D| mean={summary['overall_mean_mag']:.4f}, max={summary['overall_max_mag']:.4f}")
            for rname, rstats in regions_stats.items():
                if rstats["count"] > 0:
                    print(f"  {rname}: mean={rstats['mean_mag']:.4f}, max={rstats['max_mag']:.4f}, count={rstats['count']}")

        return result

    except Exception as e:
        print(f"[FaceMesh] Error in compute_donor_asymmetry_delta: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False}


# =====================================================================
# CONVENIENCE WRAPPERS AND DUMP FUNCTIONS
# =====================================================================

def save_facemesh_numpy_dumps(
    result: Dict,
    output_dir: str = "outputs/diagnostics/facemesh"
) -> None:
    """
    Save numpy arrays from asymmetry computation result.

    Args:
        result: Result dictionary from compute_donor_asymmetry_delta.
        output_dir: Output directory for .npy files.
    """
    if not result.get("ok", False):
        print("[FaceMesh] Cannot save dumps: result not ok")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        np.save(os.path.join(output_dir, "donor_landmarks.npy"), result["L_d"])
        np.save(os.path.join(output_dir, "donor_flip_landmarks.npy"), result["L_f"])
        np.save(os.path.join(output_dir, "donor_flip_back_landmarks.npy"), result["L_f_back"])
        np.save(os.path.join(output_dir, "delta.npy"), result["delta"])
        np.save(os.path.join(output_dir, "weights.npy"), result["weights"])

        delta_weighted = result["delta"] * result["weights"][:, np.newaxis]
        np.save(os.path.join(output_dir, "delta_weighted.npy"), delta_weighted)

        # Save flip-back permutation if available
        if result.get("flipback_perm") is not None:
            np.save(os.path.join(output_dir, "flipback_perm.npy"), result["flipback_perm"])

        print(f"[FaceMesh] Saved numpy dumps to {output_dir}")
    except Exception as e:
        print(f"[FaceMesh] Failed to save numpy dumps: {e}")


def save_facemesh_summary(
    result: Dict,
    output_path: str = "outputs/diagnostics/facemesh/summary.json"
) -> None:
    """
    Save summary JSON from asymmetry computation result.

    Args:
        result: Result dictionary from compute_donor_asymmetry_delta.
        output_path: Output path for summary.json.
    """
    if not result.get("ok", False):
        print("[FaceMesh] Cannot save summary: result not ok")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    try:
        summary_data = {
            "summary": convert_to_serializable(result.get("summary", {})),
            "regions": convert_to_serializable(result.get("regions", {})),
            "clamp_params": convert_to_serializable(result.get("clamp_params", {})),
        }

        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"[FaceMesh] Saved summary to {output_path}")
    except Exception as e:
        print(f"[FaceMesh] Failed to save summary: {e}")
