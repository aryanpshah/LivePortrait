"""Small FaceMesh helpers used by mouth TPS code."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Same lip edges as mediapipe.solutions.face_mesh_connections (468-landmark mesh).
# Kept here so lip index lists still work if someone has mediapipe>=0.10.30 (no solutions API).
_FACEMESH_LIPS_FALLBACK = frozenset(
    [
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    ]
)


def _lip_pairs():
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        return mp.solutions.face_mesh_connections.FACEMESH_LIPS
    return _FACEMESH_LIPS_FALLBACK


def facemesh_lips_indices() -> list[int]:
    pts: set[int] = set()
    for a, b in _lip_pairs():
        pts.add(a)
        pts.add(b)
    return sorted(pts)


def default_facemesh_regions() -> dict:
    return {"FACEMESH_LIPS_IDX": facemesh_lips_indices()}


class FaceMeshExtractor:
    """Get 468 FaceMesh points for one RGB image."""

    def __init__(
        self,
        static_image_mode: bool = True,
        refine_landmarks: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise ImportError(
                "mediapipe is missing the legacy solutions API (removed in 0.10.30+). "
                "Mouth TPS needs Face Mesh: pip install 'mediapipe>=0.10.14,<0.10.30'"
            )

        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def process_rgb(self, rgb_u8: np.ndarray) -> Optional[np.ndarray]:
        if rgb_u8 is None or rgb_u8.size == 0:
            return None
        h, w = rgb_u8.shape[:2]
        res = self._mesh.process(rgb_u8)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        out = np.zeros((468, 2), dtype=np.float64)
        for i in range(468):
            out[i, 0] = lm[i].x * w
            out[i, 1] = lm[i].y * h
        return out

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "FaceMeshExtractor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
