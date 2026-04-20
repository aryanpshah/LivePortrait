import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np

# Make src importable when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from keypoint_map import kp_to_pixels


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_points(canvas: np.ndarray, pts: np.ndarray, color_bgr: tuple[int, int, int], r: int) -> None:
    h, w = canvas.shape[:2]
    pts_i = np.round(pts).astype(np.int32)
    pts_i[:, 0] = np.clip(pts_i[:, 0], 0, w - 1)
    pts_i[:, 1] = np.clip(pts_i[:, 1], 0, h - 1)
    for x, y in pts_i:
        cv2.circle(canvas, (int(x), int(y)), r, color_bgr, -1, lineType=cv2.LINE_AA)


def extract_points_from_lip_overlay(overlay_bgr: np.ndarray, expected_count: int) -> np.ndarray:
    """
    Extract mouth control point centers from lip_ctrl_src_overlay.png.
    Points there are drawn in yellow-ish BGR=(0,255,255), anti-aliased.
    """
    hsv = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 120, 120), (40, 255, 255))
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    samples = np.stack([xs, ys], axis=1).astype(np.float32)

    k = int(max(1, min(expected_count, samples.shape[0])))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _compactness, _labels, centers = cv2.kmeans(
        samples,
        k,
        None,
        criteria,
        6,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay 21 LivePortrait KPs + 40 FaceMesh mouth KPs.")
    ap.add_argument(
        "--image",
        "-i",
        default="outputs/debug_diag_right/liveportrait_base.png",
        help="Input face image.",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="outputs/debug_diag_right/combined_kp_map.png",
        help="Output overlay image.",
    )
    ap.add_argument(
        "--mouth-overlay",
        default="outputs/debug_diag_right/lip_ctrl_src_overlay.png",
        help="Existing mouth-control overlay used to recover 40 mouth points.",
    )
    ap.add_argument(
        "--summary-json",
        default="outputs/debug_diag_right/summary.json",
        help="Optional summary.json to auto-read Nlip.",
    )
    ap.add_argument(
        "--mouth-count",
        type=int,
        default=40,
        help="Expected mouth-residual control-point count (fallback if summary missing).",
    )
    ap.add_argument(
        "--models-config",
        default=None,
        help="Optional models config path for InferenceConfig.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    img_rgb = read_rgb(args.image)
    src_h, src_w = img_rgb.shape[:2]

    cfg_kwargs = {}
    if args.models_config:
        cfg_kwargs["models_config"] = args.models_config
    wrapper = LivePortraitWrapper(InferenceConfig(**cfg_kwargs))

    source_tensor = wrapper.prepare_source(img_rgb)
    kp_info = wrapper.get_kp_info(source_tensor, flag_refine_info=True)
    transformed = wrapper.transform_keypoint(kp_info)[0][:, :2]
    model_h, model_w = wrapper.inference_cfg.input_shape
    lp_model_px = kp_to_pixels(transformed, model_h, model_w)

    lp_pixels = lp_model_px.copy()
    lp_pixels[:, 0] = np.clip(lp_pixels[:, 0] * (src_w / float(model_w)), 0, src_w - 1)
    lp_pixels[:, 1] = np.clip(lp_pixels[:, 1] * (src_h / float(model_h)), 0, src_h - 1)

    mouth_overlay_bgr = cv2.imread(args.mouth_overlay, cv2.IMREAD_COLOR)
    if mouth_overlay_bgr is None:
        raise FileNotFoundError(args.mouth_overlay)
    expected_n = int(args.mouth_count)
    if args.summary_json and osp.exists(args.summary_json):
        try:
            with open(args.summary_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            expected_n = int(payload.get("Nlip", expected_n))
        except Exception:
            pass

    mouth_pixels = extract_points_from_lip_overlay(mouth_overlay_bgr, expected_n)
    if mouth_pixels.shape[0] == 0:
        raise RuntimeError("Could not recover mouth points from --mouth-overlay.")

    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    draw_points(out, lp_pixels, color_bgr=(0, 255, 255), r=4)      # yellow: 21 LivePortrait
    draw_points(out, mouth_pixels, color_bgr=(255, 80, 255), r=2)  # magenta: ~40 mouth residual

    cv2.putText(out, f"LivePortrait: {lp_pixels.shape[0]} pts", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Mouth residual: {mouth_pixels.shape[0]} pts", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 255), 2, cv2.LINE_AA)

    out_dir = osp.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f"[combined_keypoint_map] Saved to {osp.abspath(args.output)}")
    print(f"[combined_keypoint_map] liveportrait_total={lp_pixels.shape[0]}")
    print(f"[combined_keypoint_map] mouth_residual_total={mouth_pixels.shape[0]}")


if __name__ == "__main__":
    main()
