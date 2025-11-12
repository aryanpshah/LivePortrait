import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict
import cv2
import numpy as np
import torch

# make src importable when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper


def read_rgb(path: str) -> np.ndarray:
    # Read an image as RGB
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def kp_to_pixels(kp_xy: torch.Tensor, height: int, width: int) -> np.ndarray:
    """
    Convert normalized [-1, 1] keypoints (or already-pixel coords) into pixel space.
    """
    if isinstance(kp_xy, torch.Tensor):
        xy = kp_xy.detach().float().cpu().numpy()
    else:
        xy = np.asarray(kp_xy, dtype=np.float32)

    x = xy[:, 0]
    y = xy[:, 1]

    # Heuristic: treat coordinates as normalized if their magnitude is O(1).
    if np.all(np.isfinite(xy)) and np.median(np.abs(xy)) <= 1.2:
        px = (x + 1.0) * 0.5 * (width - 1)
        py = (y + 1.0) * 0.5 * (height - 1)
    else:
        px, py = x, y

    pts = np.stack([px, py], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts


def draw_keypoint_map(
    img_rgb: np.ndarray,
    kp_pixels: np.ndarray | torch.Tensor,
    output_path: str,
) -> None:
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    canvas = img_rgb.copy()
    if isinstance(kp_pixels, torch.Tensor):
        pts = kp_pixels.detach().cpu().numpy()
    else:
        pts = np.asarray(kp_pixels, dtype=np.float32)
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, canvas.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, canvas.shape[0] - 1)

    for u, v in pts:
        cv2.circle(canvas, (int(u), int(v)), 3, (64, 224, 255), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_keypoints_npy(kp: torch.Tensor, path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    np.save(path, kp.detach().cpu().numpy())


def save_keypoints_json(kp_info: dict, path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    payload = {
        "kp": kp_info["kp"][0].detach().cpu().numpy().tolist(),
        "exp": kp_info["exp"][0].detach().cpu().numpy().tolist(),
        "t": kp_info["t"][0].detach().cpu().numpy().tolist(),
        "scale": float(kp_info["scale"][0].detach().cpu().item()),
        "pitch_deg": float(kp_info["pitch"][0].detach().cpu().item()),
        "yaw_deg": float(kp_info["yaw"][0].detach().cpu().item()),
        "roll_deg": float(kp_info["roll"][0].detach().cpu().item()),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_region_masks(kp: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    xy = kp[:, :2].to(torch.float32)
    cx, cy = xy.mean(dim=0)
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    sx = torch.std(dx) + 1e-6
    sy = torch.std(dy) + 1e-6

    xn = dx / sx
    yn = dy / sy

    y_lo = torch.quantile(yn, 0.45)
    y_hi = torch.quantile(yn, 0.95)
    x_mid = torch.quantile(xn.abs(), 0.60)
    x_wide = torch.quantile(xn.abs(), 0.90)

    lips_mask = (yn >= y_lo) & (yn <= y_hi) & (xn.abs() <= x_mid)
    corner_mask = (yn >= y_lo) & (yn <= y_hi) & (xn.abs() > x_mid) & (xn.abs() <= x_wide)
    center_lip_mask = (yn >= y_lo) & (yn <= y_hi) & (xn.abs() <= (0.6 * x_mid))

    if int(lips_mask.sum().item()) < 6:
        y_lo_f = torch.quantile(yn, 0.40)
        y_hi_f = torch.quantile(yn, 0.98)
        x_mid_f = torch.quantile(xn.abs(), 0.75)
        x_wide_f = torch.quantile(xn.abs(), 0.97)
        lips_mask = (yn >= y_lo_f) & (yn <= y_hi_f) & (xn.abs() <= x_mid_f)
        corner_mask = (yn >= y_lo_f) & (yn <= y_hi_f) & (xn.abs() > x_mid_f) & (xn.abs() <= x_wide_f)
        center_lip_mask = (yn >= y_lo_f) & (yn <= y_hi_f) & (xn.abs() <= (0.6 * x_mid_f))

    eyes_mask = (yn < torch.quantile(yn, 0.25)) & (xn.abs() > torch.quantile(xn.abs(), 0.55))
    brows_mask = (yn < torch.quantile(yn, 0.15)) & (xn.abs() > torch.quantile(xn.abs(), 0.45))
    eye_brow_mask = eyes_mask | brows_mask

    mouth_mask = lips_mask | corner_mask
    others_mask = ~(mouth_mask | eye_brow_mask)

    return OrderedDict(
        [
            ("total", torch.ones_like(lips_mask, dtype=torch.bool)),
            ("mouth_total", mouth_mask),
            ("lips", lips_mask),
            ("lip_corners", corner_mask),
            ("center_lip_band", center_lip_mask),
            ("eyes", eyes_mask),
            ("brows", brows_mask),
            ("eyes_and_brows", eye_brow_mask),
            ("other_regions", others_mask),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LivePortrait keypoint map for a single image."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/keypoints/keypoint_map.jpg",
    )
    parser.add_argument(
        "--models-config",
    )
    parser.add_argument(
        "--save-npy",
    )
    parser.add_argument(
        "--save-json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_kwargs = {}
    if args.models_config:
        cfg_kwargs["models_config"] = args.models_config

    cfg = InferenceConfig(**cfg_kwargs)
    wrapper = LivePortraitWrapper(cfg)

    img_rgb = read_rgb(args.image)
    src_h, src_w = img_rgb.shape[:2]
    source_tensor = wrapper.prepare_source(img_rgb)
    kp_info = wrapper.get_kp_info(source_tensor, flag_refine_info=True)
    kp = kp_info["kp"][0]

    region_masks = compute_region_masks(kp)

    transformed = wrapper.transform_keypoint(kp_info)[0][:, :2]
    model_h, model_w = wrapper.inference_cfg.input_shape
    kp_model_px = kp_to_pixels(transformed, model_h, model_w)

    scale_x = src_w / float(model_w)
    scale_y = src_h / float(model_h)
    kp_pixels = kp_model_px.copy()
    kp_pixels[:, 0] = np.clip(kp_pixels[:, 0] * scale_x, 0, src_w - 1)
    kp_pixels[:, 1] = np.clip(kp_pixels[:, 1] * scale_y, 0, src_h - 1)

    draw_keypoint_map(img_rgb, kp_pixels, args.output)
    print(f"[keypoint_map] Saved visualization to {osp.abspath(args.output)}")

    for name, mask in region_masks.items():
        count = int(mask.sum().item())
        print(f"[keypoint_map] {name}: {count} keypoints")

    if args.save_npy:
        save_keypoints_npy(kp, args.save_npy)
    if args.save_json:
        save_keypoints_json(kp_info, args.save_json)

if __name__ == "__main__":
    main()
