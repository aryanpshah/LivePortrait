import sys, os, os.path as osp, cv2, torch, math, json
import numpy as np
from typing import Dict, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from scripts.facemesh_landmarks import (
    FaceMeshLandmarkExtractor,
    compute_donor_asymmetry_delta,
    load_facemesh_regions_config,
    draw_facemesh_overlay,
    draw_delta_heatmap,
    draw_comparison_overlay,
    draw_flipback_indexed_lines,
    save_facemesh_numpy_dumps,
    save_facemesh_summary,
    FACEMESH_LIPS_IDX,
    FACEMESH_FACE_OVAL_IDX,
    FACEMESH_ANCHOR_IDX,
)
from scripts.facemesh_warp import (
    apply_facemesh_warp,
    validate_warp,
)
from scripts.facemesh_metrics import compute_metrics
from scripts.facemesh_exp_assist import apply_facemesh_exp_assist

# basic run command: python asym_transfer.py \--donor path/to/donor.jpg \--target path/to/target.jpg \--out outputs/asym_transfer.jpg

# Fixed mouth indices from labeled diagnostic map
LIP_IDX = [5, 17, 20, 18, 19]       # all mouth border points
LIP_CORNER_IDX = [5, 18]           # left and right mouth corners


def read_rgb(p):
    # Read an image as RGB
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_rgb(p, img_rgb):
    # Save RBG image
    os.makedirs(osp.dirname(p), exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def letterbox_to(img_rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    # Resize with aspect ratio preserved and pad to (out_h, out_w)
    H, W = img_rgb.shape[:2]
    if H == 0 or W == 0:
        raise ValueError("Invalid image size for letterbox.")
    scale = min(out_w / W, out_h / H)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=resized.dtype)
    left = (out_w - new_w) // 2
    top = (out_h - new_h) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

def resolve_output_size(target_rgb: np.ndarray, out_w: int, out_h: int):
    Ht, Wt = target_rgb.shape[:2]
    if out_w <= 0 and out_h <= 0:
        return Wt, Ht
    if out_w <= 0 and out_h > 0:
        return int(round(out_h * (Wt / Ht))), out_h
    if out_h <= 0 and out_w > 0:
        return out_w, int(round(out_w * (Ht / Wt)))
    return out_w, out_h

def procrustes_scale(src_xy: torch.Tensor, dst_xy: torch.Tensor) -> float:
    src = src_xy - src_xy.mean(dim=0, keepdim=True)
    dst = dst_xy - dst_xy.mean(dim=0, keepdim=True)
    src_norm = torch.linalg.norm(src)
    dst_norm = torch.linalg.norm(dst)
    if float(src_norm) < 1e-8:
        return 1.0
    return float((dst_norm / (src_norm + 1e-8)).item())

def similarity_decompose(src_xy: torch.Tensor, dst_xy: torch.Tensor):
    # Compute best-fit 2x2 rotation matrix and adjust rotation
    src = src_xy - src_xy.mean(dim=0, keepdim=True)
    dst = dst_xy - dst_xy.mean(dim=0, keepdim=True)
    H = src.T @ dst
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    theta = float(torch.atan2(R[1, 0], R[0, 0]).item())
    return R, theta

def affine_detrend_dx(dx: torch.Tensor, kp_xy: torch.Tensor):
    # Remove global widening from X motion by fitting: dx ~ a*(x-xc) + b*(y-yc) + c and subtracting it off
    x = kp_xy[:, 0]; y = kp_xy[:, 1]
    xc = x.mean(); yc = y.mean()
    X = torch.stack([x - xc, y - yc, torch.ones_like(x)], dim=1)
    coeff = torch.linalg.pinv(X) @ dx
    trend = X @ coeff
    return dx - trend, coeff

def affine_detrend_dy(dy: torch.Tensor, kp_xy: torch.Tensor):
    # Remove global vertical drift/scale/shear in Y: dy ~ a*(x-xc) + b*(y-yc) + c
    x = kp_xy[:, 0]; y = kp_xy[:, 1]
    xc = x.mean(); yc = y.mean()
    X = torch.stack([x - xc, y - yc, torch.ones_like(x)], dim=1)
    coeff = torch.linalg.pinv(X) @ dy
    trend = X @ coeff
    return dy - trend, coeff

def soft_knee_vec(V: torch.Tensor, tau: float) -> torch.Tensor:
    # Compress vector magnitudes to avoid extreme values
    # m' = tau * tanh(m / tau)
    m = torch.linalg.norm(V, dim=-1, keepdim=True) + 1e-8
    m_prime = float(tau) * torch.tanh(m / float(tau))
    gain = (m_prime / m).clamp(max=1.0)
    return V * gain

def soft_knee_scalar(d: torch.Tensor, tau: float) -> torch.Tensor:
    # Scalar version for X only
    a = d.abs() + 1e-8
    a_prime = float(tau) * torch.tanh(a / float(tau))
    return torch.sign(d) * a_prime

def kp_to_pixels(kp_xy: torch.Tensor | np.ndarray, height: int, width: int) -> np.ndarray:
    if isinstance(kp_xy, torch.Tensor):
        xy = kp_xy.detach().float().cpu().numpy()
    else:
        xy = np.asarray(kp_xy, dtype=np.float32)

    x = xy[:, 0]
    y = xy[:, 1]
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
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
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

def draw_keypoint_map_labeled(
    img_rgb: np.ndarray,
    kp_pixels: np.ndarray | torch.Tensor,
    output_path: str,
) -> None:
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    pts = kp_pixels.detach().cpu().numpy() if isinstance(kp_pixels, torch.Tensor) else np.asarray(kp_pixels, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, canvas.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, canvas.shape[0] - 1)

    for i, (u, v) in enumerate(pts):
        cv2.circle(canvas_bgr, (int(u), int(v)), 3, (40, 220, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas_bgr, str(i), (int(u) + 4, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas_bgr)

def project_keypoints_to_image(
    wrapper: LivePortraitWrapper,
    kp_info: Dict[str, torch.Tensor],
    img_rgb: np.ndarray,
) -> np.ndarray:
    if img_rgb is None or kp_info is None:
        raise ValueError("img_rgb and kp_info must be provided for keypoint visualization.")
    src_h, src_w = img_rgb.shape[:2]
    transformed = wrapper.transform_keypoint(kp_info)[0][:, :2]
    model_h, model_w = wrapper.inference_cfg.input_shape
    kp_model_px = kp_to_pixels(transformed, model_h, model_w)

    scale_x = src_w / float(model_w)
    scale_y = src_h / float(model_h)
    kp_pixels = kp_model_px.copy()
    kp_pixels[:, 0] = np.clip(kp_pixels[:, 0] * scale_x, 0, src_w - 1)
    kp_pixels[:, 1] = np.clip(kp_pixels[:, 1] * scale_y, 0, src_h - 1)
    return kp_pixels

def save_keypoint_map(
    wrapper: LivePortraitWrapper,
    img_rgb: np.ndarray,
    kp_info: Dict[str, torch.Tensor],
    output_path: str,
    kp_pixels: np.ndarray | torch.Tensor | None = None,
) -> None:
    if kp_pixels is None:
        kp_pixels = project_keypoints_to_image(wrapper, kp_info, img_rgb)
    draw_keypoint_map(img_rgb, kp_pixels, output_path)

def draw_kp_map_simple(img_rgb: np.ndarray, kp_xy: torch.Tensor, path: str, mouth_idxs=None):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    H, W = canvas.shape[:2]
    xy = kp_xy.detach().cpu().numpy()
    x = xy[:, 0]
    y = xy[:, 1]
    x_ = (x - x.min()) / max(1e-6, (x.max() - x.min()))
    y_ = (y - y.min()) / max(1e-6, (y.max() - y.min()))
    px = (x_ * (W - 1)).astype(np.int32)
    py = (y_ * (H - 1)).astype(np.int32)
    for i, (u, v) in enumerate(zip(px, py)):
        c = (40, 220, 255) if (mouth_idxs is not None and i in mouth_idxs) else (200, 200, 200)
        cv2.circle(canvas, (int(u), int(v)), 3, c, -1, lineType=cv2.LINE_AA)
        if mouth_idxs is not None and i in mouth_idxs:
            cv2.putText(
                canvas,
                str(i),
                (int(u) + 3, int(v) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def draw_mouth_vectors(
    img_rgb: np.ndarray,
    kp_pixels: np.ndarray | torch.Tensor,
    delta: torch.Tensor,
    mouth_mask: torch.Tensor,
    path: str,
    scale_px: float = 320.0,
    min_arrow_px: float = 6.0,
) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if isinstance(kp_pixels, torch.Tensor):
        pts = kp_pixels.detach().cpu().numpy()
    else:
        pts = np.asarray(kp_pixels, dtype=np.float32)
    d = delta.detach().cpu().numpy()
    mouth_idx = np.nonzero(mouth_mask.detach().cpu().numpy())[0]
    for i in mouth_idx:
        u = float(pts[i, 0])
        v = float(pts[i, 1])
        du = float(d[i, 0]) * float(scale_px)
        dv = float(d[i, 1]) * float(scale_px)
        mag = math.hypot(du, dv)
        if mag < 1e-6:
            continue
        if mag < float(min_arrow_px):
            scale = float(min_arrow_px) / (mag + 1e-8)
            du *= scale
            dv *= scale
        start = (int(round(u)), int(round(v)))
        end = (int(round(u + du)), int(round(v + dv)))
        # Use an orange arrow to visualize motion
        cv2.arrowedLine(canvas_bgr, start, end, (0, 140, 255), 2, tipLength=0.25)

    cv2.imwrite(path, canvas_bgr)


def _two_nearest_x_midpoint(pts: np.ndarray, center_x: float) -> Optional[np.ndarray]:
    if pts.shape[0] == 0:
        return None
    order = np.argsort(np.abs(pts[:, 0] - center_x))
    chosen = pts[order[:2]]
    if chosen.shape[0] == 1:
        return chosen[0]
    return chosen.mean(axis=0)


def compute_lip_metrics(
    kp_pixels: np.ndarray,
    lip_idx,
    img_shape: tuple[int, int],
    verbose: bool = False,
    corner_idx=None,
) -> Optional[Dict[str, np.ndarray]]:
    if kp_pixels is None or lip_idx is None:
        return None
    try:
        pts_all = kp_pixels[np.array(lip_idx, dtype=int)]
    except Exception:
        return None
    if pts_all.shape[0] < 2:
        if verbose:
            print(f"[lip-metrics/debug] insufficient lip pts: {pts_all.shape[0]}")
        return None

    mouth_center = pts_all.mean(axis=0)

    upper_pts = pts_all[pts_all[:, 1] < mouth_center[1]]
    lower_pts = pts_all[pts_all[:, 1] >= mouth_center[1]]
    if upper_pts.shape[0] == 0 or lower_pts.shape[0] == 0:
        sorted_y = pts_all[np.argsort(pts_all[:, 1])]
        upper_pts = sorted_y[:2] if sorted_y.shape[0] >= 2 else sorted_y
        lower_pts = sorted_y[-2:] if sorted_y.shape[0] >= 2 else sorted_y

    upper_mid = _two_nearest_x_midpoint(upper_pts, mouth_center[0])
    lower_mid = _two_nearest_x_midpoint(lower_pts, mouth_center[0])
    if upper_mid is None or lower_mid is None:
        return None

    left_corner = None
    right_corner = None
    if corner_idx is not None and len(corner_idx) == 2:
        try:
            c_pts = kp_pixels[np.array(corner_idx, dtype=int)]
            if c_pts.shape[0] == 2:
                if c_pts[0, 0] <= c_pts[1, 0]:
                    left_corner, right_corner = c_pts[0], c_pts[1]
                else:
                    left_corner, right_corner = c_pts[1], c_pts[0]
        except Exception:
            left_corner, right_corner = None, None
    if left_corner is None or right_corner is None:
        left_corner = pts_all[np.argmin(pts_all[:, 0])]
        right_corner = pts_all[np.argmax(pts_all[:, 0])]

    lip_gap = float(abs(upper_mid[1] - lower_mid[1]))
    lip_width = float(np.linalg.norm(right_corner - left_corner))

    return {
        "lip_gap": lip_gap,
        "lip_width": lip_width,
        "upper_mid": upper_mid,
        "lower_mid": lower_mid,
        "left_corner": left_corner,
        "right_corner": right_corner,
    }


def draw_lip_metrics_overlay(img_rgb: np.ndarray, lip_info: Dict[str, np.ndarray], out_path: str) -> None:
    if lip_info is None or img_rgb is None or img_rgb.size == 0:
        return
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    canvas = img_rgb.copy()
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    def _pt(p):
        return (int(round(p[0])), int(round(p[1])))

    upper_mid = _pt(lip_info["upper_mid"])
    lower_mid = _pt(lip_info["lower_mid"])
    left_corner = _pt(lip_info["left_corner"])
    right_corner = _pt(lip_info["right_corner"])

    cv2.circle(canvas_bgr, upper_mid, 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)  # yellow
    cv2.circle(canvas_bgr, lower_mid, 4, (0, 255, 200), -1, lineType=cv2.LINE_AA)  # cyan-ish
    cv2.circle(canvas_bgr, left_corner, 4, (255, 180, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas_bgr, right_corner, 4, (255, 100, 0), -1, lineType=cv2.LINE_AA)
    cv2.line(canvas_bgr, upper_mid, lower_mid, (0, 220, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(canvas_bgr, left_corner, right_corner, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas_bgr,
        f"gap: {lip_info['lip_gap']:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_bgr,
        f"width: {lip_info['lip_width']:.2f}",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 128, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(out_path, canvas_bgr)
def _pct(a: torch.Tensor, q: float):
    return torch.quantile(a, torch.tensor(q, device=a.device))

def _summ_stats(vec: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(vec.mean().item()),
        "median": float(vec.median().item()),
        "max": float(vec.abs().max().item()),
        "p05": float(_pct(vec, 0.05).item()),
        "p95": float(_pct(vec, 0.95).item()),
    }

def log_stage(stage_name: str, exp_delta: torch.Tensor, mask_dict: Dict[str, torch.Tensor]):
    for k in ["lips_mask", "corner_mask", "center_lip_mask"]:
        if k not in mask_dict:
            continue
        m = mask_dict[k]
        n = int(m.sum().item())
        idx_show = torch.nonzero(m, as_tuple=False).squeeze(-1).tolist()
        idx_show = idx_show[:8] if isinstance(idx_show, list) else []
        if n > 0:
            dx = exp_delta[0, m, 0]
            dy = exp_delta[0, m, 1]
            sx = _summ_stats(dx)
            sy = _summ_stats(dy)
def draw_masks_overlay(img_rgb: np.ndarray, kpts_xy: torch.Tensor, masks: Dict[str, torch.Tensor], path: str):
    color_map = {
        "eyes_mask": (255, 200, 0),
        "brows_mask": (255, 100, 0),
        "lips_mask": (50, 220, 255),
        "corner_mask": (0, 180, 255),
        "center_lip_mask": (255, 50, 200),
    }
    canvas = img_rgb.copy()
    H, W = canvas.shape[:2]
    xy = kpts_xy.detach().cpu().numpy()
    x = xy[:,0]; y = xy[:,1]
    x_ = (x - x.min()) / max(1e-6, (x.max() - x.min()))
    y_ = (y - y.min()) / max(1e-6, (y.max() - y.min()))
    px = (x_ * (W-1)).astype(np.int32)
    py = (y_ * (H-1)).astype(np.int32)
    for name, m in masks.items():
        c = color_map.get(name, (200, 200, 200))
        idxs = torch.nonzero(m, as_tuple=False).squeeze(-1).cpu().numpy()
        for i in idxs:
            cv2.circle(canvas, (px[i], py[i]), 3, c, -1, lineType=cv2.LINE_AA)
    os.makedirs(osp.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def compute_masks(kp_can_t: torch.Tensor):
    xy = kp_can_t[:, :2]
    cx, cy = xy.mean(dim=0)
    dxn = xy[:, 0] - cx
    dyn = xy[:, 1] - cy
    sx = torch.std(dxn) + 1e-6
    sy = torch.std(dyn) + 1e-6
    xn = dxn / sx
    yn = dyn / sy
    eyes_mask = (yn < -0.15) & (xn.abs() > 0.35)
    brows_mask = (yn < -0.30) & (xn.abs() > 0.25)
    lips_mask = torch.zeros_like(xn, dtype=torch.bool)
    corner_mask = torch.zeros_like(xn, dtype=torch.bool)
    center_lip_mask = torch.zeros_like(xn, dtype=torch.bool)
    for idx in LIP_IDX:
        if 0 <= idx < lips_mask.shape[0]:
            lips_mask[idx] = True
            center_lip_mask[idx] = True
    for idx in LIP_CORNER_IDX:
        if 0 <= idx < corner_mask.shape[0]:
            corner_mask[idx] = True
    return {
        "eyes_mask": eyes_mask,
        "brows_mask": brows_mask,
        "lips_mask": lips_mask,
        "corner_mask": corner_mask,
        "center_lip_mask": center_lip_mask,
        "xn": xn,
        "yn": yn
    }

def get_exp_tensor(wrap: LivePortraitWrapper, img_rgb: np.ndarray):
    # run motion extractor and pull the expression tensor (B,N,3)
    T = wrap.prepare_source(img_rgb)
    kp = wrap.get_kp_info(T, flag_refine_info=True)
    return kp["exp"]

def choose_best_flip(wrap, donor_rgb, target_kp_xy):
    # returns donor_img_best, flip_used (bool)
    donor_flip = donor_rgb[:, ::-1, :].copy()

    D1 = wrap.prepare_source(donor_rgb)
    kp1 = wrap.get_kp_info(D1, flag_refine_info=True)["kp"][0][:, :2]

    D2 = wrap.prepare_source(donor_flip)
    kp2 = wrap.get_kp_info(D2, flag_refine_info=True)["kp"][0][:, :2]
    kp2 = kp2.clone()
    kp2[:, 0] *= -1

    # compare after Procrustes scale + rotation align
    def _align_dist(a, b):
        s = torch.tensor(procrustes_scale(a, b))
        a2 = a * s
        R, _ = similarity_decompose(a2, b)
        a2 = a2 @ R
        return torch.linalg.norm(a2 - b)

    d1 = _align_dist(kp1, target_kp_xy)
    d2 = _align_dist(kp2, target_kp_xy)
    return (donor_rgb, False) if d1 <= d2 else (donor_flip, True)

def main(
    donor_path,
    target_path,
    out_path="outputs/asym_transfer.jpg",
    cfg_path="src/config/models.yaml",
    scale=1.0,
    edge_dampen=0.55,
    out_w=0,
    out_h=0,
    asym_side="right",
    midline_eps=0.05,
    clamp_q=0.90,
    boost_base=0.50,
    eye_gain=0.70,
    brow_gain=0.85,
    lip_gain=1.40,
    auto_flip=False,
    boost_base_floor=0.5,
    boost_focus_gain=0.5,
    lip_gain_x=None,
    lip_gain_y=None,
    corner_gain=1.6,
    norm_cap=2.5,
    mouth_sym_alpha=0.0,
    y_drift_fix="nonmouth",
    y_anchor=0.5,
    # control how much of the nonmouth bias we subtract from mouth Y (0=keep mouth free)
    y_drift_mouth_bias: float = 0.0,
    lip_metrics: bool = False,
    # FaceMesh parameters
    facemesh_driving: bool = False,
    facemesh_debug: bool = False,
    facemesh_regions: Optional[str] = None,
    facemesh_max_delta_px: Optional[float] = None,
    facemesh_clamp_percentile: float = 98.0,
    facemesh_save_npy: bool = False,
    cheek_radius_px: Optional[float] = None,
    facemesh_refine: bool = False,
    # FaceMesh Expression Assist parameters
    facemesh_exp_assist: bool = False,
    facemesh_exp_beta: float = 1.0,
    facemesh_exp_mouth_alpha: float = 1.0,
    facemesh_exp_method: str = "knn",
    facemesh_exp_knn_k: int = 8,
    facemesh_exp_inject_stage: str = "post_drift",
    facemesh_exp_debug: bool = False,
    facemesh_exp_max_disp_px: Optional[float] = None,
    facemesh_exp_cap_percentile: float = 98.0,
    facemesh_exp_smooth: bool = True,
    facemesh_exp_smooth_k: int = 6,
    facemesh_exp_zero_stable: bool = True,
    # FaceMesh warp parameters (Phase 3-5)
    facemesh_warp: bool = False,
    facemesh_warp_method: str = "tps",
    facemesh_warp_alpha: float = 1.0,
    facemesh_warp_reg: float = 1e-3,
    facemesh_warp_grid_step: int = 2,
    facemesh_warp_lock_boundary: bool = True,
    facemesh_warp_validate: bool = True,
    facemesh_warp_save_field: bool = False,
    # Phase 6 guardrails
    facemesh_guards: Optional[bool] = None,
    facemesh_guard_debug: bool = False,
    guard_max_delta_px: Optional[float] = None,
    guard_cap_percentile: float = 98.0,
    guard_cap_region: str = "weighted_only",
    guard_cap_after_align: bool = True,
    guard_smooth_delta: bool = True,
    guard_smooth_iterations: int = 2,
    guard_smooth_lambda: float = 0.6,
    guard_smooth_mode: str = "knn",
    guard_knn_k: int = 8,
    guard_zero_anchor: bool = True,
    guard_anchor_idx: Optional[List[int]] = None,
    guard_anchor_strength: float = 0.95,
    guard_softmask: bool = True,
    guard_softmask_sigma: float = 25.0,
    guard_softmask_forehead_fade: bool = True,
    guard_softmask_forehead_yfrac: float = 0.22,
    guard_softmask_min: float = 0.0,
    guard_softmask_max: float = 1.0,
    guard_face_mask: bool = True,
    guard_face_mask_mode: str = "hull",
    guard_face_mask_dilate: int = 12,
    guard_face_mask_erode: int = 0,
    guard_face_mask_blur: int = 11,
    guard_warp_face_only: bool = True,
    guard_mouth_only: bool = False,
    guard_mouth_radius_px: int = 90,
    guard_alpha_start: float = 0.3,
):
    # load models
    cfg = InferenceConfig(models_config=cfg_path)
    wrap = LivePortraitWrapper(cfg)

    # read donor (has asymmetry) and target (neutral avatar)
    donor_rgb = read_rgb(donor_path)
    target_rgb = read_rgb(target_path)

    # =========================================================================
    # FaceMesh-based donor asymmetry analysis (if enabled)
    # =========================================================================
    facemesh_result = None
    if facemesh_driving:
        print("\n[FaceMesh] Computing donor asymmetry driving signal...")

        # Initialize extractor
        extractor = FaceMeshLandmarkExtractor(
            static_image_mode=True,
            max_num_faces=1,
            verbose=True,
            refine_landmarks=facemesh_refine,
            debug=facemesh_debug,
        )

        # Load region configuration
        regions_config = load_facemesh_regions_config(facemesh_regions)

        # Compute asymmetry
        facemesh_result = compute_donor_asymmetry_delta(
            donor_rgb,
            extractor,
            regions_config,
            apply_bias_removal=True,
            max_delta_px=facemesh_max_delta_px,
            clamp_percentile=facemesh_clamp_percentile,
            cheek_radius_px=cheek_radius_px,
            verbose=True
        )

        if facemesh_result.get("ok", False):
            print("[FaceMesh] ✓ Asymmetry computation successful")

            # Debug outputs
            if facemesh_debug:
                print("[FaceMesh] Saving debug visualizations...")
                os.makedirs("outputs/diagnostics/facemesh", exist_ok=True)

                # All landmarks overlay on donor
                groups = {
                    "lips": regions_config["lips"],
                    "face_oval": regions_config["face_oval"],
                    "cheek_patch": facemesh_result.get("cheek_patch", []),
                    "anchors": regions_config["anchors"],
                }
                draw_facemesh_overlay(
                    donor_rgb,
                    facemesh_result["L_d"],
                    groups,
                    "outputs/diagnostics/facemesh/donor_facemesh_all.png",
                    label_some=True
                )

                # Flipped donor landmarks
                draw_facemesh_overlay(
                    donor_rgb[:, ::-1, :].copy(),
                    facemesh_result["L_f"],
                    groups,
                    "outputs/diagnostics/facemesh/donor_flip_facemesh_all.png",
                    label_some=False
                )

                # Comparison: flip-back vs original on donor image
                draw_comparison_overlay(
                    donor_rgb,
                    facemesh_result["L_d"],
                    facemesh_result["L_f_back"],
                    "outputs/diagnostics/facemesh/donor_flip_back_overlay.png",
                    label_1="donor_original",
                    label_2="flip_mirrored_back"
                )

                # Indexed alignment visualization (if permutation available)
                if facemesh_result.get("flipback_perm") is not None:
                    draw_flipback_indexed_lines(
                        donor_rgb,
                        facemesh_result["L_d"],
                        facemesh_result["L_f_back"],
                        num_indices=50,
                        out_path="outputs/diagnostics/facemesh/donor_flip_back_indexed_lines.png"
                    )

                # Regions visualization
                draw_facemesh_overlay(
                    donor_rgb,
                    facemesh_result["L_d"],
                    groups,
                    "outputs/diagnostics/facemesh/regions_overlay.png",
                    label_some=True
                )

                # Delta heatmap
                roi_union = set(
                    regions_config["lips"] +
                    regions_config["face_oval"] +
                    facemesh_result.get("cheek_patch", [])
                )
                roi_union = sorted(list(roi_union))
                draw_delta_heatmap(
                    donor_rgb,
                    facemesh_result["L_d"],
                    facemesh_result["delta"],
                    roi_union,
                    "outputs/diagnostics/facemesh/delta_heatmap.png",
                    scale_px=320.0
                )

                print("[FaceMesh] ✓ Debug visualizations saved")

            # Save numpy dumps
            if facemesh_save_npy:
                save_facemesh_numpy_dumps(facemesh_result)
                save_facemesh_summary(facemesh_result)
        else:
            print("[FaceMesh] ✗ Asymmetry computation failed, continuing without FaceMesh")
            facemesh_driving = False

    # get target keypoints in canonical coords to define left/right etc
    T = wrap.prepare_source(target_rgb)
    T_kp = wrap.get_kp_info(T, flag_refine_info=True)
    kp_can_t = T_kp["kp"][0]
    kp_can_t_xy = kp_can_t[:, :2]
    x_can = kp_can_t[:, 0]
    x_center = x_can.mean()
    left_mask = x_can < x_center
    right_mask = ~left_mask

    os.makedirs("outputs/diagnostics", exist_ok=True)
    target_kp_pixels = project_keypoints_to_image(wrap, T_kp, target_rgb)
    save_keypoint_map(
        wrap,
        target_rgb,
        T_kp,
        "outputs/diagnostics/target_keypoint_map.jpg",
        kp_pixels=target_kp_pixels,
    )
    draw_keypoint_map_labeled(
        target_rgb,
        target_kp_pixels,
        "outputs/diagnostics/target_keypoint_map_labeled.jpg",
    )

    flip_used = False
    if auto_flip:
        donor_rgb, flip_used = choose_best_flip(wrap, donor_rgb, kp_can_t_xy)

    # donor LR delta via flip and map back
    E_donor = get_exp_tensor(wrap, donor_rgb)
    donor_flip = donor_rgb[:, ::-1, :].copy()
    E_flip = get_exp_tensor(wrap, donor_flip)
    E_flip_map = E_flip.clone(); E_flip_map[..., 0] *= -1

    # Visualize donor keypoints
    D = wrap.prepare_source(donor_rgb)
    D_kp = wrap.get_kp_info(D, flag_refine_info=True)
    kp_can_d = D_kp["kp"][0]
    save_keypoint_map(
        wrap,
        donor_rgb,
        D_kp,
        "outputs/diagnostics/donor_keypoint_map.jpg",
    )

    # Canonical LR delta (right-minus-left) Use CLI to decide sign
    raw_delta = (E_donor - E_flip_map)
    side = asym_side.lower()
    exp_delta = raw_delta * float(scale) if side == "right" else -raw_delta * float(scale)

    pre_norm = exp_delta.detach().float().norm().item()
    S0 = exp_delta.clone()

    # normalize donor vs target pose/size in canonical space
    kp_can_d_xy = kp_can_d[:, :2]
    # scale (zoom) match
    s_ratio = procrustes_scale(kp_can_d_xy, kp_can_t_xy)
    exp_delta[0, :, 0:2] *= s_ratio
    # in-plane rotation (roll) match
    R, _theta = similarity_decompose(kp_can_d_xy, kp_can_t_xy)
    R = R.to(dtype=exp_delta.dtype, device=exp_delta.device)
    exp_delta[0, :, 0:2] = exp_delta[0, :, 0:2] @ R
    S1 = exp_delta.clone()
    with torch.no_grad():
        xy = kp_can_t[:, :2]
        cx, cy = xy.mean(dim=0)
        dxn = (xy[:,0] - cx); dyn = (xy[:,1] - cy)
        sx = torch.std(dxn) + 1e-6
        sy = torch.std(dyn) + 1e-6

        sig_x = 0.95
        sig_y = 0.90
        w_roi = torch.exp(-0.5*((dxn/(sig_x*sx))**2 + (dyn/(sig_y*sy))**2))
        w_roi = 0.25 + 0.75*w_roi
        exp_delta[0,:,0] *= w_roi
        span_x = (kp_can_t[:,0] - kp_can_t[:,0].mean()).abs().max() + 1e-6
        edge_thresh = 0.88
        edge_mask = ((kp_can_t[:,0] - kp_can_t[:,0].mean()).abs() / span_x) > edge_thresh
        if int(edge_mask.sum()) > 0:
            exp_delta[0, edge_mask, 0] = 0.0
    S2 = exp_delta.clone()

    # =========================================================================
    # FACEMESH EXPRESSION ASSIST (pre_gain injection point)
    # =========================================================================
    if facemesh_exp_assist and facemesh_exp_inject_stage == "pre_gain":
        try:
            # Compute mouth stats BEFORE injection
            exp_delta_before_inject = exp_delta.clone()
            mouth_kp_idx = [idx for idx in LIP_IDX + LIP_CORNER_IDX if 0 <= idx < exp_delta.shape[1]]
            if len(mouth_kp_idx) > 0:
                mouth_delta_before = exp_delta_before_inject[0, mouth_kp_idx, :2]
                mouth_mag_before = torch.linalg.norm(mouth_delta_before, dim=1)
                mag_mean_before = float(mouth_mag_before.mean())
                mag_max_before = float(mouth_mag_before.max())
            else:
                mag_mean_before = 0.0
                mag_max_before = 0.0

            # Get LP keypoints in pixel space for projection
            target_kp_pixels_for_fm = kp_to_pixels(kp_can_t[:, :2], target_rgb.shape[0], target_rgb.shape[1])

            # Get or create extractor
            if facemesh_driving and 'extractor' in locals():
                fm_extractor = extractor
            else:
                fm_extractor = FaceMeshLandmarkExtractor(
                    static_image_mode=True,
                    max_num_faces=1,
                    verbose=False,
                    refine_landmarks=facemesh_refine,
                    debug=False,
                )

            # Apply FaceMesh expression assist
            exp_delta_fm, fm_debug_dict = apply_facemesh_exp_assist(
                target_rgb=target_rgb,
                target_kp_px=target_kp_pixels_for_fm,
                extractor=fm_extractor,
                beta=beta,
                method=facemesh_exp_method,
                knn_k=facemesh_exp_knn_k,
                smooth=facemesh_exp_smooth,
                max_disp_px=facemesh_exp_max_disp_px,
                debug=facemesh_exp_debug,
                verbose=verbose,
            )

            if exp_delta_fm is not None:
                # Add FaceMesh correction to exp_delta BEFORE gains/compression
                exp_delta = exp_delta + exp_delta_fm

                # Compute mouth stats AFTER injection
                mouth_delta_after = exp_delta[0, mouth_kp_idx, :2]
                mouth_mag_after = torch.linalg.norm(mouth_delta_after, dim=1)
                mag_mean_after = float(mouth_mag_after.mean())
                mag_max_after = float(mouth_mag_after.max())

                # Log
                print(f"[FaceMesh-EXP] inject_stage=pre_gain: adding mouth residual BEFORE gains/compression")
                print(f"[FaceMesh-EXP]   mouth magnitude: before_inject mean={mag_mean_before:.3f}px max={mag_max_before:.3f}px")
                print(f"[FaceMesh-EXP]   mouth magnitude: after_inject  mean={mag_mean_after:.3f}px max={mag_max_after:.3f}px")

                # Save debug artifacts if requested
                if facemesh_exp_debug:
                    try:
                        exp_delta_before_np = exp_delta_before_inject.detach().cpu().numpy()
                        exp_delta_after_np = exp_delta.detach().cpu().numpy()
                        os.makedirs("outputs/diagnostics/facemesh_exp_assist", exist_ok=True)
                        np.save("outputs/diagnostics/facemesh_exp_assist/exp_delta_before_inject.npy", exp_delta_before_np)
                        np.save("outputs/diagnostics/facemesh_exp_assist/exp_delta_after_inject.npy", exp_delta_after_np)
                    except Exception as e:
                        print(f"[FaceMesh-EXP] Warning: Failed to save debug artifacts: {e}")
        except Exception as e:
            print(f"[FaceMesh-EXP] Error in pre_gain injection: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    # region specific control
    mouth_mask = None
    with torch.no_grad():
        xy = kp_can_t[:, :2].to(exp_delta.device, exp_delta.dtype)
        cx, cy = xy.mean(dim=0)
        dxn = xy[:, 0] - cx
        dyn = xy[:, 1] - cy
        sx = torch.std(dxn) + 1e-6
        sy = torch.std(dyn) + 1e-6

        # Normalized coords (zero-mean, unit-ish std)
        xn = dxn / sx
        yn = dyn / sy
        lips_mask = torch.zeros_like(xn, dtype=torch.bool)
        corner_mask = torch.zeros_like(xn, dtype=torch.bool)
        center_lip_mask = torch.zeros_like(xn, dtype=torch.bool)
        for idx in LIP_IDX:
            if 0 <= idx < lips_mask.shape[0]:
                lips_mask[idx] = True
                center_lip_mask[idx] = True
        for idx in LIP_CORNER_IDX:
            if 0 <= idx < corner_mask.shape[0]:
                corner_mask[idx] = True

        # Eyes / brows (leave as-is; cap to not exceed 1x)
        eyes_mask = (yn < torch.quantile(yn, torch.tensor(0.25, device=yn.device))) & (xn.abs() > torch.quantile(xn.abs(), torch.tensor(0.55, device=xn.device)))
        brows_mask = (yn < torch.quantile(yn, torch.tensor(0.15, device=yn.device))) & (xn.abs() > torch.quantile(xn.abs(), torch.tensor(0.45, device=xn.device)))
        eye_brow_mask = eyes_mask | brows_mask

        w_focus   = torch.exp(-0.5 * (xn**2 + yn**2))
        k_vec = (float(boost_base_floor) + float(boost_focus_gain) * w_focus)

        # Region gains
        g_eye = float(eye_gain)
        g_brow = float(brow_gain)
        g_lip = float(lip_gain)
        g_lip_x = float(lip_gain_x) if lip_gain_x is not None else g_lip
        g_lip_y = float(lip_gain_y) if lip_gain_y is not None else g_lip
        g_corner = float(corner_gain)

        gains_xy = torch.ones((xy.shape[0], 2), dtype=exp_delta.dtype, device=exp_delta.device)
        eb = min(g_eye * g_brow, 1.0)
        gains_xy[eye_brow_mask,0] *= eb
        gains_xy[eye_brow_mask,1] *= eb
        gains_xy[lips_mask,0] *= g_lip_x
        gains_xy[lips_mask,1] *= g_lip_y
        gains_xy[corner_mask,0] *= g_corner
        gains_xy[corner_mask,1] *= g_corner

        no_k  = lips_mask | corner_mask
        k_vec = torch.where(no_k, torch.ones_like(k_vec), k_vec)

        # Optional symmetric Y add-in at the center of the mouth
        if float(mouth_sym_alpha) > 0.0:
            sym_term = 0.5 * (E_donor + E_flip_map) - T_kp["exp"]
            sym_term = sym_term.to(exp_delta.device, exp_delta.dtype)
            sym_y = sym_term[0, :, 1]
            center_weight = torch.exp(-0.5 * ((xn/0.30)**2 + (yn/0.55)**2))
            mouth_mask_loc = lips_mask | corner_mask
            add_y = float(mouth_sym_alpha) * sym_y * center_weight
            add_y = torch.where(center_lip_mask, 1.6 * sym_y * center_weight, add_y)
            exp_delta[0, mouth_mask_loc, 1] = exp_delta[0, mouth_mask_loc, 1] + add_y[mouth_mask_loc]

        # Apply base & XY gains
        exp_delta[0, :, 0:2] = exp_delta[0, :, 0:2] * k_vec.unsqueeze(-1) * gains_xy

        # Mouth-friendly soft-knee compression
        V = exp_delta[0]
        mags = torch.linalg.norm(V, dim=-1)
        tau_v_all = float(torch.quantile(mags, torch.tensor(0.98, device=mags.device))) * 2.0
        tau_v = torch.full_like(mags, tau_v_all)
        tau_v[lips_mask | corner_mask] = tau_v_all * 3.0
        m = torch.linalg.norm(V, dim=-1, keepdim=True) + 1e-8
        m_prime = tau_v.unsqueeze(-1) * torch.tanh(m / tau_v.unsqueeze(-1))
        gain = (m_prime / m).clamp(max=1.0)
        V = V * gain

        dx = V[:, 0]
        tau_x_all = float(torch.quantile(dx.abs(), torch.tensor(0.95, device=dx.device))) * 1.20
        tau_x = torch.full_like(dx, tau_x_all)
        tau_x[lips_mask | corner_mask] = tau_x_all * 1.6
        a = dx.abs() + 1e-8
        a_prime = tau_x * torch.tanh(a / tau_x)
        V[:, 0] = torch.sign(dx) * a_prime

        # Diagnostics
        masks_dict = {
            "lips_mask":       lips_mask,
            "corner_mask":     corner_mask,
            "center_lip_mask": center_lip_mask,
            "eyes_mask":       eyes_mask,
            "brows_mask":      brows_mask,
        }
        log_stage("S0 (raw pre-align)", S0, masks_dict)
        log_stage("S1 (post scale+rot)", S1, masks_dict)
        log_stage("S2 (after ROI & edge zero)", S2, masks_dict)
        log_stage("S3 (after gains + soft-knee)", exp_delta, masks_dict)
        try:
            draw_masks_overlay(target_rgb, kp_can_t[:, :2], masks_dict, "outputs/diagnostics/kp_masks.jpg")
        except Exception as e:
            pass

        mouth_mask = lips_mask | corner_mask
        mouth_idx_list = torch.nonzero(mouth_mask, as_tuple=False).squeeze(-1).cpu().tolist()
        if len(mouth_idx_list) == 0:
            # absolute last-resort fallback: lower half + center-ish width
            mouth_mask = (yn > torch.quantile(yn, torch.tensor(0.50, device=yn.device))) & (xn.abs() < torch.quantile(xn.abs(), torch.tensor(0.85, device=xn.device)))
            mouth_idx_list = torch.nonzero(mouth_mask, as_tuple=False).squeeze(-1).cpu().tolist()

        draw_kp_map_simple(target_rgb, kp_can_t[:, :2], "outputs/diagnostics/target_kp_mouth_labeled.jpg", mouth_idxs=set(mouth_idx_list))

    def _save_mouth_vectors(tag: str, delta_tensor: torch.Tensor, out_path: str):
        if mouth_mask is None:
            return
        try:
            mouth_delta = delta_tensor[0, mouth_mask, :]
            if mouth_delta.numel() == 0:
                print(f"[viz] mouth_vectors {tag} skipped: mouth_mask empty")
                return
            mags = mouth_delta.norm(dim=1)
            print(f"[viz] mouth_vectors {tag} -> mean|d|={mags.mean().item():.6f}, max|d|={mags.max().item():.6f}")
            draw_mouth_vectors(
                target_rgb,
                target_kp_pixels,
                delta_tensor[0, :, :].clamp(-1.0, 1.0),
                mouth_mask,
                out_path,
                scale_px=360.0,
            )
            print(f"[viz] saved {out_path}")
        except Exception as e:
            print(f"[viz] mouth_vectors {tag} skip:", e)

    def _log_vertical_metrics(tag: str, delta_tensor: torch.Tensor, stable_mask: torch.Tensor, mouth_like_mask: torch.Tensor):
        dy = delta_tensor[0, :, 1]
        def _metrics(mask, name):
            if mask is None or (not mask.any()):
                print(f"[guard] {tag} {name}: no samples")
                return
            sel = dy[mask]
            mean_abs = sel.abs().mean().item()
            max_abs = sel.abs().max().item()
            mean_signed = sel.mean().item()
            print(f"[guard] {tag} {name}: mean|dy|={mean_abs:.6f} max|dy|={max_abs:.6f} mean(dy)={mean_signed:.6f}")
        _metrics(stable_mask, "stable")
        _metrics(mouth_like_mask, "mouth")

    xy_t = kp_can_t[:, :2].to(exp_delta.device, exp_delta.dtype)
    cx_t, cy_t = xy_t.mean(dim=0)
    dx_t = xy_t[:, 0] - cx_t
    dy_t = xy_t[:, 1] - cy_t
    sx_t = torch.std(dx_t) + 1e-6
    sy_t = torch.std(dy_t) + 1e-6
    yn = dy_t / sy_t
    xn_t = dx_t / sx_t

    lips_mask2 = torch.zeros_like(yn, dtype=torch.bool)
    corner_mask2 = torch.zeros_like(yn, dtype=torch.bool)
    for idx in LIP_IDX:
        if 0 <= idx < lips_mask2.shape[0]:
            lips_mask2[idx] = True
    for idx in LIP_CORNER_IDX:
        if 0 <= idx < corner_mask2.shape[0]:
            corner_mask2[idx] = True
    stable = ~(lips_mask2 | corner_mask2)

    # =========================================================================
    # FACEMESH EXPRESSION ASSIST (pre_drift injection point)
    # =========================================================================
    if facemesh_exp_assist and facemesh_exp_inject_stage == "pre_drift":
        print("\n[FaceMesh Exp Assist] Applying pre-drift correction...")
        try:
            # Get LP keypoints in pixel space for projection
            target_kp_pixels_for_fm = kp_to_pixels(kp_can_t[:, :2], target_rgb.shape[0], target_rgb.shape[1])

            # Get or create extractor (reuse if facemesh_driving created one)
            if facemesh_driving and 'extractor' in locals():
                fm_extractor = extractor
            else:
                fm_extractor = FaceMeshLandmarkExtractor(
                    static_image_mode=True,
                    max_num_faces=1,
                    verbose=False,
                    refine_landmarks=facemesh_refine,
                    debug=False,
                )

            exp_delta_fm, fm_debug_dict = apply_facemesh_exp_assist(
                donor_rgb=donor_rgb,
                target_rgb=target_rgb,
                exp_delta=exp_delta,
                extractor=fm_extractor,
                lp_keypoints_px=target_kp_pixels_for_fm,
                beta=facemesh_exp_beta,
                mouth_alpha=facemesh_exp_mouth_alpha,
                method=facemesh_exp_method,
                knn_k=facemesh_exp_knn_k,
                tps_reg=1e-3,
                inject_stage=facemesh_exp_inject_stage,
                debug=facemesh_exp_debug,
                lips_mask_indices=LIP_IDX,
                corner_mask_indices=LIP_CORNER_IDX,
                verbose=True,
            )

            if exp_delta_fm is not None and fm_debug_dict.get("ok", False):
                exp_delta = exp_delta + exp_delta_fm
                print(f"[FaceMesh Exp Assist] [OK] Applied correction (shape: {tuple(exp_delta_fm.shape)})")
            else:
                error_msg = fm_debug_dict.get("error", "unknown")
                print(f"[FaceMesh Exp Assist] [SKIP] Skipped (stub or error: {error_msg})")

        except Exception as e:
            print(f"[FaceMesh Exp Assist] [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()

    dy = exp_delta[0, :, 1]
    if y_drift_fix != "none":
        if y_drift_fix == "nonmouth" and stable.any():
            dy2, _ = affine_detrend_dy(dy, kp_can_t_xy)
            # Compute the residual bias on stable landmarks only
            bias_stable = dy2[stable].mean()
            # Apply bias to non-mouth only
            exp_delta[0, stable, 1] = dy2[stable] - bias_stable
            mouth_mask_all = ~(stable)
            if float(y_drift_mouth_bias) != 0.0 and mouth_mask_all.any():
                exp_delta[0, mouth_mask_all, 1] = dy2[mouth_mask_all] - float(y_drift_mouth_bias) * bias_stable
            else:
                exp_delta[0, mouth_mask_all, 1] = dy2[mouth_mask_all]
        else:
            dy2, _ = affine_detrend_dy(dy, kp_can_t_xy)
            exp_delta[0, :, 1] = dy2

    lips_mask_log = lips_mask2
    corner_mask_log = corner_mask2
    center_lip_mask_log = (xn_t.abs() < 0.18) & (yn > -0.05) & (yn < 0.45)
    log_stage("S4 (after y_drift_fix)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })
    _log_vertical_metrics("post_y_drift_fix", exp_delta, stable, ~stable)
    _save_mouth_vectors("pre_y_anchor", exp_delta, "outputs/diagnostics/mouth_vectors_pre_y_anchor.jpg")

    # Anchor extremes to avoid vertical squash/stretch
    if float(y_anchor) > 0.0:
        yn_abs = yn.abs().clamp(0, 2.0)
        denom = yn_abs.max().clamp_min(1e-6)
        anchor = 1.0 - float(y_anchor) * (yn_abs / denom)
        # skip anchoring lips or corners so the mouth stays lively
        exp_delta[0, stable, 1] = exp_delta[0, stable, 1] * anchor[stable]

    log_stage("S5 (after y_anchor)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })
    _log_vertical_metrics("post_y_anchor", exp_delta, stable, ~stable)
    _save_mouth_vectors("post_y_anchor", exp_delta, "outputs/diagnostics/mouth_vectors_post_y_anchor.jpg")

    dx = exp_delta[0, :, 0]
    dx, _coeff = affine_detrend_dx(dx, kp_can_t_xy)
    exp_delta[0, :, 0] = dx

    log_stage("S6 (after affine_detrend_dx)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })

    # clamp extreme X motions
    if clamp_q > 0.0:
        dx = exp_delta[0, :, 0]
        q = torch.quantile(dx.abs(), torch.tensor(float(clamp_q), device=dx.device))
        exp_delta[0, :, 0] = dx.clamp(min=-q, max=q)

    log_stage("S7 (after clamp_q)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })

    # per-side mean removal so left/right don't drift apart on average
    dx = exp_delta[0, :, 0]
    if (left_mask.any()):
        dx[left_mask] = dx[left_mask] - dx[left_mask].mean()
    if (right_mask.any()):
        dx[right_mask] = dx[right_mask] - dx[right_mask].mean()
    exp_delta[0, :, 0] = dx

    # zero net push safety: total lateral push cancels out
    sign_x = torch.sign(kp_can_t[:,0] - x_center)
    dx = exp_delta[0,:,0]
    exp_delta[0,:,0] = dx - (dx * sign_x).mean() * sign_x

    # final linear falloff toward the edges (feels more natural)
    with torch.no_grad():
        span = (x_can - x_center).abs()
        span_norm = span / (span.max() + 1e-6)
        w = 1.0 - float(edge_dampen) * span_norm
        w = w.clamp(1.0 - float(edge_dampen), 1.0)
        exp_delta[0, :, 0] = exp_delta[0, :, 0] * w

    # S8
    log_stage("S8 (after per-side mean, zero-net-push, edge falloff)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })

    _save_mouth_vectors("final", exp_delta, "outputs/diagnostics/mouth_vectors.jpg")
    _log_vertical_metrics("final", exp_delta, stable, ~stable)

    post_norm = exp_delta.detach().float().norm().item()
    cap = float(pre_norm) * float(norm_cap)
    if (post_norm > cap) and (cap > 1e-8):
        exp_delta *= (cap / post_norm)

    # =========================================================================
    # FACEMESH EXPRESSION ASSIST (post_drift injection point)
    # =========================================================================
    if facemesh_exp_assist and facemesh_exp_inject_stage == "post_drift":
        print("\n[FaceMesh Exp Assist] Applying post-drift correction...")
        try:
            # Get LP keypoints in pixel space for projection
            target_kp_pixels_for_fm = kp_to_pixels(kp_can_t[:, :2], target_rgb.shape[0], target_rgb.shape[1])

            # Get or create extractor (reuse if facemesh_driving created one)
            if facemesh_driving and 'extractor' in locals():
                fm_extractor = extractor
            else:
                fm_extractor = FaceMeshLandmarkExtractor(
                    static_image_mode=True,
                    max_num_faces=1,
                    verbose=False,
                    refine_landmarks=facemesh_refine,
                    debug=False,
                )

            exp_delta_fm, fm_debug_dict = apply_facemesh_exp_assist(
                donor_rgb=donor_rgb,
                target_rgb=target_rgb,
                exp_delta=exp_delta,
                extractor=fm_extractor,
                lp_keypoints_px=target_kp_pixels_for_fm,
                beta=facemesh_exp_beta,
                mouth_alpha=facemesh_exp_mouth_alpha,
                method=facemesh_exp_method,
                knn_k=facemesh_exp_knn_k,
                tps_reg=1e-3,
                inject_stage=facemesh_exp_inject_stage,
                debug=facemesh_exp_debug,
                lips_mask_indices=LIP_IDX,
                corner_mask_indices=LIP_CORNER_IDX,
                verbose=True,
            )

            if exp_delta_fm is not None and fm_debug_dict.get("ok", False):
                # Combine: exp_delta = exp_delta + beta * exp_delta_fm
                # (beta is already applied inside apply_facemesh_exp_assist, so just add)
                exp_delta = exp_delta + exp_delta_fm
                print(f"[FaceMesh Exp Assist] [OK] Applied correction (shape: {tuple(exp_delta_fm.shape)})")
            else:
                error_msg = fm_debug_dict.get("error", "unknown")
                print(f"[FaceMesh Exp Assist] [SKIP] Skipped (stub or error: {error_msg})")

        except Exception as e:
            print(f"[FaceMesh Exp Assist] [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()

    # inject delta into target keypoints and render
    T_kp_mod = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in T_kp.items()}
    T_kp_mod["exp"] = T_kp_mod["exp"] + exp_delta

    kp_source = wrap.transform_keypoint(T_kp)
    kp_driving = wrap.transform_keypoint(T_kp_mod)
    feat_3d = wrap.extract_feature_3d(T)

    out_base = wrap.warp_decode(feat_3d, kp_source, kp_source)
    img_base = wrap.parse_output(out_base["out"])[0]

    out_asym = wrap.warp_decode(feat_3d, kp_source, kp_driving)
    img_asym = wrap.parse_output(out_asym["out"])[0]

    save_keypoint_map(
        wrap,
        img_asym,
        T_kp_mod,
        "outputs/diagnostics/result_keypoint_map.jpg",
    )

    if lip_metrics:
        lip_dir = osp.join("outputs", "lip_metrics")
        os.makedirs(lip_dir, exist_ok=True)

        donor_kp_pixels = project_keypoints_to_image(wrap, D_kp, donor_rgb)
        donor_metrics = compute_lip_metrics(donor_kp_pixels, LIP_IDX, donor_rgb.shape, verbose=True, corner_idx=LIP_CORNER_IDX)

        target_before_metrics = compute_lip_metrics(target_kp_pixels, LIP_IDX, target_rgb.shape, corner_idx=LIP_CORNER_IDX)
        target_after_kp_pixels = project_keypoints_to_image(wrap, T_kp_mod, img_asym)
        target_after_metrics = compute_lip_metrics(target_after_kp_pixels, LIP_IDX, img_asym.shape, corner_idx=LIP_CORNER_IDX)

        def _fmt(name: str, info: Optional[Dict[str, np.ndarray]]) -> str:
            if info is None:
                return f"{name}: n/a"
            return f"{name}: gap={info['lip_gap']:.2f}, width={info['lip_width']:.2f}"

        print("[lip-metrics]")
        print("  " + _fmt("donor", donor_metrics))
        print("  " + _fmt("target_before", target_before_metrics))
        print("  " + _fmt("target_after", target_after_metrics))

        if donor_metrics is not None:
            draw_lip_metrics_overlay(donor_rgb, donor_metrics, osp.join(lip_dir, "donor_lip_metrics.png"))
        if target_before_metrics is not None:
            draw_lip_metrics_overlay(target_rgb, target_before_metrics, osp.join(lip_dir, "target_before_lip_metrics.png"))
        if target_after_metrics is not None:
            draw_lip_metrics_overlay(img_asym, target_after_metrics, osp.join(lip_dir, "target_after_lip_metrics.png"))

    # =========================================================================
    # PHASE 3-5: FACEMESH WARP POST-PROCESS
    # =========================================================================
    # Apply post-process warp to transfer donor asymmetry using dense FaceMesh control points

    if facemesh_driving and facemesh_warp:
        print("\n[FaceMesh Warp] Starting post-process warp phase...")

        try:
            # Extract landmarks on LivePortrait output (target_after)
            ok_out, L_out, L_out_vis = extractor.extract(img_asym)

            if not ok_out:
                print("[FaceMesh Warp] ✗ Failed to extract landmarks on output image, skipping warp")
            else:
                print(f"[FaceMesh Warp] ✓ Extracted {len(L_out)} landmarks on output image")

                # Apply warp
                warp_output_dir = "outputs/diagnostics/facemesh_warp"
                os.makedirs(warp_output_dir, exist_ok=True)

                guard_enable = facemesh_guards if facemesh_guards is not None else facemesh_warp
                guard_args = {
                    "guard_max_delta_px": guard_max_delta_px,
                    "guard_cap_percentile": guard_cap_percentile,
                    "guard_cap_region": guard_cap_region,
                    "guard_cap_after_align": guard_cap_after_align,
                    "guard_smooth_delta": guard_smooth_delta,
                    "guard_smooth_iterations": guard_smooth_iterations,
                    "guard_smooth_lambda": guard_smooth_lambda,
                    "guard_smooth_mode": guard_smooth_mode,
                    "guard_knn_k": guard_knn_k,
                    "guard_zero_anchor": guard_zero_anchor,
                    "guard_anchor_idx": guard_anchor_idx,
                    "guard_anchor_strength": guard_anchor_strength,
                    "guard_softmask": guard_softmask,
                    "guard_softmask_sigma": guard_softmask_sigma,
                    "guard_softmask_forehead_fade": guard_softmask_forehead_fade,
                    "guard_softmask_forehead_yfrac": guard_softmask_forehead_yfrac,
                    "guard_softmask_min": guard_softmask_min,
                    "guard_softmask_max": guard_softmask_max,
                    "guard_face_mask": guard_face_mask,
                    "guard_face_mask_mode": guard_face_mask_mode,
                    "guard_face_mask_dilate": guard_face_mask_dilate,
                    "guard_face_mask_erode": guard_face_mask_erode,
                    "guard_face_mask_blur": guard_face_mask_blur,
                    "guard_warp_face_only": guard_warp_face_only,
                    "guard_mouth_only": guard_mouth_only,
                    "guard_mouth_radius_px": guard_mouth_radius_px,
                    "guard_alpha_start": guard_alpha_start,
                    "facemesh_guard_debug": facemesh_guard_debug,
                    "facemesh_warp_alpha": facemesh_warp_alpha,
                }

                if guard_face_mask_mode != "hull":
                    print("[FaceMesh Guard] segmentation mode not implemented; falling back to hull")
                    guard_args["guard_face_mask_mode"] = "hull"
                if guard_smooth_mode not in ["knn", "graph"]:
                    print("[FaceMesh Guard] Unknown smooth mode; using knn")
                    guard_args["guard_smooth_mode"] = "knn"
                if guard_mouth_only and facemesh_warp_alpha > max(guard_alpha_start, 0.6):
                    print("[FaceMesh Guard] Warning: mouth-only mode works best with alpha <= 0.6")

                ok_warp, img_asym_warped, warp_summary = apply_facemesh_warp(
                    img_asym,
                    facemesh_result["L_d"],
                    facemesh_result["delta"],
                    facemesh_result["weights"],
                    L_out,
                    L_out_vis=L_out_vis,
                    alpha=facemesh_warp_alpha,
                    reg=facemesh_warp_reg,
                    grid_step=facemesh_warp_grid_step,
                    lock_boundary=facemesh_warp_lock_boundary,
                    verbose=True,
                    output_dir=warp_output_dir,
                    guards=guard_enable,
                    guard_args=guard_args,
                    regions_config=regions_config,
                    guard_output_dir=os.path.join(warp_output_dir, "guards"),
                    save_debug=facemesh_debug
                )

                if ok_warp and img_asym_warped is not None:
                    print("[FaceMesh Warp] ✓ Warp succeeded")

                    # Apply warp result to img_asym for downstream processing
                    img_asym = img_asym_warped

                    # Save displacement field if requested
                    if facemesh_warp_save_field and "disp_field" in warp_summary:
                        np.save(
                            osp.join(warp_output_dir, "disp_field.npy"),
                            warp_summary["disp_field"]
                        )

                    # Run validation if enabled
                    if facemesh_warp_validate:
                        print("[FaceMesh Warp] Running validation...")

                        # Compute target landmarks for all control points
                        # Use the aligned delta from warp computation for correctness
                        L_out_target = L_out.copy()
                        delta_out_aligned = warp_summary.get("delta_out_aligned", None)

                        if delta_out_aligned is None:
                            # Fallback: recompute (should not happen in normal flow)
                            delta_out_aligned = facemesh_result["delta"] @ warp_summary.get("sR", np.eye(2))

                        sel_idx_validate = np.where(facemesh_result["weights"] > 0)[0]
                        for idx in sel_idx_validate:
                            if 0 <= idx < 468:
                                L_out_target[idx] = L_out[idx] + facemesh_warp_alpha * delta_out_aligned[idx]

                        validation = validate_warp(
                            img_asym,
                            L_out_target,
                            sel_idx_validate,
                            extractor,
                            output_dir=warp_output_dir,
                            verbose=True
                        )

                        warp_summary.update(validation)

                    # Phase 7: evaluation metrics
                    metrics_summary = {}
                    donor_metrics = compute_metrics(facemesh_result.get("L_d") if facemesh_result else None)
                    out_before_metrics = compute_metrics(L_out)
                    out_after_metrics = None

                    ok_after_metrics, L_after, _ = extractor.extract(img_asym)
                    if ok_after_metrics:
                        out_after_metrics = compute_metrics(L_after)

                    metrics_summary["donor"] = donor_metrics
                    metrics_summary["out_before"] = out_before_metrics
                    metrics_summary["out_after"] = out_after_metrics

                    def _fmt_metrics(name: str, m: Optional[Dict[str, float]]) -> str:
                        if m is None:
                            return f"{name}: score=N/A"
                        return (
                            f"{name}: score={m['score']:.3f}, "
                            f"droop={m['droop']:.3f}, tilt={m['tilt_deg']:.3f}, "
                            f"cheek={m['cheek_diff']:.3f}, sag={m['sag']:.3f}"
                        )

                    print("\nFACEMESH ASYMMETRY METRICS")
                    print("  " + _fmt_metrics("donor", donor_metrics))
                    print("  " + _fmt_metrics("out_before", out_before_metrics))
                    print("  " + _fmt_metrics("out_after", out_after_metrics))

                    metrics_path = osp.join(warp_output_dir, "metrics.json")
                    with open(metrics_path, "w") as f:
                        json.dump(metrics_summary, f, indent=2)
                    warp_summary["metrics_path"] = metrics_path

                    # Save warp summary
                    summary_path = osp.join(warp_output_dir, "warp_summary.json")
                    summary_to_save = {
                        k: v for k, v in warp_summary.items()
                        if isinstance(v, (int, float, str, bool, type(None)))
                    }
                    with open(summary_path, "w") as f:
                        json.dump(summary_to_save, f, indent=2)

                    print(f"[FaceMesh Warp] ✓ Summary saved to {summary_path}")
                else:
                    print("[FaceMesh Warp] ✗ Warp failed, using original output")
                    warp_summary.update({"warp_succeeded": False})

        except Exception as e:
            print(f"[FaceMesh Warp] Error: {e}")
            import traceback
            traceback.print_exc()

    # save diagnostics
    os.makedirs("outputs/diagnostics", exist_ok=True)
    out_w_res, out_h_res = resolve_output_size(target_rgb, out_w, out_h)
    img_base_lb = letterbox_to(img_base, out_w_res, out_h_res)
    img_asym_lb = letterbox_to(img_asym, out_w_res, out_h_res)
    donor_lb = letterbox_to(donor_rgb, out_w_res, out_h_res)

    side_img = np.hstack([img_base_lb, img_asym_lb])
    cv2.imwrite("outputs/diagnostics/baseline_vs_asym.jpg", cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR))

    side_dr = np.hstack([donor_lb, img_asym_lb])
    cv2.imwrite("outputs/diagnostics/donor_vs_result.jpg", cv2.cvtColor(side_dr, cv2.COLOR_RGB2BGR))

    # final output
    save_rgb(out_path, img_asym_lb)


    masks_fin = compute_masks(kp_can_t)
    lips_mask_fin = masks_fin["lips_mask"]
    corner_mask_fin = masks_fin["corner_mask"]
    # if lips_mask_fin.any():
    #     lip_delta_y = exp_delta[0, lips_mask_fin, 1].abs()
    #     print(f"Lip Y |mean|={lip_delta_y.mean():.6f} |max|={lip_delta_y.max():.6f}")
    # if corner_mask_fin.any():
    #     corner_delta_y = exp_delta[0, corner_mask_fin, 1].abs()
    #     print(f"Corner Y |mean|={corner_delta_y.mean():.6f} |max|={corner_delta_y.max():.6f}")

    def _region_ratio(base: torch.Tensor, now: torch.Tensor, m: torch.Tensor) -> float:
        if not m.any():
            return 0.0
        b = base[0, m, 1].abs().mean().item() + 1e-8
        n = now[0, m, 1].abs().mean().item()
        return float(n / b)

    r_corners = _region_ratio(S1, exp_delta, corner_mask_fin)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Left-Right asymmetry transfer (donor -> neutral target)")
    # donor image path
    ap.add_argument("--donor","-d", required=True)
    # target image path
    ap.add_argument("--target","-t", required=True)
    # output path
    ap.add_argument("--out","-o", default="outputs/asym_transfer.jpg")
    # model config path
    ap.add_argument("--config", default="src/config/models.yaml")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--edge_dampen", type=float, default=0.55)
    # output width
    ap.add_argument("--out_w", type=int, default=0)
    # output height
    ap.add_argument("--out_h", type=int, default=0)
    ap.add_argument("--asym_side", type=str, default="right", choices=["right","left"])
    # Midline control
    ap.add_argument("--midline_eps", type=float, default=0.05)
    # Extreme clamp control
    ap.add_argument("--clamp_q", type=float, default=0.90)
    # Region/boost controls
    ap.add_argument("--boost_base", type=float, default=0.50)
    ap.add_argument("--eye_gain", type=float, default=0.70)
    ap.add_argument("--brow_gain", type=float, default=0.85)
    ap.add_argument("--lip_gain", type=float, default=1.40)
    ap.add_argument("--boost_base_floor", type=float, default=0.5)
    ap.add_argument("--boost_focus_gain", type=float, default=0.5)
    ap.add_argument("--lip_gain_x", type=float, default=None)
    ap.add_argument("--lip_gain_y", type=float, default=None)
    ap.add_argument("--corner_gain", type=float, default=1.6)
    # Global motion cap control
    ap.add_argument("--norm_cap", type=float, default=2.5)
    # Pose alignment
    ap.add_argument("--auto_flip", action="store_true")
    # Mouth symmetry blend (0=off)
    ap.add_argument("--mouth_sym_alpha", type=float, default=0.0)
    # Y-drift correction
    ap.add_argument("--y_drift_fix", type=str, default="nonmouth", choices=["none", "global", "nonmouth"])
    ap.add_argument("--y_anchor", type=float, default=0.5)
    ap.add_argument("--y_drift_mouth_bias", type=float, default=0.0)
    # Debug: quantify mouth opening (gap) and corner spread (width) for stroke-related asymmetry checks
    ap.add_argument("--lip-metrics", action="store_true", help="Compute lip gap/width metrics and debug overlay.")

    # FaceMesh-based asymmetry analysis
    ap.add_argument("--facemesh-driving", action="store_true",
                    help="Enable MediaPipe FaceMesh-based donor asymmetry analysis.")
    ap.add_argument("--facemesh-debug", action="store_true",
                    help="Save FaceMesh debug visualizations (overlays, heatmaps).")
    ap.add_argument("--facemesh-regions", type=str, default=None,
                    help="Path to JSON config for FaceMesh region indices.")
    ap.add_argument("--facemesh-max-delta-px", type=float, default=None,
                    help="Absolute maximum delta magnitude in pixels (default: percentile-based).")
    ap.add_argument("--facemesh-clamp-percentile", type=float, default=98.0,
                    help="Percentile for auto-clamping delta magnitudes (default: 98).")
    ap.add_argument("--facemesh-save-npy", action="store_true",
                    help="Save FaceMesh numpy arrays and summary JSON.")
    ap.add_argument("--cheek-radius-px", type=float, default=None,
                    help="Override radius for dynamic cheek patch in pixels.")
    ap.add_argument("--facemesh-refine", action="store_true",
                    help="Use refined FaceMesh landmarks (if supported).")

    # FaceMesh Expression Assist - inject FaceMesh mouth signal into exp_delta
    ap.add_argument("--facemesh-exp-assist", action="store_true",
                    help="Enable FaceMesh-based mouth correction in exp_delta pipeline.")
    ap.add_argument("--facemesh-exp-beta", type=float, default=1.0,
                    help="Overall scaling factor for FaceMesh exp correction (default 1.0).")
    ap.add_argument("--facemesh-exp-mouth-alpha", type=float, default=1.0,
                    help="Extra scaling for mouth region (default 1.0).")
    ap.add_argument("--facemesh-exp-method", type=str, default="knn", choices=["knn", "tps", "basis"],
                    help="Projection method: 'knn' (default), 'tps', or 'basis' (shape-preserving).")
    ap.add_argument("--facemesh-exp-knn-k", type=int, default=8,
                    help="K for KNN projection (default 8).")
    ap.add_argument("--facemesh-exp-inject-stage", type=str, default="post_drift",
                    choices=["pre_gain", "pre_drift", "post_drift"],
                    help="When to inject FaceMesh correction: 'pre_gain' (before gains), 'pre_drift' (before drift fix), or 'post_drift' (default).")
    ap.add_argument("--facemesh-exp-debug", action="store_true",
                    help="Save FaceMesh exp assist debug outputs (JSON summary).")
    # FaceMesh Expression Assist - Guardrails (Phase 3)
    ap.add_argument("--facemesh-exp-max-disp-px", type=float, default=None,
                    help="Absolute cap for displacement magnitude in pixels. If not set, uses percentile.")
    ap.add_argument("--facemesh-exp-cap-percentile", type=float, default=98.0,
                    help="Percentile for auto-cap of displacements (default 98.0, range 0-100).")
    ap.add_argument("--facemesh-exp-smooth", dest="facemesh_exp_smooth",
                    action="store_true", help="Enable KNN-based smoothing of lip displacements (default: True).")
    ap.add_argument("--no-facemesh-exp-smooth", dest="facemesh_exp_smooth",
                    action="store_false", help="Disable KNN-based smoothing of lip displacements.")
    ap.set_defaults(facemesh_exp_smooth=True)
    ap.add_argument("--facemesh-exp-smooth-k", type=int, default=6,
                    help="K for smoothing neighbors (default 6).")
    ap.add_argument("--facemesh-exp-zero-stable", action="store_true", default=True,
                    help="Zero out non-mouth keypoints (default: True, prevents 'cursed' outputs).")

    # FaceMesh-based post-process warp (Phase 3-5)
    ap.add_argument("--facemesh-warp", action="store_true",
                    help="Enable post-process warp to transfer donor asymmetry to output.")
    ap.add_argument("--facemesh-warp-method", type=str, default="tps", choices=["tps", "pwa"],
                    help="Warp method: 'tps' (Thin Plate Spline, default) or 'pwa' (piecewise affine).")
    ap.add_argument("--facemesh-warp-alpha", type=float, default=1.0,
                    help="Warp strength (0-1, default 1.0). Lower = weaker deformation.")
    ap.add_argument("--facemesh-warp-reg", type=float, default=1e-3,
                    help="TPS regularization parameter (default 1e-3). Higher = smoother.")
    ap.add_argument("--facemesh-warp-grid-step", type=int, default=2,
                    help="TPS coarse-to-fine grid step (default 2). 1=full resolution, higher=faster.")
    ap.add_argument("--facemesh-warp-lock-boundary", dest="facemesh_warp_lock_boundary",
                    action="store_true", help="Lock boundary points during warp")
    ap.add_argument("--no-facemesh-warp-lock-boundary", dest="facemesh_warp_lock_boundary",
                    action="store_false", help="Disable boundary locking during warp")
    ap.set_defaults(facemesh_warp_lock_boundary=True)
    ap.add_argument("--facemesh-warp-validate", action="store_true", default=True,
                    help="Validate warp by re-running FaceMesh on warped output (default: True).")
    ap.add_argument("--facemesh-warp-save-field", action="store_true",
                    help="Save displacement field as numpy array for inspection.")

    # Phase 6: Guardrails
    ap.add_argument("--facemesh-guards", dest="facemesh_guards", action="store_true", default=None,
                    help="Enable guardrails (defaults to True when facemesh-warp is on).")
    ap.add_argument("--no-facemesh-guards", dest="facemesh_guards", action="store_false", default=None,
                    help="Disable guardrails explicitly.")
    ap.add_argument("--facemesh-guard-debug", action="store_true",
                    help="Save guardrail debug artifacts (masks, capped deltas).")
    ap.add_argument("--guard-max-delta-px", type=float, default=None,
                    help="Absolute per-point delta cap (px). If None, use percentile cap.")
    ap.add_argument("--guard-cap-percentile", type=float, default=98.0,
                    help="Percentile for automatic cap (default 98).")
    ap.add_argument("--guard-cap-region", type=str, default="weighted_only", choices=["weighted_only", "all"],
                    help="Which points to consider when computing cap.")
    ap.add_argument("--guard-cap-after-align", action="store_true", default=True,
                    help="Apply cap after delta alignment (default True).")
    ap.add_argument("--no-guard-cap-after-align", dest="guard_cap_after_align", action="store_false",
                    help="Skip cap after alignment.")
    ap.add_argument("--guard-smooth-delta", action="store_true", default=True,
                    help="Enable spatial smoothing of delta (default True).")
    ap.add_argument("--no-guard-smooth-delta", dest="guard_smooth_delta", action="store_false",
                    help="Disable spatial smoothing.")
    ap.add_argument("--guard-smooth-iterations", type=int, default=2,
                    help="Number of smoothing iterations (default 2).")
    ap.add_argument("--guard-smooth-lambda", type=float, default=0.6,
                    help="Smoothing blend factor (0-1, default 0.6).")
    ap.add_argument("--guard-smooth-mode", type=str, default="knn", choices=["knn", "graph"],
                    help="Smoothing mode (default knn).")
    ap.add_argument("--guard-knn-k", type=int, default=8,
                    help="K for KNN smoothing (default 8).")
    ap.add_argument("--guard-zero-anchor", action="store_true", default=True,
                    help="Dampen anchor deltas toward zero (default True).")
    ap.add_argument("--no-guard-zero-anchor", dest="guard_zero_anchor", action="store_false",
                    help="Disable anchor damping.")
    ap.add_argument("--guard-anchor-idx", type=int, nargs="*", default=None,
                    help="Optional override for anchor indices (space-separated list).")
    ap.add_argument("--guard-anchor-strength", type=float, default=0.95,
                    help="Anchor damping strength (default 0.95).")
    ap.add_argument("--guard-softmask", action="store_true", default=True,
                    help="Enable soft effect mask (default True).")
    ap.add_argument("--no-guard-softmask", dest="guard_softmask", action="store_false",
                    help="Disable soft effect mask.")
    ap.add_argument("--guard-softmask-sigma", type=float, default=25.0,
                    help="Gaussian sigma for soft mask (px).")
    ap.add_argument("--guard-softmask-forehead-fade", action="store_true", default=True,
                    help="Fade mask near forehead (default True).")
    ap.add_argument("--no-guard-softmask-forehead-fade", dest="guard_softmask_forehead_fade", action="store_false",
                    help="Disable forehead fade in mask.")
    ap.add_argument("--guard-softmask-forehead-yfrac", type=float, default=0.22,
                    help="Top fraction to fade mask (default 0.22).")
    ap.add_argument("--guard-softmask-min", type=float, default=0.0,
                    help="Minimum mask value after clamping (default 0.0).")
    ap.add_argument("--guard-softmask-max", type=float, default=1.0,
                    help="Maximum mask value after clamping (default 1.0).")
    ap.add_argument("--guard-face-mask", action="store_true", default=True,
                    help="Enable face hull mask (default True).")
    ap.add_argument("--no-guard-face-mask", dest="guard_face_mask", action="store_false",
                    help="Disable face hull mask.")
    ap.add_argument("--guard-face-mask-mode", type=str, default="hull", choices=["hull", "segmentation"],
                    help="Face mask mode (segmentation not implemented, hull default).")
    ap.add_argument("--guard-face-mask-dilate", type=int, default=12,
                    help="Dilation pixels for face mask (default 12).")
    ap.add_argument("--guard-face-mask-erode", type=int, default=0,
                    help="Erode pixels for face mask (default 0).")
    ap.add_argument("--guard-face-mask-blur", type=int, default=11,
                    help="Gaussian blur kernel for face mask (odd, default 11).")
    ap.add_argument("--guard-warp-face-only", action="store_true", default=True,
                    help="Composite warped face onto original background (default True).")
    ap.add_argument("--no-guard-warp-face-only", dest="guard_warp_face_only", action="store_false",
                    help="Disable face-only compositing.")
    ap.add_argument("--guard-mouth-only", action="store_true",
                    help="Mouth-only MVP warp (default False).")
    ap.add_argument("--guard-mouth-radius-px", type=int, default=90,
                    help="Mouth-only mask radius in pixels (default 90).")
    ap.add_argument("--guard-alpha-start", type=float, default=0.3,
                    help="Recommended starting alpha for mouth-only (default 0.3).")

    a = ap.parse_args()
    main(
        donor_path=a.donor,
        target_path=a.target,
        out_path=a.out,
        cfg_path=a.config,
        scale=a.scale,
        edge_dampen=a.edge_dampen,
        out_w=a.out_w,
        out_h=a.out_h,
        asym_side=a.asym_side,
        midline_eps=a.midline_eps,
        clamp_q=a.clamp_q,
        boost_base=a.boost_base,
        eye_gain=a.eye_gain,
        brow_gain=a.brow_gain,
        lip_gain=a.lip_gain,
        auto_flip=a.auto_flip,
        boost_base_floor=a.boost_base_floor,
        boost_focus_gain=a.boost_focus_gain,
        lip_gain_x=a.lip_gain_x,
        lip_gain_y=a.lip_gain_y,
        corner_gain=a.corner_gain,
        norm_cap=a.norm_cap,
        mouth_sym_alpha=a.mouth_sym_alpha,
        y_drift_fix=a.y_drift_fix,
        y_anchor=a.y_anchor,
        y_drift_mouth_bias=a.y_drift_mouth_bias,
        lip_metrics=a.lip_metrics,
        facemesh_driving=a.facemesh_driving,
        facemesh_debug=a.facemesh_debug,
        facemesh_regions=a.facemesh_regions,
        facemesh_max_delta_px=a.facemesh_max_delta_px,
        facemesh_clamp_percentile=a.facemesh_clamp_percentile,
        facemesh_save_npy=a.facemesh_save_npy,
        cheek_radius_px=a.cheek_radius_px,
        facemesh_refine=a.facemesh_refine,
        facemesh_exp_assist=a.facemesh_exp_assist,
        facemesh_exp_beta=a.facemesh_exp_beta,
        facemesh_exp_mouth_alpha=a.facemesh_exp_mouth_alpha,
        facemesh_exp_method=a.facemesh_exp_method,
        facemesh_exp_knn_k=a.facemesh_exp_knn_k,
        facemesh_exp_inject_stage=a.facemesh_exp_inject_stage,
        facemesh_exp_debug=a.facemesh_exp_debug,
        facemesh_exp_max_disp_px=a.facemesh_exp_max_disp_px,
        facemesh_exp_cap_percentile=a.facemesh_exp_cap_percentile,
        facemesh_exp_smooth=a.facemesh_exp_smooth,
        facemesh_exp_smooth_k=a.facemesh_exp_smooth_k,
        facemesh_exp_zero_stable=a.facemesh_exp_zero_stable,
        facemesh_warp=a.facemesh_warp,
        facemesh_warp_method=a.facemesh_warp_method,
        facemesh_warp_alpha=a.facemesh_warp_alpha,
        facemesh_warp_reg=a.facemesh_warp_reg,
        facemesh_warp_grid_step=a.facemesh_warp_grid_step,
        facemesh_warp_lock_boundary=a.facemesh_warp_lock_boundary,
        facemesh_warp_validate=a.facemesh_warp_validate,
        facemesh_warp_save_field=a.facemesh_warp_save_field,
        facemesh_guards=a.facemesh_guards,
        facemesh_guard_debug=a.facemesh_guard_debug,
        guard_max_delta_px=a.guard_max_delta_px,
        guard_cap_percentile=a.guard_cap_percentile,
        guard_cap_region=a.guard_cap_region,
        guard_cap_after_align=a.guard_cap_after_align,
        guard_smooth_delta=a.guard_smooth_delta,
        guard_smooth_iterations=a.guard_smooth_iterations,
        guard_smooth_lambda=a.guard_smooth_lambda,
        guard_smooth_mode=a.guard_smooth_mode,
        guard_knn_k=a.guard_knn_k,
        guard_zero_anchor=a.guard_zero_anchor,
        guard_anchor_idx=a.guard_anchor_idx,
        guard_anchor_strength=a.guard_anchor_strength,
        guard_softmask=a.guard_softmask,
        guard_softmask_sigma=a.guard_softmask_sigma,
        guard_softmask_forehead_fade=a.guard_softmask_forehead_fade,
        guard_softmask_forehead_yfrac=a.guard_softmask_forehead_yfrac,
        guard_softmask_min=a.guard_softmask_min,
        guard_softmask_max=a.guard_softmask_max,
        guard_face_mask=a.guard_face_mask,
        guard_face_mask_mode=a.guard_face_mask_mode,
        guard_face_mask_dilate=a.guard_face_mask_dilate,
        guard_face_mask_erode=a.guard_face_mask_erode,
        guard_face_mask_blur=a.guard_face_mask_blur,
        guard_warp_face_only=a.guard_warp_face_only,
        guard_mouth_only=a.guard_mouth_only,
        guard_mouth_radius_px=a.guard_mouth_radius_px,
        guard_alpha_start=a.guard_alpha_start,
    )
