import sys, os, os.path as osp, cv2, torch
import numpy as np
from typing import Dict

# make "src" importable when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper

def read_rgb(p):
    # Read an image from disk as RGB
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_rgb(p, img_rgb):
    # Save an RGB image to disk, mkdir as needed
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
    """ Decide final output size
    - If neither width nor height provided: use target image size
    - If one side provided: compute the other to preserve target aspect
    - If both provided: use them as is
    """
    Ht, Wt = target_rgb.shape[:2]
    if out_w <= 0 and out_h <= 0:
        return Wt, Ht
    if out_w <= 0 and out_h > 0:
        return int(round(out_h * (Wt / Ht))), out_h
    if out_h <= 0 and out_w > 0:
        return out_w, int(round(out_w * (Ht / Wt)))
    return out_w, out_h

def procrustes_scale(src_xy: torch.Tensor, dst_xy: torch.Tensor) -> float:
    """ Compute the best-fit similarity *scale* that maps src->dst (least-squares)
    Only keep the scalar (zoom). Adjusts for different zooms in X/Y
    """
    src = src_xy - src_xy.mean(dim=0, keepdim=True)
    dst = dst_xy - dst_xy.mean(dim=0, keepdim=True)
    src_norm = torch.linalg.norm(src)
    dst_norm = torch.linalg.norm(dst)
    if float(src_norm) < 1e-8:
        return 1.0
    return float((dst_norm / (src_norm + 1e-8)).item())

def similarity_decompose(src_xy: torch.Tensor, dst_xy: torch.Tensor):
    # Compute best-fit 2x2 rotation matrix. Adjusts for rotation in X/Y
    src = src_xy - src_xy.mean(dim=0, keepdim=True)
    dst = dst_xy - dst_xy.mean(dim=0, keepdim=True)
    H = src.T @ dst
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt
    if torch.det(R) < 0:  # handle reflections to keep a proper rotation
        Vt[-1, :] *= -1
        R = U @ Vt
    theta = float(torch.atan2(R[1, 0], R[0, 0]).item())
    return R, theta

def affine_detrend_dx(dx: torch.Tensor, kp_xy: torch.Tensor):
    # Remove global widening from X motion by fitting: dx ≈ a*(x-xc) + b*(y-yc) + c and subtracting it off
    x = kp_xy[:, 0]; y = kp_xy[:, 1]
    xc = x.mean(); yc = y.mean()
    X = torch.stack([x - xc, y - yc, torch.ones_like(x)], dim=1)  # (N,3)
    coeff = torch.linalg.pinv(X) @ dx  # (3,)
    trend = X @ coeff  # (N,)
    return dx - trend, coeff

def affine_detrend_dy(dy: torch.Tensor, kp_xy: torch.Tensor):
    # Remove global vertical drift/scale/shear in Y: dy ≈ a*(x-xc) + b*(y-yc) + c
    x = kp_xy[:, 0]; y = kp_xy[:, 1]
    xc = x.mean(); yc = y.mean()
    X = torch.stack([x - xc, y - yc, torch.ones_like(x)], dim=1)  # (N,3)
    coeff = torch.linalg.pinv(X) @ dy  # (3,)
    trend = X @ coeff
    return dy - trend, coeff

def soft_knee_vec(V: torch.Tensor, tau: float) -> torch.Tensor:
    # Compress vector magnitudes to avoid extreme values
    # m' = tau * tanh(m / tau) -> behaves linear for small m, saturates for big m
    m = torch.linalg.norm(V, dim=-1, keepdim=True) + 1e-8  # per keypoint motion magnitude
    m_prime = float(tau) * torch.tanh(m / float(tau))
    gain = (m_prime / m).clamp(max=1.0)  # dont amplify
    return V * gain

def soft_knee_scalar(d: torch.Tensor, tau: float) -> torch.Tensor:
    # Scalar version for X only
    a = d.abs() + 1e-8
    a_prime = float(tau) * torch.tanh(a / float(tau))
    return torch.sign(d) * a_prime

# ====== NEW: simple keypoint and mouth-vector visualizers ======
def draw_kp_map_simple(img_rgb: np.ndarray, kp_xy: torch.Tensor, path: str, mouth_idxs=None):
    """Render all keypoints; color mouth differently; label indices for mouth."""
    import cv2, numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas = img_rgb.copy()
    H, W = canvas.shape[:2]
    xy = kp_xy.detach().cpu().numpy()
    x = xy[:,0]; y = xy[:,1]
    # normalize to image box for plotting
    x_ = (x - x.min()) / max(1e-6, (x.max() - x.min()))
    y_ = (y - y.min()) / max(1e-6, (y.max() - y.min()))
    px = (x_ * (W-1)).astype(np.int32)
    py = (y_ * (H-1)).astype(np.int32)

    # draw all points
    for i,(u,v) in enumerate(zip(px,py)):
        c = (40, 220, 255) if (mouth_idxs is not None and i in mouth_idxs) else (200,200,200)
        cv2.circle(canvas, (u,v), 3, c, -1, lineType=cv2.LINE_AA)
        if mouth_idxs is not None and i in mouth_idxs:
            cv2.putText(canvas, str(i), (u+3, v-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def draw_mouth_vectors(img_rgb: np.ndarray, kp_xy: torch.Tensor, delta: torch.Tensor, mouth_mask: torch.Tensor, path: str, scale_px: float=180.0):
    """Plot motion vectors on mouth keypoints only."""
    import cv2, numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas = img_rgb.copy()
    H, W = canvas.shape[:2]
    xy = kp_xy.detach().cpu().numpy()
    d  = delta.detach().cpu().numpy()  # (N,3)
    x = xy[:,0]; y = xy[:,1]
    # normalize to image box
    x_ = (x - x.min()) / max(1e-6, (x.max() - x.min()))
    y_ = (y - y.min()) / max(1e-6, (y.max() - y.min()))
    px = (x_ * (W-1)).astype(np.int32)
    py = (y_ * (H-1)).astype(np.int32)

    mouth_idx = np.nonzero(mouth_mask.detach().cpu().numpy())[0]
    for i in mouth_idx:
        u, v = int(px[i]), int(py[i])
        du = int(d[i,0] * scale_px)
        dv = int(d[i,1] * scale_px)
        cv2.arrowedLine(canvas, (u,v), (u+du, v+dv), (0,180,255), 2, tipLength=0.25)
        cv2.circle(canvas, (u,v), 2, (255,255,255), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
# ===============================================================

# ====== DIAGNOSTICS HELPERS ======
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
    """Print stats for X and Y on lip/corner masks."""
    print(f"\n[STAGE {stage_name}]")
    for k in ["lips_mask", "corner_mask", "center_lip_mask"]:
        if k not in mask_dict:
            continue
        m = mask_dict[k]
        n = int(m.sum().item())
        idx_show = torch.nonzero(m, as_tuple=False).squeeze(-1).tolist()
        idx_show = idx_show[:8] if isinstance(idx_show, list) else []
        print(f" - {k}: count={n} examples={idx_show}")
        if n > 0:
            dx = exp_delta[0, m, 0]
            dy = exp_delta[0, m, 1]
            sx = _summ_stats(dx)
            sy = _summ_stats(dy)
            print(f" X: mean={sx['mean']:.5f} med={sx['median']:.5f} max|.|={sx['max']:.5f} p05={sx['p05']:.5f} p95={sx['p95']:.5f}")
            print(f" Y: mean={sy['mean']:.5f} med={sy['median']:.5f} max|.|={sy['max']:.5f} p05={sy['p05']:.5f} p95={sy['p95']:.5f}")

def draw_masks_overlay(img_rgb: np.ndarray, kpts_xy: torch.Tensor, masks: Dict[str, torch.Tensor], path: str):
    """Save a quick overlay to verify mouth coverage."""
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
    lips_mask = (yn > -0.35) & (xn.abs() < 0.95)
    corner_mask = (yn > -0.20) & (xn.abs() >= 0.30) & (xn.abs() < 0.95)
    center_lip_mask = (xn.abs() < 0.18) & (yn > -0.05) & (yn < 0.45)
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
    kp2[:, 0] *= -1  # mirror canonical X for flipped donor

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
    norm_cap=2.5,  # NEW:
    mouth_sym_alpha=0.0,
    y_drift_fix="nonmouth",  # "none" | "global" | "nonmouth"
    y_anchor=0.5,  # 0..1
    # control how much of the nonmouth bias we subtract from mouth Y (0=keep mouth free)
    y_drift_mouth_bias: float = 0.0,
):
    # load models
    cfg = InferenceConfig(models_config=cfg_path)
    wrap = LivePortraitWrapper(cfg)

    # read donor (has asymmetry) and target (neutral avatar)
    donor_rgb = read_rgb(donor_path)
    target_rgb = read_rgb(target_path)

    # get target keypoints in canonical coords to define left/right etc
    T = wrap.prepare_source(target_rgb)
    T_kp = wrap.get_kp_info(T, flag_refine_info=True)
    kp_can_t = T_kp["kp"][0]  # (N,3)
    kp_can_t_xy = kp_can_t[:, :2]
    x_can = kp_can_t[:, 0]
    x_center = x_can.mean()
    left_mask = x_can < x_center
    right_mask = ~left_mask

    # A) Save a simple target keypoint map
    os.makedirs("outputs/diagnostics", exist_ok=True)
    draw_kp_map_simple(
        target_rgb,
        kp_can_t[:, :2],
        "outputs/diagnostics/target_kp.jpg"
    )

    # -------- Auto flip donor to best match target pose (optional) --------
    flip_used = False
    if auto_flip:
        donor_rgb, flip_used = choose_best_flip(wrap, donor_rgb, kp_can_t_xy)
        print(f"[auto_flip] Using {'FLIPPED' if flip_used else 'ORIGINAL'} donor orientation.")

    # donor LR delta via flip and map back
    E_donor = get_exp_tensor(wrap, donor_rgb)
    donor_flip = donor_rgb[:, ::-1, :].copy()
    E_flip = get_exp_tensor(wrap, donor_flip)
    E_flip_map = E_flip.clone(); E_flip_map[..., 0] *= -1

    # B) Visualize donor keypoints too
    D = wrap.prepare_source(donor_rgb)
    D_kp = wrap.get_kp_info(D, flag_refine_info=True)
    kp_can_d = D_kp["kp"][0]
    draw_kp_map_simple(
        donor_rgb,
        kp_can_d[:, :2],
        "outputs/diagnostics/donor_kp.jpg"
    )

    # Canonical LR delta (right-minus-left). Use flag to choose sign.
    raw_delta = (E_donor - E_flip_map)  # right minus left, canonical
    side = asym_side.lower()
    exp_delta = raw_delta * float(scale) if side == "right" else -raw_delta * float(scale)

    pre_norm = exp_delta.detach().float().norm().item()
    print("\n=== ASYMMETRY DETECTION ===")
    print(f"Raw delta magnitude: {raw_delta.abs().mean().item():.6f}")
    print(f"Max delta: {raw_delta.abs().max().item():.6f}")
    print(f"Delta X range: [{raw_delta[0,:,0].min():.4f}, {raw_delta[0,:,0].max():.4f}]")
    print(f"Delta Y range: [{raw_delta[0,:,1].min():.4f}, {raw_delta[0,:,1].max():.4f}]")

    # S0
    S0 = exp_delta.clone()

    # normalize donor vs target pose/size in canonical space
    # (recompute donor kp here as float and use for alignment)
    kp_can_d_xy = kp_can_d[:, :2]
    # scale (zoom) match
    s_ratio = procrustes_scale(kp_can_d_xy, kp_can_t_xy)
    exp_delta[0, :, 0:2] *= s_ratio
    # in-plane rotation (roll) match
    R, _theta = similarity_decompose(kp_can_d_xy, kp_can_t_xy)
    R = R.to(dtype=exp_delta.dtype, device=exp_delta.device)
    exp_delta[0, :, 0:2] = exp_delta[0, :, 0:2] @ R

    # S1
    S1 = exp_delta.clone()

    """ Keep motion concentrated around the middle of the face
    Cheeks tend to “swell/widen” if outer landmarks move laterally
    This weights X-motion down toward the edges
    """
    with torch.no_grad():
        xy = kp_can_t[:, :2]
        cx, cy = xy.mean(dim=0)
        dxn = (xy[:,0] - cx); dyn = (xy[:,1] - cy)
        sx = torch.std(dxn) + 1e-6
        sy = torch.std(dyn) + 1e-6

        sig_x = 0.95
        sig_y = 0.90
        w_roi = torch.exp(-0.5*((dxn/(sig_x*sx))**2 + (dyn/(sig_y*sy))**2))  # (N,)
        w_roi = 0.25 + 0.75*w_roi
        exp_delta[0,:,0] *= w_roi  # only X (sideways)

        # fully zero X on the outer ~12% of face width (hard guardrail)
        span_x = (kp_can_t[:,0] - kp_can_t[:,0].mean()).abs().max() + 1e-6
        edge_thresh = 0.88
        edge_mask = ((kp_can_t[:,0] - kp_can_t[:,0].mean()).abs() / span_x) > edge_thresh
        if int(edge_mask.sum()) > 0:
            exp_delta[0, edge_mask, 0] = 0.0

    # S2
    S2 = exp_delta.clone()

    # ---------------- Region-specific control (ADAPTIVE) ----------------
    mouth_mask = None  # will fill in and reuse later
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

        # ---- ADAPTIVE MOUTH BANDS (quantile-based instead of fixed numbers) ----
        # choose the lower-mid vertical band as "mouth zone" and allow corners a wider |x|
        y_lo = torch.quantile(yn, torch.tensor(0.45, device=yn.device))
        y_hi = torch.quantile(yn, torch.tensor(0.95, device=yn.device))

        # mid and wide horizontal bands for lips vs corners
        x_mid  = torch.quantile(xn.abs(), torch.tensor(0.60, device=xn.device))
        x_wide = torch.quantile(xn.abs(), torch.tensor(0.90, device=xn.device))

        lips_mask       = (yn >= y_lo) & (yn <= y_hi) & (xn.abs() <= x_mid)
        corner_mask     = (yn >= y_lo) & (yn <= y_hi) & (xn.abs() >  x_mid) & (xn.abs() <= x_wide)
        center_lip_mask = (yn >= y_lo) & (yn <= y_hi) & (xn.abs() <= (0.6 * x_mid))

        # --- Robust fallback if selection is too small (some faces/chkpts bunch up) ---
        if int(lips_mask.sum()) < 6:
            # widen vertically and horizontally until we get a reasonable set
            y_lo_f = torch.quantile(yn, torch.tensor(0.40, device=yn.device))
            y_hi_f = torch.quantile(yn, torch.tensor(0.98, device=yn.device))
            x_mid_f  = torch.quantile(xn.abs(), torch.tensor(0.75, device=xn.device))
            x_wide_f = torch.quantile(xn.abs(), torch.tensor(0.97, device=xn.device))
            lips_mask       = (yn >= y_lo_f) & (yn <= y_hi_f) & (xn.abs() <= x_mid_f)
            corner_mask     = (yn >= y_lo_f) & (yn <= y_hi_f) & (xn.abs() > x_mid_f) & (xn.abs() <= x_wide_f)
            center_lip_mask = (yn >= y_lo_f) & (yn <= y_hi_f) & (xn.abs() <= (0.6 * x_mid_f))

        # Eyes / brows (leave as-is; cap to not exceed 1x)
        eyes_mask     = (yn < torch.quantile(yn, torch.tensor(0.25, device=yn.device))) & (xn.abs() > torch.quantile(xn.abs(), torch.tensor(0.55, device=xn.device)))
        brows_mask    = (yn < torch.quantile(yn, torch.tensor(0.15, device=yn.device))) & (xn.abs() > torch.quantile(xn.abs(), torch.tensor(0.45, device=xn.device)))
        eye_brow_mask = eyes_mask | brows_mask

        # Kelvin-style base curve
        w_focus   = torch.exp(-0.5 * (xn**2 + yn**2))  # [0..1], peak at center
        kelvin_vec = (float(boost_base_floor) + float(boost_focus_gain) * w_focus)

        # Region gains
        g_eye   = float(eye_gain)
        g_brow  = float(brow_gain)
        g_lip   = float(lip_gain)
        g_lip_x = float(lip_gain_x) if lip_gain_x is not None else g_lip
        g_lip_y = float(lip_gain_y) if lip_gain_y is not None else g_lip
        g_corner = float(corner_gain)

        gains_xy = torch.ones((xy.shape[0], 2), dtype=exp_delta.dtype, device=exp_delta.device)
        eb = min(g_eye * g_brow, 1.0)
        gains_xy[eye_brow_mask, 0] *= eb
        gains_xy[eye_brow_mask, 1] *= eb
        gains_xy[lips_mask,   0]  *= g_lip_x
        gains_xy[lips_mask,   1]  *= g_lip_y
        gains_xy[corner_mask, 0]  *= g_corner
        gains_xy[corner_mask, 1]  *= g_corner

        # do NOT Kelvin-dampen lips/corners
        no_kelvin  = lips_mask | corner_mask
        kelvin_vec = torch.where(no_kelvin, torch.ones_like(kelvin_vec), kelvin_vec)

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
        exp_delta[0, :, 0:2] = exp_delta[0, :, 0:2] * kelvin_vec.unsqueeze(-1) * gains_xy

        # Mouth-friendly soft-knee compression
        V    = exp_delta[0]
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

        # Diagnostics + overlays
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
            print("  [viz] saved outputs/diagnostics/kp_masks.jpg")
        except Exception as e:
            print("  [viz] skip:", e)

        mouth_mask = lips_mask | corner_mask
        mouth_idx_list = torch.nonzero(mouth_mask, as_tuple=False).squeeze(-1).cpu().tolist()
        if len(mouth_idx_list) == 0:
            # absolute last-resort fallback: lower half + center-ish width
            mouth_mask = (yn > torch.quantile(yn, torch.tensor(0.50, device=yn.device))) & (xn.abs() < torch.quantile(xn.abs(), torch.tensor(0.85, device=xn.device)))
            mouth_idx_list = torch.nonzero(mouth_mask, as_tuple=False).squeeze(-1).cpu().tolist()

        draw_kp_map_simple(target_rgb, kp_can_t[:, :2], "outputs/diagnostics/target_kp_mouth_labeled.jpg", mouth_idxs=set(mouth_idx_list))
        print("[diag] mouth keypoint indices:", mouth_idx_list)

    # --- Vertical drift guard & top/bottom anchoring ---
    # Recompute yn using target keypoints (outside no_grad for clarity)
    xy_t = kp_can_t[:, :2].to(exp_delta.device, exp_delta.dtype)
    cx_t, cy_t = xy_t.mean(dim=0)
    dx_t = xy_t[:, 0] - cx_t
    dy_t = xy_t[:, 1] - cy_t
    sx_t = torch.std(dx_t) + 1e-6
    sy_t = torch.std(dy_t) + 1e-6
    yn = dy_t / sy_t
    xn_t = dx_t / sx_t

    lips_mask2 = (yn > -0.35) & (xn_t.abs() < 0.95)
    corner_mask2 = (yn > -0.20) & (xn_t.abs() >= 0.30) & (xn_t.abs() < 0.95)
    stable = ~(lips_mask2 | corner_mask2)

    dy = exp_delta[0, :, 1]
    if y_drift_fix != "none":
        if y_drift_fix == "nonmouth" and stable.any():
            dy2, _ = affine_detrend_dy(dy, kp_can_t_xy)  # detrend using all points' geometry
            # Compute the residual bias on stable landmarks ONLY
            bias_stable = dy2[stable].mean()
            # Apply bias to NON-MOUTH only; keep mouth Y free (or small fraction)
            exp_delta[0, stable, 1] = dy2[stable] - bias_stable
            mouth_mask_all = ~(stable)
            if float(y_drift_mouth_bias) != 0.0 and mouth_mask_all.any():
                exp_delta[0, mouth_mask_all, 1] = dy2[mouth_mask_all] - float(y_drift_mouth_bias) * bias_stable
            else:
                exp_delta[0, mouth_mask_all, 1] = dy2[mouth_mask_all]
        else:
            dy2, _ = affine_detrend_dy(dy, kp_can_t_xy)
            exp_delta[0, :, 1] = dy2

    # S4
    lips_mask_log = lips_mask2
    corner_mask_log = corner_mask2
    center_lip_mask_log = (xn_t.abs() < 0.18) & (yn > -0.05) & (yn < 0.45)
    log_stage("S4 (after y_drift_fix)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })

    # Anchor extremes to avoid vertical squash/stretch
    if float(y_anchor) > 0.0:
        yn_abs = yn.abs().clamp(0, 2.0)
        denom = yn_abs.max().clamp_min(1e-6)
        anchor = 1.0 - float(y_anchor) * (yn_abs / denom)
        # do NOT anchor lips/corners to preserve dip—apply only to non-mouth
        exp_delta[0, stable, 1] = exp_delta[0, stable, 1] * anchor[stable]

    # S5
    log_stage("S5 (after y_anchor)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })

    # kill residual global widening in X (affine detrend)
    dx = exp_delta[0, :, 0]
    dx, _coeff = affine_detrend_dx(dx, kp_can_t_xy)
    exp_delta[0, :, 0] = dx

    # S6
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

    # S7
    log_stage("S7 (after clamp_q)", exp_delta, {
        "lips_mask": lips_mask_log,
        "corner_mask": corner_mask_log,
        "center_lip_mask": center_lip_mask_log
    })

    # per-side mean removal so left/right don’t drift apart on average
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

    # D) Vector plot of current delta on mouth points only (after S3+ downstream edits)
    if mouth_mask is not None:
        try:
            draw_mouth_vectors(
                target_rgb,
                kp_can_t[:, :2],
                exp_delta[0, :, :].clamp(-1.0, 1.0),  # avoid NaNs/huge values
                mouth_mask,
                "outputs/diagnostics/mouth_vectors.jpg",
                scale_px=260.0  # bump a bit so short motions still show
            )
        except Exception as e:
            print("[viz] mouth_vectors skip:", e)

    # ---- global motion cap allowing amplification up to pre_norm * norm_cap
    post_norm = exp_delta.detach().float().norm().item()
    cap = float(pre_norm) * float(norm_cap)
    if (post_norm > cap) and (cap > 1e-8):
        exp_delta *= (cap / post_norm)

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

    # save diagnostics
    os.makedirs("outputs/diagnostics", exist_ok=True)
    out_w_res, out_h_res = resolve_output_size(target_rgb, out_w, out_h)
    img_base_lb = letterbox_to(img_base, out_w_res, out_h_res)
    img_asym_lb = letterbox_to(img_asym, out_w_res, out_h_res)
    donor_lb = letterbox_to(donor_rgb, out_w_res, out_h_res)

    side_img = np.hstack([img_base_lb, img_asym_lb])  # target before/after
    cv2.imwrite("outputs/diagnostics/baseline_vs_asym.jpg", cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR))

    side_dr = np.hstack([donor_lb, img_asym_lb])  # donor vs result
    cv2.imwrite("outputs/diagnostics/donor_vs_result.jpg", cv2.cvtColor(side_dr, cv2.COLOR_RGB2BGR))

    # final output
    save_rgb(out_path, img_asym_lb)
    print("[OK]", out_path)

    print("\n=== FINAL DELTA ===")
    print(f"Final delta magnitude: {exp_delta.abs().mean().item():.6f}")
    print(f"Pre-norm: {pre_norm:.4f}, Post-norm: {post_norm:.4f}, Cap: {cap:.4f}")

    # Final lip/corner stats and acceptance vs S1
    masks_fin = compute_masks(kp_can_t)
    lips_mask_fin = masks_fin["lips_mask"]
    corner_mask_fin = masks_fin["corner_mask"]
    if lips_mask_fin.any():
        lip_delta_y = exp_delta[0, lips_mask_fin, 1].abs()
        print(f"Lip Y |mean|={lip_delta_y.mean():.6f} |max|={lip_delta_y.max():.6f}")
    if corner_mask_fin.any():
        corner_delta_y = exp_delta[0, corner_mask_fin, 1].abs()
        print(f"Corner Y |mean|={corner_delta_y.mean():.6f} |max|={corner_delta_y.max():.6f}")

    def _region_ratio(base: torch.Tensor, now: torch.Tensor, m: torch.Tensor) -> float:
        if not m.any():
            return 0.0
        b = base[0, m, 1].abs().mean().item() + 1e-8
        n = now[0, m, 1].abs().mean().item()
        return float(n / b)

    r_corners = _region_ratio(S1, exp_delta, corner_mask_fin)
    print(f"[ACCEPTANCE] Corner Y retention vs S1: {r_corners*100:.1f}% (target ≥60–80%)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Left–Right asymmetry transfer (donor -> neutral target)")
    ap.add_argument("--donor","-d", required=True)
    ap.add_argument("--target","-t", required=True)
    ap.add_argument("--out","-o", default="outputs/asym_transfer.jpg")
    ap.add_argument("--config", default="src/config/models.yaml")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--edge_dampen", type=float, default=0.55)
    ap.add_argument("--out_w", type=int, default=0)
    ap.add_argument("--out_h", type=int, default=0)
    ap.add_argument("--asym_side", type=str, default="right", choices=["right","left"])
    ap.add_argument("--midline_eps", type=float, default=0.05)
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
    ap.add_argument("--norm_cap", type=float, default=2.5, help="Allow total motion up to pre_norm * norm_cap (default 2.5). Use 1.0 to keep old behavior.")
    # Pose alignment
    ap.add_argument("--auto_flip", action="store_true")
    # --- NEW controls ---
    ap.add_argument("--mouth_sym_alpha", type=float, default=0.0, help="Blend of donor symmetric Y motion on lips/corners (0..1). 0 = off.")
    ap.add_argument("--y_drift_fix", type=str, default="nonmouth", choices=["none", "global", "nonmouth"], help="Remove vertical bias: 'global' removes overall mean; 'nonmouth' uses non-mouth landmarks.")
    ap.add_argument("--y_anchor", type=float, default=0.5, help="0..1 damping toward very top/bottom landmarks to avoid vertical squash. 0=off.")
    ap.add_argument("--y_drift_mouth_bias", type=float, default=0.0, help="Fraction of nonmouth vertical bias to subtract from mouth Y (0=do not subtract).")

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
    )
