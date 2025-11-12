import sys, os, os.path as osp, cv2, torch
import numpy as np

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
    top  = (out_h - new_h) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

def resolve_output_size(target_rgb: np.ndarray, out_w: int, out_h: int):
    """
    Decide final output size
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
    """
    Compute the best-fit similarity *scale* that maps src->dst (least-squares)
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
    if torch.det(R) < 0:             # handle reflections to keep a proper rotation
        Vt[-1, :] *= -1
        R = U @ Vt
    theta = float(torch.atan2(R[1, 0], R[0, 0]).item())
    return R, theta

def affine_detrend_dx(dx: torch.Tensor, kp_xy: torch.Tensor):
    # Remove global widening from X motion by fitting: dx ≈ a*(x-xc) + b*(y-yc) + c and subtracting it off
    x = kp_xy[:, 0]; y = kp_xy[:, 1]
    xc = x.mean(); yc = y.mean()
    X = torch.stack([x - xc, y - yc, torch.ones_like(x)], dim=1)  # (N,3)
    coeff = torch.linalg.pinv(X) @ dx                              # (3,)
    trend = X @ coeff                                              # (N,)
    return dx - trend, coeff

def soft_knee_vec(V: torch.Tensor, tau: float) -> torch.Tensor:
    # Compress vector magnitudes to avoid extreme values
    # m' = tau * tanh(m / tau)  -> behaves linear for small m, saturates for big m
    m = torch.linalg.norm(V, dim=-1, keepdim=True) + 1e-8 # per keypoint motion magnitude
    m_prime = float(tau) * torch.tanh(m / float(tau))
    gain = (m_prime / m).clamp(max=1.0)  # dont amplify
    return V * gain

def soft_knee_scalar(d: torch.Tensor, tau: float) -> torch.Tensor:
    # Scalar version for X only
    a = d.abs() + 1e-8
    a_prime = float(tau) * torch.tanh(a / float(tau))
    return torch.sign(d) * a_prime


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

    if np.all(np.isfinite(xy)) and np.median(np.abs(xy)) <= 1.2:
        px = (x + 1.0) * 0.5 * (width - 1)
        py = (y + 1.0) * 0.5 * (height - 1)
    else:
        px, py = x, y

    pts = np.stack([px, py], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts


def draw_keypoint_overlay(
    img_rgb: np.ndarray,
    kp_xy: torch.Tensor,
    model_shape: tuple[int, int],
    output_path: str,
) -> None:
    """
    Draw projected keypoints on the original image so diagnostics align with the face.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    canvas = img_rgb.copy()
    if isinstance(kp_xy, torch.Tensor):
        pts = kp_xy.detach().cpu()
        pts = pts[:, :2]
    else:
        pts = np.asarray(kp_xy, dtype=np.float32)
    model_h, model_w = model_shape
    pts_model = kp_to_pixels(pts, model_h, model_w)

    scale_x = img_rgb.shape[1] / float(model_w)
    scale_y = img_rgb.shape[0] / float(model_h)
    pts_img = pts_model.copy()
    pts_img[:, 0] = np.clip(pts_img[:, 0] * scale_x, 0, img_rgb.shape[1] - 1)
    pts_img[:, 1] = np.clip(pts_img[:, 1] * scale_y, 0, img_rgb.shape[0] - 1)

    pts_int = np.round(pts_img).astype(np.int32)
    for u, v in pts_int:
        cv2.circle(canvas, (int(u), int(v)), 3, (64, 224, 255), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def get_exp_tensor(wrap: LivePortraitWrapper, img_rgb: np.ndarray):
    # run motion extractor and pull the expression tensor (B,N,3)
    T = wrap.prepare_source(img_rgb)
    kp = wrap.get_kp_info(T, flag_refine_info=True)
    return kp["exp"]

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
):
    # load models
    cfg = InferenceConfig(models_config=cfg_path)
    wrap = LivePortraitWrapper(cfg)

    # read donor (has asymmetry) and target (neutral avatar)
    donor_rgb  = read_rgb(donor_path)
    target_rgb = read_rgb(target_path)

    # get target keypoints in canonical coords to define left/right etc
    T = wrap.prepare_source(target_rgb)
    T_kp = wrap.get_kp_info(T, flag_refine_info=True)
    kp_can_t = T_kp["kp"][0]         # (N,3)
    x_can = kp_can_t[:, 0]
    x_center = x_can.mean()
    left_mask  = x_can < x_center
    right_mask = ~left_mask

    # donor LR delta via flip and map back
    # If E_donor = S+A (symmetric + asymmetric), then
    # flipping horizontally and mapping back gives S-A
    # Subtract them to isolate 2*A (the LR difference)
    E_donor    = get_exp_tensor(wrap, donor_rgb)
    donor_flip = donor_rgb[:, ::-1, :].copy()
    E_flip     = get_exp_tensor(wrap, donor_flip)
    E_flip_map = E_flip.clone(); E_flip_map[..., 0] *= -1

    if asym_side.lower() == "right":
        exp_delta = (E_donor - E_flip_map) * float(scale)
    elif asym_side.lower() == "left":
        exp_delta = (E_flip_map - E_donor) * float(scale)
    else:
        raise ValueError("asym_side must be 'right' or 'left'")

    pre_norm = exp_delta.detach().float().norm().item()

    # normalize donor vs target pose/size in canonical space
    D = wrap.prepare_source(donor_rgb)
    D_kp = wrap.get_kp_info(D, flag_refine_info=True)
    kp_can_d = D_kp["kp"][0]
    kp_can_d_xy = kp_can_d[:, :2]
    kp_can_t_xy = kp_can_t[:, :2]

    # scale (zoom) match
    s_ratio = procrustes_scale(kp_can_d_xy, kp_can_t_xy)
    exp_delta[0, :, 0:2] *= s_ratio

    # in-plane rotation (roll) match
    R, _theta = similarity_decompose(kp_can_d_xy, kp_can_t_xy)
    R = R.to(dtype=exp_delta.dtype, device=exp_delta.device)
    exp_delta[0, :, 0:2] = exp_delta[0, :, 0:2] @ R

    """
    Keep motion concentrated around the middle of the face
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

    # central feature boost
    # this makes asymmetry more visible on eyes/nose/mouth without puffing cheeks
    with torch.no_grad():
        xy = kp_can_t[:, :2]
        cx, cy = xy.mean(dim=0)
        dxn = xy[:, 0] - cx
        dyn = xy[:, 1] - cy
        sx = torch.std(dxn) + 1e-6
        sy = torch.std(dyn) + 1e-6
        bf_x = 1.0 * sx
        bf_y = 1.0 * sy
        w_focus = torch.exp(-0.5 * ((dxn / bf_x) ** 2 + (dyn / bf_y) ** 2))  # (N,)
        boost = (1.0 + 0.50 * w_focus).to(dtype=exp_delta.dtype, device=exp_delta.device).unsqueeze(-1)
        exp_delta[0, :, 0:2] = exp_delta[0, :, 0:2] * boost

        # soft-knee compression so a few extreme keypoints don’t dominate
        V = exp_delta[0]
        mags = torch.linalg.norm(V, dim=-1)
        tau_v = float(torch.quantile(mags, torch.tensor(0.95, device=mags.device))) * 1.20
        V = soft_knee_vec(V, tau_v)
        dx = V[:, 0]
        tau_x = float(torch.quantile(dx.abs(), torch.tensor(0.95, device=dx.device))) * 1.05
        V[:, 0] = soft_knee_scalar(dx, tau_x)
        exp_delta[0] = V

    # kill residual global widening in X (affine detrend)
    dx = exp_delta[0, :, 0]
    dx, _coeff = affine_detrend_dx(dx, kp_can_t_xy)
    exp_delta[0, :, 0] = dx

    # clamp extreme X motions
    if clamp_q > 0.0:
        dx = exp_delta[0, :, 0]
        q = torch.quantile(dx.abs(), torch.tensor(float(clamp_q), device=dx.device))
        exp_delta[0, :, 0] = dx.clamp(min=-q, max=q)

    # freeze a small band around the midline (nose area stays stable)
    if midline_eps > 0:
        span_x = (x_can - x_center).abs().max() + 1e-6
        mid_mask = (x_can - x_center).abs() < (float(midline_eps) * span_x)
        exp_delta[0, mid_mask, 0] = 0.0

    # per-side mean removal so left/right don’t drift apart on average
    dx = exp_delta[0, :, 0]
    if (left_mask.any()):
        dx[left_mask]  = dx[left_mask]  - dx[left_mask].mean()
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

    # doesnt let the overall magnitude exceed the pre-normalized amount
    post_norm = exp_delta.detach().float().norm().item()
    if (post_norm > pre_norm) and (pre_norm > 1e-8):
        exp_delta *= (pre_norm / post_norm)

    # inject delta into target keypoints and render
    T_kp_mod = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in T_kp.items()}
    T_kp_mod["exp"] = T_kp_mod["exp"] + exp_delta
    kp_source  = wrap.transform_keypoint(T_kp)
    kp_driving = wrap.transform_keypoint(T_kp_mod)
    kp_donor_vis = wrap.transform_keypoint(D_kp)

    feat_3d  = wrap.extract_feature_3d(T)
    out_base = wrap.warp_decode(feat_3d, kp_source,  kp_source)
    img_base = wrap.parse_output(out_base["out"])[0]
    out_asym = wrap.warp_decode(feat_3d, kp_source,  kp_driving)
    img_asym = wrap.parse_output(out_asym["out"])[0]

    # save diagnostics
    diag_dir = "outputs/diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    model_shape = wrap.inference_cfg.input_shape
    try:
        draw_keypoint_overlay(target_rgb, kp_source[0], model_shape, osp.join(diag_dir, "target_kp_overlay.jpg"))
        draw_keypoint_overlay(donor_rgb, kp_donor_vis[0], model_shape, osp.join(diag_dir, "donor_kp_overlay.jpg"))
    except Exception as e:
        print("[viz] keypoint overlay skip:", e)
    out_w_res, out_h_res = resolve_output_size(target_rgb, out_w, out_h)
    img_base_lb = letterbox_to(img_base, out_w_res, out_h_res)
    img_asym_lb = letterbox_to(img_asym, out_w_res, out_h_res)
    donor_lb    = letterbox_to(donor_rgb, out_w_res, out_h_res)

    side = np.hstack([img_base_lb, img_asym_lb])                   # target before/after
    cv2.imwrite(osp.join(diag_dir, "baseline_vs_asym.jpg"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    side_dr = np.hstack([donor_lb, img_asym_lb])                   # donor vs result
    cv2.imwrite(osp.join(diag_dir, "donor_vs_result.jpg"), cv2.cvtColor(side_dr, cv2.COLOR_RGB2BGR))

    # final output
    save_rgb(out_path, img_asym_lb)
    print("[OK]", out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Left–Right asymmetry transfer (donor -> neutral target)")
    ap.add_argument("--donor","-d", required=True)
    ap.add_argument("--target","-t", required=True)
    ap.add_argument("--out","-o", default="outputs/asym_transfer.jpg")
    ap.add_argument("--config", default="src/config/models.yaml")

    ap.add_argument("--scale",  type=float, default=1.0)
    ap.add_argument("--edge_dampen", type=float, default=0.55)
    ap.add_argument("--out_w", type=int, default=0)
    ap.add_argument("--out_h", type=int, default=0)
    ap.add_argument("--asym_side", type=str, default="right", choices=["right","left"])
    ap.add_argument("--midline_eps", type=float, default=0.05)
    ap.add_argument("--clamp_q", type=float, default=0.90)

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
    )
