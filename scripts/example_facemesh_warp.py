"""
Example: FaceMesh warp pipeline (Phase 3-5).

Demonstrates applying asymmetry-based warping to an image using:
1. FaceMesh landmark extraction
2. Donor asymmetry computation (Phase 1-2)
3. Post-process warp via TPS (Phase 3-5)
4. Validation and debug outputs

Usage:
  python example_facemesh_warp.py --donor donor.jpg --target target.jpg --output-dir outputs
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.facemesh_landmarks import (
    FaceMeshLandmarkExtractor,
    compute_donor_asymmetry_delta,
    load_facemesh_regions_config,
)
from scripts.facemesh_warp import (
    apply_facemesh_warp,
    validate_warp,
)


def read_rgb(path):
    """Read image as RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, img_rgb):
    """Save image as RGB."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def run_warp_pipeline(
    donor_path,
    target_after_path,
    output_dir="outputs",
    alpha=1.0,
    reg=1e-3,
    grid_step=2,
    lock_boundary=True,
    validate=True,
    verbose=True,
):
    """
    Apply full FaceMesh warp pipeline.
    
    Args:
        donor_path: Path to donor image (with asymmetry)
        target_after_path: Path to target image (LivePortrait output or input)
        output_dir: Directory for outputs
        alpha: Warp strength (0-1)
        reg: TPS regularization
        grid_step: TPS grid step
        lock_boundary: Lock image boundaries
        validate: Run validation
        verbose: Print progress
    
    Returns:
        Dictionary with results and stats
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print("FaceMesh Warp Pipeline Example")
        print(f"{'='*70}")
        print(f"Donor:  {donor_path}")
        print(f"Target: {target_after_path}")
        print(f"Output: {output_dir}")
        print(f"Params: alpha={alpha}, reg={reg}, grid_step={grid_step}")
    
    # Load images
    if verbose:
        print("\n[1/6] Loading images...")
    
    donor_rgb = read_rgb(donor_path)
    target_rgb = read_rgb(target_after_path)
    
    H, W = target_rgb.shape[:2]
    if verbose:
        print(f"  Donor shape: {donor_rgb.shape}")
        print(f"  Target shape: {target_rgb.shape}")
    
    # Initialize FaceMesh extractor
    if verbose:
        print("\n[2/6] Initializing FaceMesh...")
    
    extractor = FaceMeshLandmarkExtractor(
        static_image_mode=True,
        max_num_faces=1,
        verbose=verbose,
        refine_landmarks=False
    )
    
    # Compute donor asymmetry
    if verbose:
        print("\n[3/6] Computing donor asymmetry...")
    
    regions_config = load_facemesh_regions_config(None)
    
    facemesh_result = compute_donor_asymmetry_delta(
        donor_rgb,
        extractor,
        regions_config,
        apply_bias_removal=True,
        max_delta_px=None,
        clamp_percentile=98.0,
        verbose=verbose
    )
    
    if not facemesh_result.get("ok", False):
        print("✗ Failed to compute donor asymmetry")
        return {"ok": False}
    
    if verbose:
        print(f"  ✓ Asymmetry computed")
        print(f"    L_d shape: {facemesh_result['L_d'].shape}")
        print(f"    delta shape: {facemesh_result['delta'].shape}")
    
    # Extract landmarks on target
    if verbose:
        print("\n[4/6] Extracting landmarks on target image...")
    
    ok_target, L_out, L_out_vis = extractor.extract(target_rgb)
    
    if not ok_target:
        print("✗ Failed to extract landmarks on target")
        return {"ok": False}
    
    if verbose:
        print(f"  ✓ Landmarks extracted")
        print(f"    L_out shape: {L_out.shape}")
    
    # Apply warp
    if verbose:
        print("\n[5/6] Applying TPS warp...")
    
    warp_output_dir = os.path.join(output_dir, "warp_diagnostics")
    
    ok_warp, img_warped, warp_summary = apply_facemesh_warp(
        target_rgb,
        facemesh_result["L_d"],
        facemesh_result["delta"],
        facemesh_result["weights"],
        L_out,
        L_out_vis=L_out_vis,
        alpha=alpha,
        reg=reg,
        grid_step=grid_step,
        lock_boundary=lock_boundary,
        verbose=verbose,
        output_dir=warp_output_dir
    )
    
    if not ok_warp or img_warped is None:
        print("✗ Warp failed")
        return {"ok": False}
    
    if verbose:
        print(f"  ✓ Warp succeeded")
        print(f"    Fold fraction: {warp_summary.get('fold_fraction', 0)*100:.2f}%")
    
    # Validation
    validation = {"success": False}
    if validate:
        if verbose:
            print("\n[6/6] Running validation...")
        
        # Compute target landmarks
        L_out_target = L_out.copy()
        delta_out = facemesh_result["delta"] @ warp_summary.get("sR", np.eye(2))
        sel_idx = np.where(facemesh_result["weights"] > 0)[0]
        for idx in sel_idx:
            if 0 <= idx < 468:
                L_out_target[idx] = L_out[idx] + alpha * delta_out[idx]
        
        validation = validate_warp(
            img_warped,
            L_out_target,
            sel_idx,
            extractor,
            output_dir=warp_output_dir,
            verbose=verbose
        )
    else:
        if verbose:
            print("\n[6/6] Skipping validation (disabled)")
    
    # Save outputs
    if verbose:
        print("\n[7/7] Saving outputs...")
    
    # Main output
    save_rgb(os.path.join(output_dir, "warped.png"), img_warped)
    
    # Summary
    summary = {
        "ok": True,
        "donor_shape": donor_rgb.shape,
        "target_shape": target_rgb.shape,
        "warped_shape": img_warped.shape,
        "parameters": {
            "alpha": alpha,
            "reg": reg,
            "grid_step": grid_step,
            "lock_boundary": lock_boundary,
        },
        "warp": {
            "fold_fraction": warp_summary.get("fold_fraction", 0),
            "num_control_points": warp_summary.get("num_control_points", 0),
        },
        "validation": validation,
        "output_dir": warp_output_dir,
    }
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, "warp_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\n{'='*70}")
        print("Pipeline complete!")
        print(f"  Output: {os.path.join(output_dir, 'warped.png')}")
        print(f"  Diagnostics: {warp_output_dir}")
        print(f"  Summary: {summary_path}")
        print(f"  Fold fraction: {summary['warp']['fold_fraction']*100:.2f}%")
        if validation["success"]:
            print(f"  Validation error: {validation['mean_error_px']:.2f}px (mean), "
                  f"{validation['max_error_px']:.2f}px (max)")
        print(f"{'='*70}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="FaceMesh warp pipeline example"
    )
    parser.add_argument("--donor", required=True, help="Path to donor image")
    parser.add_argument("--target", required=True, help="Path to target image")
    parser.add_argument("--output-dir", default="outputs/warp_example",
                        help="Output directory")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Warp strength (0-1)")
    parser.add_argument("--reg", type=float, default=1e-3,
                        help="TPS regularization")
    parser.add_argument("--grid-step", type=int, default=2,
                        help="TPS coarse-to-fine step")
    parser.add_argument("--no-lock-boundary", action="store_true",
                        help="Disable boundary locking")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    result = run_warp_pipeline(
        donor_path=args.donor,
        target_after_path=args.target,
        output_dir=args.output_dir,
        alpha=args.alpha,
        reg=args.reg,
        grid_step=args.grid_step,
        lock_boundary=not args.no_lock_boundary,
        validate=not args.no_validate,
        verbose=not args.quiet,
    )
    
    if not result.get("ok", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
