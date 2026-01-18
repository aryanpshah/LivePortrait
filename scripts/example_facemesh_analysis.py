#!/usr/bin/env python3
"""
Example: Using FaceMesh for Donor Asymmetry Analysis

This example shows how to use the FaceMesh module to compute and visualize
facial asymmetry signals, with full debugging outputs.

Prerequisites:
  - pip install mediapipe opencv-python numpy

Usage:
  python scripts/example_facemesh_analysis.py --donor path/to/donor.jpg --target path/to/target.jpg
"""

import argparse
import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.facemesh_landmarks import (
    FaceMeshLandmarkExtractor,
    compute_donor_asymmetry_delta,
    load_facemesh_regions_config,
    draw_facemesh_overlay,
    draw_delta_heatmap,
    draw_comparison_overlay,
    save_facemesh_numpy_dumps,
    save_facemesh_summary,
    FACEMESH_LIPS_IDX,
    FACEMESH_FACE_OVAL_IDX,
    FACEMESH_ANCHOR_IDX,
)

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """Read image as RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze facial asymmetry using MediaPipe FaceMesh"
    )
    parser.add_argument("--donor", required=True, help="Path to donor image")
    parser.add_argument("--target", required=True, help="Path to target image")
    parser.add_argument("--output-dir", default="outputs/facemesh_analysis",
                        help="Output directory for visualizations")
    parser.add_argument("--max-delta-px", type=float, default=None,
                        help="Absolute max delta magnitude (pixels)")
    parser.add_argument("--clamp-percentile", type=float, default=98.0,
                        help="Percentile for auto-clamping")
    parser.add_argument("--cheek-radius-px", type=float, default=None,
                        help="Override cheek patch radius")
    parser.add_argument("--no-debug", action="store_true",
                        help="Skip debug visualizations")
    parser.add_argument("--refine", action="store_true",
                        help="Use refined FaceMesh landmarks")

    args = parser.parse_args()

    print("=" * 70)
    print("FaceMesh Asymmetry Analysis Example")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read images
    print("\n[1/4] Loading images...")
    try:
        donor_rgb = read_image(args.donor)
        target_rgb = read_image(args.target)
        print(f"  ✓ Donor: {donor_rgb.shape}")
        print(f"  ✓ Target: {target_rgb.shape}")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return 1

    # Initialize extractor
    print("\n[2/4] Initializing FaceMesh extractor...")
    extractor = FaceMeshLandmarkExtractor(
        static_image_mode=True,
        max_num_faces=1,
        verbose=True,
        refine_landmarks=args.refine
    )

    # Load region configuration
    print("\n[3/4] Loading region configuration...")
    config = load_facemesh_regions_config(None)
    print(f"  ✓ Loaded default configuration")
    print(f"    - lips: {len(config['lips'])} landmarks")
    print(f"    - face_oval: {len(config['face_oval'])} landmarks")
    print(f"    - anchors: {len(config['anchors'])} landmarks")
    print(f"    - cheek_seeds: {config['cheek_seeds']}")

    # Compute asymmetry
    print("\n[4/4] Computing asymmetry delta...")
    result = compute_donor_asymmetry_delta(
        donor_rgb,
        extractor,
        config,
        apply_bias_removal=True,
        max_delta_px=args.max_delta_px,
        clamp_percentile=args.clamp_percentile,
        cheek_radius_px=args.cheek_radius_px,
        verbose=True
    )

    if not result.get("ok", False):
        print("✗ Asymmetry computation failed")
        return 1

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Print summary statistics
    summary = result.get("summary", {})
    print(f"\nImage shape: {summary.get('donor_image_shape')}")
    print(f"Landmarks detected: 468")
    print(f"Bias removal: {summary.get('bias_removed')}")
    if summary.get("bias_vector") is not None:
        bias = summary["bias_vector"]
        print(f"  Bias vector: ({bias[0]:.4f}, {bias[1]:.4f}) px")

    print(f"\nAsymmetry magnitudes:")
    print(f"  Mean |Δ|: {summary.get('overall_mean_mag', 0):.4f} px")
    print(f"  Max |Δ|:  {summary.get('overall_max_mag', 0):.4f} px")
    print(f"  Min |Δ|:  {summary.get('overall_min_mag', 0):.4f} px")

    clamp = result.get("clamp_params", {})
    print(f"\nClamping:")
    print(f"  Method: {clamp.get('method')}")
    print(f"  Max magnitude: {clamp.get('max_px'):.4f} px")

    print(f"\nPer-region statistics:")
    regions = result.get("regions", {})
    for region_name, stats in regions.items():
        if stats.get("count", 0) > 0:
            print(f"  {region_name}:")
            print(f"    Mean |Δ|: {stats.get('mean_mag', 0):.4f} px")
            print(f"    Max |Δ|:  {stats.get('max_mag', 0):.4f} px")
            print(f"    Count: {stats.get('count')}")

    # Generate debug visualizations
    if not args.no_debug:
        print("\n" + "=" * 70)
        print("GENERATING DEBUG VISUALIZATIONS")
        print("=" * 70)

        # Landmark overlays
        print("\n  Drawing landmark overlays...")
        groups = {
            "lips": config["lips"],
            "face_oval": config["face_oval"],
            "cheek_patch": result.get("cheek_patch", []),
            "anchors": config["anchors"],
        }

        draw_facemesh_overlay(
            donor_rgb,
            result["L_d"],
            groups,
            os.path.join(args.output_dir, "donor_landmarks_all.png"),
            label_some=True
        )
        print(f"    ✓ {os.path.join(args.output_dir, 'donor_landmarks_all.png')}")

        # Flipped donor
        donor_flip = donor_rgb[:, ::-1, :].copy()
        draw_facemesh_overlay(
            donor_flip,
            result["L_f"],
            groups,
            os.path.join(args.output_dir, "donor_flip_landmarks.png"),
            label_some=False
        )
        print(f"    ✓ {os.path.join(args.output_dir, 'donor_flip_landmarks.png')}")

        # Comparison overlay
        draw_comparison_overlay(
            donor_rgb,
            result["L_d"],
            result["L_f_back"],
            os.path.join(args.output_dir, "flip_back_comparison.png"),
            label_1="original",
            label_2="flip_mirrored_back"
        )
        print(f"    ✓ {os.path.join(args.output_dir, 'flip_back_comparison.png')}")

        # Delta heatmap
        print("  Drawing delta heatmap...")
        roi_union = set(
            config["lips"] +
            config["face_oval"] +
            result.get("cheek_patch", [])
        )
        roi_union = sorted(list(roi_union))

        draw_delta_heatmap(
            donor_rgb,
            result["L_d"],
            result["delta"],
            roi_union,
            os.path.join(args.output_dir, "delta_vectors.png"),
            scale_px=320.0
        )
        print(f"    ✓ {os.path.join(args.output_dir, 'delta_vectors.png')}")

        # Save numpy arrays and summary
        print("  Saving numpy arrays...")
        save_facemesh_numpy_dumps(result, args.output_dir)
        save_facemesh_summary(result, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print(f"✓ Analysis complete. Outputs saved to: {args.output_dir}")
    print("=" * 70)

    extractor.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
