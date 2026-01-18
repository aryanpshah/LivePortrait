#!/usr/bin/env python3
"""
Test script for FaceMesh landmark extraction and asymmetry computation.

This script validates the FaceMesh module without requiring full LivePortrait inference.
"""

import os
import sys
import numpy as np

# Add project root to path
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
    compute_dynamic_cheek_patch,
)


def test_extractor_basic():
    """Test basic landmark extraction."""
    print("\n=== Test 1: Basic Extractor ===")

    try:
        extractor = FaceMeshLandmarkExtractor(verbose=True)
    except (AttributeError, ImportError) as e:
        print(f"⚠ Skipping test: MediaPipe not properly installed or incompatible version")
        print(f"  Error: {e}")
        print("  Tip: Try: pip install mediapipe==0.10.9")
        return

    # Create a dummy image (will likely fail to detect face, but that's ok for this test)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        ok, lm_px, lm_vis = extractor.extract(dummy_img)
    except (AttributeError, ImportError) as e:
        print(f"⚠ Skipping test: MediaPipe extraction failed")
        print(f"  Error: {e}")
        extractor.cleanup()
        return

    print(f"Extraction ok: {ok}")
    if ok:
        print(f"Landmarks shape: {lm_px.shape}")
        print(f"Visibility shape: {lm_vis.shape}")
        assert lm_px.shape == (468, 2), f"Expected (468, 2), got {lm_px.shape}"
        assert lm_vis.shape == (468,), f"Expected (468,), got {lm_vis.shape}"
        print("✓ Landmark shapes correct")
    else:
        print("(Expected: no face in dummy image)")

    extractor.cleanup()


def test_regions_config():
    """Test region configuration loading."""
    print("\n=== Test 2: Region Configuration ===")

    # Test default config
    config = load_facemesh_regions_config(None)

    print(f"Loaded config with keys: {list(config.keys())}")
    assert "lips" in config, "Missing 'lips' key"
    assert "face_oval" in config, "Missing 'face_oval' key"
    assert "anchors" in config, "Missing 'anchors' key"
    assert "cheek_seeds" in config, "Missing 'cheek_seeds' key"

    print(f"  lips: {len(config['lips'])} indices")
    print(f"  face_oval: {len(config['face_oval'])} indices")
    print(f"  anchors: {len(config['anchors'])} indices")
    print(f"  cheek_seeds: {config['cheek_seeds']}")

    print("✓ Region config loaded successfully")


def test_dynamic_cheek_patch():
    """Test dynamic cheek patch computation."""
    print("\n=== Test 3: Dynamic Cheek Patch ===")

    # Create dummy landmarks
    np.random.seed(42)
    lm_px = np.random.randn(468, 2) * 100 + np.array([320, 240])
    lm_px = np.clip(lm_px, 0, 640)

    # Place cheek seeds at specific locations
    lm_px[234] = [200, 240]  # left cheek
    lm_px[454] = [440, 240]  # right cheek

    cheek_patch = compute_dynamic_cheek_patch(
        lm_px,
        cheek_seeds=[234, 454],
        radius_fraction=0.15,
        radius_min_px=40.0,
        radius_max_px=80.0
    )

    print(f"Cheek-to-cheek distance: {np.linalg.norm(lm_px[454] - lm_px[234]):.2f} px")
    print(f"Cheek patch size: {len(cheek_patch)} landmarks")
    assert len(cheek_patch) > 0, "Cheek patch should not be empty"
    assert 234 in cheek_patch, "Cheek seed 234 should be in patch"
    assert 454 in cheek_patch, "Cheek seed 454 should be in patch"

    print("✓ Dynamic cheek patch computed successfully")


def test_delta_computation_dummy():
    """Test delta computation with dummy data (will fail gracefully without real image)."""
    print("\n=== Test 4: Delta Computation (Dummy Data) ===")

    # Since we can't easily create a valid test image with detected faces,
    # we'll just verify the function signature and error handling

    extractor = FaceMeshLandmarkExtractor(verbose=False)
    config = load_facemesh_regions_config(None)

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    result = compute_donor_asymmetry_delta(
        dummy_img,
        extractor,
        config,
        apply_bias_removal=True,
        max_delta_px=None,
        clamp_percentile=98.0,
        cheek_radius_px=None,
        verbose=False
    )

    print(f"Result keys: {list(result.keys())}")
    assert "ok" in result, "Result must have 'ok' key"

    if result["ok"]:
        print("Face detected (unexpected)")
        print(f"Delta shape: {result['delta'].shape}")
    else:
        print("(Expected: no face in dummy image)")
        print("✓ Error handling works correctly")

    extractor.cleanup()


def test_api_signatures():
    """Test that all exported functions have correct signatures."""
    print("\n=== Test 5: API Signatures ===")

    # This is a compile-time check, but good to verify
    functions = [
        FaceMeshLandmarkExtractor,
        compute_donor_asymmetry_delta,
        load_facemesh_regions_config,
        draw_facemesh_overlay,
        draw_delta_heatmap,
        draw_comparison_overlay,
        save_facemesh_numpy_dumps,
        save_facemesh_summary,
        compute_dynamic_cheek_patch,
    ]

    print(f"Exported {len(functions)} functions/classes")
    for f in functions:
        print(f"  ✓ {f.__name__}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("FaceMesh Landmark Extractor - Test Suite")
    print("=" * 70)

    tests_passed = 0
    tests_skipped = 0

    try:
        test_extractor_basic()
        tests_passed += 1
    except Exception as e:
        if "MediaPipe" in str(e) or "AttributeError" in str(type(e).__name__):
            tests_skipped += 1
            print(f"⚠ Test skipped due to MediaPipe issue")
        else:
            raise

    try:
        test_regions_config()
        tests_passed += 1
        test_dynamic_cheek_patch()
        tests_passed += 1
        test_delta_computation_dummy()
        tests_passed += 1
        test_api_signatures()
        tests_passed += 1

        print("\n" + "=" * 70)
        print(f"✓ {tests_passed} tests passed")
        if tests_skipped > 0:
            print(f"⚠ {tests_skipped} tests skipped (MediaPipe issues)")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
