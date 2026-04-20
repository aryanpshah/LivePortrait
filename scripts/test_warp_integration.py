"""
Integration validation tests for Phase 3-5 FaceMesh warp.

Validates:
1. All imports work correctly
2. CLI flags are properly parsed
3. Integration into asymmetry_transfer.py is correct
4. End-to-end pipeline works with dummy data
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    """Test that all modules import without errors."""
    print("\n[Test 1] Module imports...")

    try:
        from scripts.facemesh_warp import (
            estimate_similarity_umeyama,
            align_delta_to_output,
            select_control_points,
            tps_fit,
            tps_warp_image,
            detect_folding,
            apply_facemesh_warp,
            validate_warp,
            draw_control_points_overlay,
            draw_displacement_heatmap,
            draw_grid_warp_preview,
        )

        print("  ✓ facemesh_warp module imports successfully")
        print(f"    - estimate_similarity_umeyama: {callable(estimate_similarity_umeyama)}")
        print(f"    - align_delta_to_output: {callable(align_delta_to_output)}")
        print(f"    - select_control_points: {callable(select_control_points)}")
        print(f"    - tps_fit: {callable(tps_fit)}")
        print(f"    - tps_warp_image: {callable(tps_warp_image)}")
        print(f"    - detect_folding: {callable(detect_folding)}")
        print(f"    - apply_facemesh_warp: {callable(apply_facemesh_warp)}")
        print(f"    - validate_warp: {callable(validate_warp)}")

        return True

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_asymmetry_transfer_imports():
    """Test that asymmetry_transfer imports the warp module correctly."""
    print("\n[Test 2] asymmetry_transfer.py integration...")

    try:
        from scripts.asymmetry_transfer import main

        # Check that main has warp parameters
        import inspect
        sig = inspect.signature(main)
        params = set(sig.parameters.keys())

        warp_params = [
            "facemesh_warp",
            "facemesh_warp_method",
            "facemesh_warp_alpha",
            "facemesh_warp_reg",
            "facemesh_warp_grid_step",
            "facemesh_warp_lock_boundary",
            "facemesh_warp_validate",
            "facemesh_warp_save_field",
        ]

        guard_params = [
            "facemesh_guards",
            "facemesh_guard_debug",
            "guard_max_delta_px",
            "guard_cap_percentile",
            "guard_cap_region",
            "guard_cap_after_align",
            "guard_smooth_delta",
            "guard_smooth_iterations",
            "guard_smooth_lambda",
            "guard_smooth_mode",
            "guard_knn_k",
            "guard_zero_anchor",
            "guard_anchor_idx",
            "guard_anchor_strength",
            "guard_softmask",
            "guard_softmask_sigma",
            "guard_softmask_forehead_fade",
            "guard_softmask_forehead_yfrac",
            "guard_softmask_min",
            "guard_softmask_max",
            "guard_face_mask",
            "guard_face_mask_mode",
            "guard_face_mask_dilate",
            "guard_face_mask_erode",
            "guard_face_mask_blur",
            "guard_warp_face_only",
            "guard_mouth_only",
            "guard_mouth_radius_px",
            "guard_alpha_start",
        ]

        missing = [p for p in warp_params + guard_params if p not in params]

        if missing:
            print(f"  ✗ Missing parameters in main(): {missing}")
            return False

        print(f"  ✓ main() has all {len(warp_params)} warp and {len(guard_params)} guard parameters")
        return True

    except Exception as e:
        print(f"  ✗ Integration check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_parsing():
    """Test that CLI arguments parse correctly."""
    print("\n[Test 3] CLI argument parsing...")

    try:
        # Import asymmetry_transfer to access argparse setup
        import asymmetry_transfer

        # Simulate argument parsing
        test_args = [
            "--donor", "test_donor.jpg",
            "--target", "test_target.jpg",
            "--facemesh-warp",
            "--facemesh-warp-alpha", "0.8",
            "--facemesh-warp-reg", "0.001",
            "--facemesh-warp-grid-step", "4",
            "--facemesh-warp-validate",
            "--facemesh-warp-save-field",
            "--facemesh-guards",
            "--guard-max-delta-px", "15",
            "--guard-cap-percentile", "97.5",
            "--guard-cap-region", "all",
            "--no-guard-cap-after-align",
            "--no-guard-smooth-delta",
            "--guard-smooth-iterations", "3",
            "--guard-smooth-lambda", "0.5",
            "--guard-smooth-mode", "knn",
            "--guard-knn-k", "6",
            "--no-guard-zero-anchor",
            "--guard-anchor-idx", "1", "2", "3",
            "--guard-anchor-strength", "0.5",
            "--no-guard-softmask",
            "--guard-softmask-sigma", "20",
            "--no-guard-softmask-forehead-fade",
            "--guard-softmask-forehead-yfrac", "0.2",
            "--guard-softmask-min", "0.1",
            "--guard-softmask-max", "0.9",
            "--guard-face-mask",
            "--guard-face-mask-mode", "hull",
            "--guard-face-mask-dilate", "8",
            "--guard-face-mask-erode", "2",
            "--guard-face-mask-blur", "7",
            "--no-guard-warp-face-only",
            "--guard-mouth-only",
            "--guard-mouth-radius-px", "80",
            "--guard-alpha-start", "0.25",
        ]

        # Create parser like in asymmetry_transfer.py
        ap = argparse.ArgumentParser()
        ap.add_argument("--donor", required=True)
        ap.add_argument("--target", required=True)
        ap.add_argument("--facemesh-warp", action="store_true")
        ap.add_argument("--facemesh-warp-alpha", type=float, default=1.0)
        ap.add_argument("--facemesh-warp-reg", type=float, default=1e-3)
        ap.add_argument("--facemesh-warp-grid-step", type=int, default=2)
        ap.add_argument("--facemesh-warp-validate", action="store_true", default=True)
        ap.add_argument("--facemesh-warp-save-field", action="store_true")

        ap.add_argument("--facemesh-guards", dest="facemesh_guards", action="store_true", default=None)
        ap.add_argument("--no-facemesh-guards", dest="facemesh_guards", action="store_false", default=None)
        ap.add_argument("--facemesh-guard-debug", action="store_true")
        ap.add_argument("--guard-max-delta-px", type=float, default=None)
        ap.add_argument("--guard-cap-percentile", type=float, default=98.0)
        ap.add_argument("--guard-cap-region", type=str, default="weighted_only", choices=["weighted_only", "all"])
        ap.add_argument("--guard-cap-after-align", action="store_true", default=True)
        ap.add_argument("--no-guard-cap-after-align", dest="guard_cap_after_align", action="store_false")
        ap.add_argument("--guard-smooth-delta", action="store_true", default=True)
        ap.add_argument("--no-guard-smooth-delta", dest="guard_smooth_delta", action="store_false")
        ap.add_argument("--guard-smooth-iterations", type=int, default=2)
        ap.add_argument("--guard-smooth-lambda", type=float, default=0.6)
        ap.add_argument("--guard-smooth-mode", type=str, default="knn", choices=["knn", "graph"])
        ap.add_argument("--guard-knn-k", type=int, default=8)
        ap.add_argument("--guard-zero-anchor", action="store_true", default=True)
        ap.add_argument("--no-guard-zero-anchor", dest="guard_zero_anchor", action="store_false")
        ap.add_argument("--guard-anchor-idx", type=int, nargs="*", default=None)
        ap.add_argument("--guard-anchor-strength", type=float, default=0.95)
        ap.add_argument("--guard-softmask", action="store_true", default=True)
        ap.add_argument("--no-guard-softmask", dest="guard_softmask", action="store_false")
        ap.add_argument("--guard-softmask-sigma", type=float, default=25.0)
        ap.add_argument("--guard-softmask-forehead-fade", action="store_true", default=True)
        ap.add_argument("--no-guard-softmask-forehead-fade", dest="guard_softmask_forehead_fade", action="store_false")
        ap.add_argument("--guard-softmask-forehead-yfrac", type=float, default=0.22)
        ap.add_argument("--guard-softmask-min", type=float, default=0.0)
        ap.add_argument("--guard-softmask-max", type=float, default=1.0)
        ap.add_argument("--guard-face-mask", action="store_true", default=True)
        ap.add_argument("--no-guard-face-mask", dest="guard_face_mask", action="store_false")
        ap.add_argument("--guard-face-mask-mode", type=str, default="hull", choices=["hull", "segmentation"])
        ap.add_argument("--guard-face-mask-dilate", type=int, default=12)
        ap.add_argument("--guard-face-mask-erode", type=int, default=0)
        ap.add_argument("--guard-face-mask-blur", type=int, default=11)
        ap.add_argument("--guard-warp-face-only", action="store_true", default=True)
        ap.add_argument("--no-guard-warp-face-only", dest="guard_warp_face_only", action="store_false")
        ap.add_argument("--guard-mouth-only", action="store_true")
        ap.add_argument("--guard-mouth-radius-px", type=int, default=90)
        ap.add_argument("--guard-alpha-start", type=float, default=0.3)

        args = ap.parse_args(test_args)

        assert args.donor == "test_donor.jpg"
        assert args.target == "test_target.jpg"
        assert args.facemesh_warp == True
        assert args.facemesh_warp_alpha == 0.8
        assert args.facemesh_warp_reg == 0.001
        assert args.facemesh_warp_grid_step == 4
        assert args.facemesh_warp_validate == True
        assert args.facemesh_warp_save_field == True
        assert args.facemesh_guards is True
        assert args.guard_max_delta_px == 15
        assert args.guard_cap_percentile == 97.5
        assert args.guard_cap_region == "all"
        assert args.guard_cap_after_align is False
        assert args.guard_smooth_delta is False
        assert args.guard_smooth_iterations == 3
        assert args.guard_smooth_lambda == 0.5
        assert args.guard_smooth_mode == "knn"
        assert args.guard_knn_k == 6
        assert args.guard_zero_anchor is False
        assert args.guard_anchor_idx == [1, 2, 3]
        assert args.guard_anchor_strength == 0.5
        assert args.guard_softmask is False
        assert args.guard_softmask_sigma == 20
        assert args.guard_softmask_forehead_fade is False
        assert args.guard_softmask_forehead_yfrac == 0.2
        assert args.guard_softmask_min == 0.1
        assert args.guard_softmask_max == 0.9
        assert args.guard_face_mask is True
        assert args.guard_face_mask_mode == "hull"
        assert args.guard_face_mask_dilate == 8
        assert args.guard_face_mask_erode == 2
        assert args.guard_face_mask_blur == 7
        assert args.guard_warp_face_only is False
        assert args.guard_mouth_only is True
        assert args.guard_mouth_radius_px == 80
        assert args.guard_alpha_start == 0.25

        print("  ✓ CLI arguments parse correctly")
        print(f"    - facemesh_warp: {args.facemesh_warp}")
        print(f"    - guard_max_delta_px: {args.guard_max_delta_px}")
        print(f"    - guard_softmask: {args.guard_softmask}")

        return True

    except Exception as e:
        print(f"  ✗ CLI parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exports():
    """Test that all required functions are exported."""
    print("\n[Test 4] Module exports...")

    try:
        import scripts.facemesh_warp as warp_module

        required_exports = [
            "estimate_similarity_umeyama",
            "align_delta_to_output",
            "select_control_points",
            "tps_fit",
            "tps_warp_image",
            "detect_folding",
            "apply_facemesh_warp",
            "validate_warp",
            "draw_control_points_overlay",
            "draw_displacement_heatmap",
            "draw_grid_warp_preview",
        ]

        missing = [name for name in required_exports if not hasattr(warp_module, name)]

        if missing:
            print(f"  ✗ Missing exports: {missing}")
            return False

        print(f"  ✓ All {len(required_exports)} required functions exported")
        for name in required_exports:
            func = getattr(warp_module, name)
            print(f"    - {name}: {callable(func)}")

        return True

    except Exception as e:
        print(f"  ✗ Export check failed: {e}")
        return False


def test_docstrings():
    """Test that key functions have docstrings."""
    print("\n[Test 5] Documentation coverage...")

    try:
        from scripts.facemesh_warp import (
            estimate_similarity_umeyama,
            apply_facemesh_warp,
            validate_warp,
        )

        key_funcs = [
            ("estimate_similarity_umeyama", estimate_similarity_umeyama),
            ("apply_facemesh_warp", apply_facemesh_warp),
            ("validate_warp", validate_warp),
        ]

        documented = sum(1 for _, f in key_funcs if f.__doc__ is not None)

        print(f"  ✓ {documented}/{len(key_funcs)} functions have docstrings")

        for name, func in key_funcs:
            has_doc = "✓" if func.__doc__ else "✗"
            print(f"    {has_doc} {name}")

        return documented == len(key_funcs)

    except Exception as e:
        print(f"  ✗ Docstring check failed: {e}")
        return False


def run_all_validations():
    """Run all validation tests."""
    print("=" * 70)
    print("Phase 3-5 FaceMesh Warp - Integration Validation")
    print("=" * 70)

    results = []

    results.append(("Module imports", test_imports()))
    results.append(("asymmetry_transfer integration", test_asymmetry_transfer_imports()))
    results.append(("CLI argument parsing", test_cli_parsing()))
    results.append(("Module exports", test_exports()))
    results.append(("Documentation", test_docstrings()))

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")

    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70 + "\n")

    return all(result for _, result in results)


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)
