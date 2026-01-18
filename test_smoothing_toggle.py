#!/usr/bin/env python3
"""
Test script to verify smoothing toggle functionality.
Run this after executing asymmetry_transfer with --facemesh-exp-debug enabled.
"""

import os
import json
import numpy as np
import sys

def test_smoothing_output(debug_dir="outputs/diagnostics/facemesh_exp_assist"):
    """Verify that smoothing was applied correctly."""
    
    if not os.path.exists(debug_dir):
        print(f"❌ Debug directory not found: {debug_dir}")
        return False
    
    # Load summary
    summary_path = os.path.join(debug_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"❌ summary.json not found: {summary_path}")
        return False
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    print("\n" + "="*70)
    print("FACEMESH EXPRESSION ASSIST - SMOOTHING VERIFICATION")
    print("="*70)
    
    # Check summary fields
    print("\n[summary.json fields]")
    print(f"  smoothing_enabled: {summary.get('smoothing_enabled', 'MISSING')}")
    print(f"  smoothing_k: {summary.get('smoothing_k', 'MISSING')}")
    print(f"  smoothing_lam: {summary.get('smoothing_lam', 'MISSING')}")
    print(f"  smoothing_mean_l2_diff: {summary.get('smoothing_mean_l2_diff', 'MISSING'):.6f}")
    
    smoothing_enabled = summary.get("smoothing_enabled", False)
    smoothing_diff = summary.get("smoothing_mean_l2_diff", 0.0)
    
    # Load lip vectors if they exist
    raw_path = os.path.join(debug_dir, "vec_lips_raw.npy")
    smoothed_path = os.path.join(debug_dir, "vec_lips_smoothed.npy")
    
    if os.path.exists(raw_path) and os.path.exists(smoothed_path):
        print("\n[vec_lips_raw.npy & vec_lips_smoothed.npy comparison]")
        raw = np.load(raw_path)
        smoothed = np.load(smoothed_path)
        
        print(f"  raw shape: {raw.shape}")
        print(f"  smoothed shape: {smoothed.shape}")
        
        if raw.shape == smoothed.shape:
            # Compute difference
            diff = np.linalg.norm(smoothed - raw, axis=1)
            mean_diff = diff.mean()
            max_diff = diff.max()
            
            print(f"  mean L2 diff (computed): {mean_diff:.6f}")
            print(f"  max L2 diff (computed): {max_diff:.6f}")
            print(f"  match with summary: {abs(mean_diff - smoothing_diff) < 1e-5}")
            
            # Verify consistency
            if smoothing_enabled:
                if mean_diff > 0.0:
                    print("\n✅ PASS: Smoothing ON → vectors changed (diff > 0)")
                else:
                    print("\n❌ FAIL: Smoothing ON → vectors identical (diff = 0)")
                    return False
            else:
                if mean_diff == 0.0:
                    print("\n✅ PASS: Smoothing OFF → vectors identical (diff = 0)")
                else:
                    print(f"\n❌ FAIL: Smoothing OFF → vectors changed (diff = {mean_diff})")
                    return False
        else:
            print("  ❌ Shape mismatch between raw and smoothed")
            return False
    else:
        print("\n[vec_lips_raw.npy & vec_lips_smoothed.npy]")
        print(f"  raw exists: {os.path.exists(raw_path)}")
        print(f"  smoothed exists: {os.path.exists(smoothed_path)}")
        
        # At least check summary consistency
        if smoothing_enabled:
            if smoothing_diff > 0.0:
                print("\n✅ PASS: Smoothing ON → summary shows diff > 0")
            else:
                print("\n⚠️  WARNING: Smoothing ON → summary shows diff = 0 (unexpected)")
        else:
            if smoothing_diff == 0.0:
                print("\n✅ PASS: Smoothing OFF → summary shows diff = 0")
            else:
                print("\n⚠️  WARNING: Smoothing OFF → summary shows diff > 0 (unexpected)")
    
    print("\n" + "="*70)
    print(f"Smoothing enabled: {smoothing_enabled}")
    print(f"Mean L2 difference: {smoothing_diff:.6f} px")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    # Optional: accept debug_dir as argument
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/diagnostics/facemesh_exp_assist"
    success = test_smoothing_output(debug_dir)
    sys.exit(0 if success else 1)
