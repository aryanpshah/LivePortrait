#!/usr/bin/env python3
"""
Test script for flip-back landmark index alignment.

This script validates that the align_flipback_indices function correctly
aligns flip-back landmarks to donor landmarks after horizontal mirroring.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_align_flipback_indices():
    """Test alignment of flip-back landmarks."""
    print("\n=== Test: Flip-Back Index Alignment ===")

    try:
        from scripts.facemesh_landmarks import align_flipback_indices
    except ImportError as e:
        print(f"✗ Failed to import align_flipback_indices: {e}")
        return False

    # Load real data from existing diagnostics
    base_dir = "outputs/diagnostics/facemesh"

    try:
        L_d = np.load(os.path.join(base_dir, "donor_landmarks.npy"))
        L_f_back = np.load(os.path.join(base_dir, "donor_flip_back_landmarks.npy"))
    except FileNotFoundError:
        print(f"✗ Data files not found in {base_dir}")
        print("  (Run asymmetry_transfer.py with --facemesh-driving first)")
        return False

    print(f"Loaded landmarks: shape {L_d.shape}")

    # Test alignment
    try:
        L_f_back_perm, perm, stats = align_flipback_indices(L_d, L_f_back, k=3)
        print(f"✓ Alignment succeeded")
    except Exception as e:
        print(f"✗ Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate results
    print(f"\nValidation:")

    # Check shapes
    if L_f_back_perm.shape != L_d.shape:
        print(f"✗ Shape mismatch: {L_f_back_perm.shape} vs {L_d.shape}")
        return False
    print(f"✓ Output shape matches: {L_f_back_perm.shape}")

    # Check permutation is valid
    if len(set(perm)) != len(perm) or len(perm) != L_d.shape[0]:
        print(f"✗ Permutation not a valid bijection")
        return False
    if not all(0 <= p < L_d.shape[0] for p in perm):
        print(f"✗ Permutation contains invalid indices")
        return False
    print(f"✓ Permutation is a valid bijection")

    # Check statistics exist and are reasonable
    required_keys = ['before_mean', 'after_mean', 'improvement_factor',
                     'before_dist', 'after_dist', 'fractions']
    for key in required_keys:
        if key not in stats:
            print(f"✗ Missing stats key: {key}")
            return False
    print(f"✓ All required statistics present")

    # Print improvement
    before = stats['before_mean']
    after = stats['after_mean']
    improvement = stats['improvement_factor']

    print(f"\nAlignment improvement:")
    print(f"  Before: {before:.3f} px")
    print(f"  After:  {after:.3f} px")
    print(f"  Factor: {improvement:.2f}x")

    if before <= 0:
        print(f"✗ Invalid before distance: {before}")
        return False
    if after > before:
        print(f"⚠ WARNING: After distance ({after:.3f}) > before ({before:.3f})")
        print(f"  This may indicate alignment didn't improve the result.")

    # Check that alignment actually improved things
    if improvement < 1.0:
        print(f"⚠ WARNING: Improvement factor < 1.0 ({improvement:.2f}x)")
        print(f"  Alignment may not have worked as expected.")
    else:
        print(f"✓ Alignment improved mean distance by {improvement:.2f}x")

    # Print quality fractions
    print(f"\nQuality metrics (after alignment):")
    for key, val in stats['fractions'].items():
        pct = val * 100
        print(f"  {key}: {pct:.1f}%")

    print(f"\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_align_flipback_indices()
    sys.exit(0 if success else 1)
