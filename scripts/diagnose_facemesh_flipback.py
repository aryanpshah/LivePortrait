import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose FaceMesh flip-back landmark index alignment."
    )
    parser.add_argument(
        "--base",
        default=os.path.join("outputs", "diagnostics", "facemesh"),
        help="Directory containing donor_landmarks.npy and donor_flip_back_landmarks.npy",
    )
    parser.add_argument(
        "--print-idx",
        default="234,454,61,291,10,152",
        help="Comma-separated list of landmark indices to print",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to save diagnostic JSON (default: <base>/flipback_diagnostic.json)",
    )
    return parser.parse_args()


def load_landmarks(base_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    ld_path = os.path.join(base_dir, "donor_landmarks.npy")
    lfback_path = os.path.join(base_dir, "donor_flip_back_landmarks.npy")

    if not os.path.isfile(ld_path):
        print(f"ERROR: Missing file: {ld_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(lfback_path):
        print(f"ERROR: Missing file: {lfback_path}", file=sys.stderr)
        sys.exit(1)

    ld = np.load(ld_path)
    lfback = np.load(lfback_path)

    # Validate that both arrays have compatible shapes
    if len(ld.shape) != 2 or len(lfback.shape) != 2:
        print(
            f"ERROR: Arrays must be 2D. Got shapes {ld.shape} and {lfback.shape}",
            file=sys.stderr,
        )
        sys.exit(1)
    if ld.shape[1] != 2 or lfback.shape[1] != 2:
        print(
            f"ERROR: Arrays must have 2 columns (x, y). Got shapes {ld.shape} and {lfback.shape}",
            file=sys.stderr,
        )
        sys.exit(1)
    if ld.shape[0] != lfback.shape[0]:
        print(
            f"ERROR: Shape mismatch. donor_landmarks.npy has {ld.shape[0]} points, "
            f"donor_flip_back_landmarks.npy has {lfback.shape[0]} points",
            file=sys.stderr,
        )
        sys.exit(1)

    return ld, lfback


def load_permutation(base_dir: str) -> np.ndarray | None:
    """Load the permutation array if available, otherwise return None.
    
    Note: The saved L_f_back is already permuted. This loads the permutation
    that was applied so we can show before/after comparison.
    """
    perm_path = os.path.join(base_dir, "flipback_perm.npy")
    if os.path.isfile(perm_path):
        try:
            perm = np.load(perm_path)
            return perm
        except Exception as e:
            print(f"WARNING: Could not load permutation: {e}", file=sys.stderr)
            return None
    return None


def parse_indices(idx_str: str) -> List[int]:
    if not idx_str:
        return []
    try:
        return [int(s.strip()) for s in idx_str.split(",") if s.strip()]
    except ValueError as exc:
        print(f"ERROR: Could not parse --print-idx '{idx_str}': {exc}", file=sys.stderr)
        sys.exit(1)


def compute_stats(distances: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
    }


def format_point(pt: np.ndarray) -> str:
    return f"[{pt[0]:.3f}, {pt[1]:.3f}]"


def main() -> None:
    args = parse_args()
    base_dir = args.base
    out_json = args.out_json or os.path.join(base_dir, "flipback_diagnostic.json")

    ld, lfback = load_landmarks(base_dir)
    perm = load_permutation(base_dir)

    # Note: The loaded lfback is ALREADY permuted (aligned).
    # To show before/after, we need to reconstruct the unaligned version.
    # Unaligned = lfback[inverse_perm]
    
    if perm is not None:
        # Compute inverse permutation: if perm[i] = j, then inv_perm[j] = i
        inv_perm = np.argsort(perm)
        lfback_unaligned = lfback[inv_perm]
        distances_before = np.linalg.norm(ld - lfback_unaligned, axis=1)
        distances_after = np.linalg.norm(ld - lfback, axis=1)  # already aligned
    else:
        lfback_unaligned = lfback
        distances_before = np.linalg.norm(ld - lfback, axis=1)
        distances_after = None

    stats_before = compute_stats(distances_before)

    print("=" * 70)
    print("BEFORE ALIGNMENT")
    print("=" * 70)
    print("Distance stats (pixels):")
    print(f"  mean:   {stats_before['mean']:.3f}")
    print(f"  median: {stats_before['median']:.3f}")
    print(f"  p95:    {stats_before['p95']:.3f}")
    print(f"  p99:    {stats_before['p99']:.3f}")
    print(f"  max:    {stats_before['max']:.3f}")

    top_indices_before = np.argsort(distances_before)[::-1][:10]
    print("Top 10 indices by distance:")
    for idx in top_indices_before:
        print(f"  {idx:3d}: {distances_before[idx]:.3f}")

    # Show after alignment if permutation exists
    if perm is not None:
        stats_after = compute_stats(distances_after)

        print("\n" + "=" * 70)
        print("AFTER ALIGNMENT")
        print("=" * 70)
        print("Distance stats (pixels):")
        print(f"  mean:   {stats_after['mean']:.3f}")
        print(f"  median: {stats_after['median']:.3f}")
        print(f"  p95:    {stats_after['p95']:.3f}")
        print(f"  p99:    {stats_after['p99']:.3f}")
        print(f"  max:    {stats_after['max']:.3f}")

        top_indices_after = np.argsort(distances_after)[::-1][:10]
        print("Top 10 indices by distance:")
        for idx in top_indices_after:
            print(f"  {idx:3d}: {distances_after[idx]:.3f}")

        print("\n" + "=" * 70)
        print("IMPROVEMENT")
        print("=" * 70)
        improvement_factor = stats_before['mean'] / (stats_after['mean'] + 1e-8)
        print(f"  Mean distance improved by {improvement_factor:.2f}x")
        print(f"  {stats_before['mean']:.3f} px → {stats_after['mean']:.3f} px")
        fracs = {
            "lt_2px": float(np.mean(distances_after <= 2.0)),
            "lt_5px": float(np.mean(distances_after <= 5.0)),
            "lt_10px": float(np.mean(distances_after <= 10.0)),
        }
        print(f"  Points within 2 px:  {fracs['lt_2px']*100:.1f}%")
        print(f"  Points within 5 px:  {fracs['lt_5px']*100:.1f}%")
        print(f"  Points within 10 px: {fracs['lt_10px']*100:.1f}%")
    else:
        stats_after = None
        distances_after = None
        fracs = None

    selected_indices = parse_indices(args.print_idx)
    if selected_indices:
        print("\nSelected indices (before alignment):")
        for idx in selected_indices:
            if idx < 0 or idx >= ld.shape[0]:
                print(f"  {idx}: INVALID INDEX (expected 0 <= idx < {ld.shape[0]})")
                continue
            print(
                f"  {idx:3d}: Ld={format_point(ld[idx])}, "
                f"Lfback={format_point(lfback_unaligned[idx])}, dist={distances_before[idx]:.3f}"
            )
        
        if distances_after is not None:
            print("\nSelected indices (after alignment):")
            for idx in selected_indices:
                if idx < 0 or idx >= ld.shape[0]:
                    continue
                print(
                    f"  {idx:3d}: Ld={format_point(ld[idx])}, "
                    f"Lfback={format_point(lfback[idx])}, dist={distances_after[idx]:.3f}"
                )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    result = {
        "base": os.path.abspath(base_dir),
        "has_permutation": perm is not None,
        "stats_before": stats_before,
        "top10_before": [{"idx": int(i), "dist": float(distances_before[i])} for i in top_indices_before.tolist()],
        "selected": [],
    }
    
    if stats_after is not None:
        result["stats_after"] = stats_after
        result["top10_after"] = [{"idx": int(i), "dist": float(distances_after[i])} for i in top_indices_after.tolist()]
        result["improvement_factor"] = float(stats_before['mean'] / (stats_after['mean'] + 1e-8))
        result["quality_fractions"] = fracs

    for idx in selected_indices:
        if idx < 0 or idx >= ld.shape[0]:
            continue
        entry = {
            "idx": int(idx),
            "Ld": [float(ld[idx][0]), float(ld[idx][1])],
            "Lfback_before": [float(lfback_unaligned[idx][0]), float(lfback_unaligned[idx][1])],
            "dist_before": float(distances_before[idx]),
        }
        if distances_after is not None:
            entry["Lfback_after"] = [float(lfback[idx][0]), float(lfback[idx][1])]
            entry["dist_after"] = float(distances_after[idx])
        result["selected"].append(entry)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_json}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if stats_after is not None:
        if stats_after["mean"] > 20.0:
            print("✗ AFTER ALIGNMENT: Still NOT index-aligned (unexpected).")
        else:
            print("✓ AFTER ALIGNMENT: Landmarks are well-aligned!")
    else:
        if stats_before["mean"] > 20.0:
            print(
                "✗ BEFORE ALIGNMENT: NOT index-aligned (left/right indices swapped)."
            )
        else:
            print("✓ BEFORE ALIGNMENT: Landmarks appear index-aligned.")


if __name__ == "__main__":
    main()
