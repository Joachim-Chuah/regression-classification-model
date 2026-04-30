"""
Experiment runner for walk-forward sweeps.

Weight sweep (default):
    python -m src.experiments
    python -m src.experiments --sweep weight --weight-clips "0.01,0.10 0.02,0.10 0.01,0.05"

Vol-label sweep:
    python -m src.experiments --sweep vol --vol-k-values 0.75,1.0,1.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.walk_forward import walk_forward_cv


def run_vol_label_sweep(
    vol_k_values: list[float],
    output_dir: str = "artifacts/metrics/experiments",
    include_fixed_baseline: bool = True,
    vol_min_threshold: float = 0.005,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Volatility-Scaled Label Sweep ===")
    print(f"  Output dir: {out_dir.resolve()}")
    print(f"  vol_k values: {vol_k_values}")
    print(f"  min threshold: {vol_min_threshold:.2%}\n")

    if include_fixed_baseline:
        fixed_path = out_dir / "walk_forward_fixed.csv"
        print(">>> Running fixed-threshold baseline")
        walk_forward_cv(
            label_mode="fixed",
            export_csv_path=str(fixed_path),
        )
        print()

    for k in vol_k_values:
        out_path = out_dir / f"walk_forward_vol_scaled_k{k:.2f}.csv"
        print(f">>> Running vol-scaled labels with k={k:.2f}")
        walk_forward_cv(
            label_mode="vol_scaled",
            vol_k=k,
            vol_min_threshold=vol_min_threshold,
            export_csv_path=str(out_path),
        )
        print()

    print("Sweep complete.")


def run_weight_sweep(
    clip_ranges: list[tuple[float, float]] | None = None,
    output_dir: str = "artifacts/metrics/experiments",
    include_unweighted_baseline: bool = True,
) -> None:
    """
    Sweep over sample-weight clip ranges and archive walk-forward CSVs.

    Default clip ranges: (0.01, 0.10), (0.02, 0.10), (0.01, 0.05).
    Always includes an unweighted baseline unless suppressed.
    Compare CSVs on p60_precision, top10_precision, top10_mean_return —
    promote only when improvements are stable across all five years.
    """
    if clip_ranges is None:
        clip_ranges = [(0.01, 0.10), (0.02, 0.10), (0.01, 0.05)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Sample Weight Sweep ===")
    print(f"  Output dir: {out_dir.resolve()}")
    print(f"  Clip ranges: {clip_ranges}\n")

    if include_unweighted_baseline:
        path = out_dir / "walk_forward_weight_none.csv"
        print(">>> Unweighted baseline (calibrated)")
        walk_forward_cv(weight_clip=None, export_csv_path=str(path))
        print()

    for low, high in clip_ranges:
        path = out_dir / f"walk_forward_weight_{low:.2f}_{high:.2f}.csv"
        print(f">>> weight_clip=({low:.2f}, {high:.2f})")
        walk_forward_cv(weight_clip=(low, high), export_csv_path=str(path))
        print()

    print("Sweep complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward sweeps (weight or vol-label).",
    )
    parser.add_argument(
        "--sweep",
        choices=["weight", "vol"],
        default="weight",
        help="weight = sample-weight clip sweep (default) | vol = vol-label k sweep",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/metrics/experiments",
        help="Directory where per-run CSV files are written.",
    )
    # Weight sweep args
    parser.add_argument(
        "--weight-clips",
        default="0.01,0.10 0.02,0.10 0.01,0.05",
        help="Space-separated clip ranges: e.g. '0.01,0.10 0.02,0.10'",
    )
    parser.add_argument(
        "--skip-unweighted-baseline",
        action="store_true",
        help="Skip the unweighted baseline in the weight sweep.",
    )
    # Vol-label sweep args
    parser.add_argument(
        "--vol-k-values",
        default="0.75,1.0,1.25",
        help="Comma-separated k values for vol-scaled neutral threshold.",
    )
    parser.add_argument(
        "--vol-min-threshold",
        type=float,
        default=0.005,
        help="Floor for dynamic threshold (e.g. 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--skip-fixed-baseline",
        action="store_true",
        help="Skip fixed-threshold baseline in the vol sweep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.sweep == "weight":
        clip_ranges = [
            tuple(float(v) for v in pair.split(","))
            for pair in args.weight_clips.split()
            if pair.strip()
        ]
        run_weight_sweep(
            clip_ranges=clip_ranges,
            output_dir=args.output_dir,
            include_unweighted_baseline=not args.skip_unweighted_baseline,
        )
    else:
        k_values = [float(x.strip()) for x in args.vol_k_values.split(",") if x.strip()]
        run_vol_label_sweep(
            vol_k_values=k_values,
            output_dir=args.output_dir,
            include_fixed_baseline=not args.skip_fixed_baseline,
            vol_min_threshold=args.vol_min_threshold,
        )
