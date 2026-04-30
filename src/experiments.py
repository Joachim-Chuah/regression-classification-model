"""
Small experiment runner for walk-forward label sweeps.

Example:
    python -m src.experiments --vol-k-values 0.75,1.0,1.25
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward sweeps for volatility-scaled labels.",
    )
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
        "--output-dir",
        default="artifacts/metrics/experiments",
        help="Directory where per-run CSV files are written.",
    )
    parser.add_argument(
        "--skip-fixed-baseline",
        action="store_true",
        help="Skip fixed-threshold baseline run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    k_values = [
        float(x.strip())
        for x in args.vol_k_values.split(",")
        if x.strip()
    ]
    run_vol_label_sweep(
        vol_k_values=k_values,
        output_dir=args.output_dir,
        include_fixed_baseline=not args.skip_fixed_baseline,
        vol_min_threshold=args.vol_min_threshold,
    )
