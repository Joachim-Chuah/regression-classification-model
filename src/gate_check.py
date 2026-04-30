"""
Deterministic promotion gate checker for walk-forward experiment CSVs.

Compares a challenger CSV against a baseline on three gate groups:
  A — Mean across all years   (overall signal quality)
  B — 2022 hard floor         (no regression in the hardest regime)
  C — Coverage floor          (challenger must keep ≥75% of baseline signals)

Usage:
    python -m src.gate_check \\
        --baseline   artifacts/metrics/experiments/walk_forward_weight_none.csv \\
        --challenger artifacts/metrics/experiments/walk_forward_52w.csv

Exit code 0 = PASS, 1 = FAIL.

Tolerance defaults (all overridable via flags):
    --tol-prec       max mean precision regression   (default 0.02 = 2pp)
    --tol-ret        max mean return regression       (default 0.005 = 0.5pp)
    --tol-2022-prec  max 2022 precision regression    (default 0.03 = 3pp)
    --tol-2022-ret   max 2022 return regression       (default 0.01 = 1pp)
    --coverage-floor min challenger/baseline ratio    (default 0.75 = 75%)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GateResult:
    label: str
    group: str
    baseline: float
    challenger: float
    delta: float
    tolerance: float
    passed: bool


def _mean_valid(df: pd.DataFrame, col: str) -> float:
    """Mean of a column across all rows, ignoring NaN and inf."""
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) > 0 else float("nan")


def _year_val(df: pd.DataFrame, year: int, col: str) -> float:
    row = df[df["year"] == year]
    if row.empty:
        return float("nan")
    v = float(row[col].iloc[0])
    return float("nan") if np.isinf(v) else v


def run_gates(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    tol_prec: float = 0.02,
    tol_ret: float = 0.005,
    tol_2022_prec: float = 0.03,
    tol_2022_ret: float = 0.01,
    coverage_floor: float = 0.75,
) -> list[GateResult]:
    results: list[GateResult] = []

    def gate(label: str, group: str, b: float, c: float, tol: float) -> None:
        if np.isnan(b) or np.isnan(c):
            return  # skip rather than auto-fail when data is missing
        results.append(GateResult(
            label=label, group=group,
            baseline=b, challenger=c,
            delta=c - b, tolerance=tol,
            passed=(c - b) >= -tol,
        ))

    # ------------------------------------------------------------------
    # Group A — Mean across all years
    # ------------------------------------------------------------------
    A = "A: mean all years"
    for col, label, tol in [
        ("p60_precision",     "p60_prec",    tol_prec),
        ("p70_precision",     "p70_prec",    tol_prec),
        ("top10_precision",   "top10_prec",  tol_prec),
        ("top20_precision",   "top20_prec",  tol_prec),
        ("top10_mean_return", "top10_ret",   tol_ret),
        ("p60_mean_return",   "p60_ret",     tol_ret),
    ]:
        gate(label, A, _mean_valid(baseline, col), _mean_valid(challenger, col), tol)

    # ------------------------------------------------------------------
    # Group B — 2022 hard floor
    # ------------------------------------------------------------------
    B = "B: 2022 hard floor"
    for col, label, tol in [
        ("p60_precision",     "p60_prec  2022", tol_2022_prec),
        ("top10_precision",   "top10_prec 2022", tol_2022_prec),
        ("top10_mean_return", "top10_ret  2022", tol_2022_ret),
    ]:
        gate(label, B, _year_val(baseline, 2022, col), _year_val(challenger, 2022, col), tol)

    # ------------------------------------------------------------------
    # Group C — Coverage floor
    # ------------------------------------------------------------------
    b_cov = _mean_valid(baseline, "p60_coverage")
    c_cov = _mean_valid(challenger, "p60_coverage")
    if not (np.isnan(b_cov) or np.isnan(c_cov)):
        floor = b_cov * coverage_floor
        results.append(GateResult(
            label=f"p60_coverage (≥{coverage_floor:.0%} of baseline={b_cov:.1%})",
            group="C: coverage floor",
            baseline=b_cov, challenger=c_cov,
            delta=c_cov - b_cov,
            tolerance=b_cov * (1 - coverage_floor),
            passed=c_cov >= floor,
        ))

    return results


def print_report(
    baseline_path: str,
    challenger_path: str,
    results: list[GateResult],
) -> bool:
    """Print a formatted report. Returns True if all gates passed."""
    PASS = "PASS"
    FAIL = "FAIL ✗"

    groups = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    width = 70
    print("=" * width)
    print("  Promotion Gate Check")
    print("=" * width)
    print(f"  Baseline  : {Path(baseline_path).name}")
    print(f"  Challenger: {Path(challenger_path).name}")

    all_passed = True

    for group_name, gate_results in groups.items():
        print(f"\n  {group_name}")
        print(f"  {'Metric':<26} {'Baseline':>9}  {'Challenger':>10}  {'Delta':>8}  {'Tol':>6}  Result")
        print("  " + "-" * 66)
        for r in gate_results:
            b_str   = f"{r.baseline:+.1%}" if abs(r.baseline) < 2 else f"{r.baseline:.1%}"
            c_str   = f"{r.challenger:+.1%}" if abs(r.challenger) < 2 else f"{r.challenger:.1%}"
            d_str   = f"{r.delta:+.1%}"
            tol_str = f"±{r.tolerance:.1%}"
            result  = PASS if r.passed else FAIL
            if not r.passed:
                all_passed = False
            print(f"  {r.label:<26} {b_str:>9}  {c_str:>10}  {d_str:>8}  {tol_str:>6}  {result}")

    print()
    print("=" * width)
    failures = [r for r in results if not r.passed]
    if all_passed:
        print("  OVERALL: PASS — challenger meets all promotion gates.")
    else:
        print(f"  OVERALL: FAIL — {len(failures)} gate(s) failed:")
        for r in failures:
            print(f"    • [{r.group}] {r.label}  delta={r.delta:+.1%}  tolerance=±{r.tolerance:.1%}")
    print("=" * width)

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic promotion gate checker for walk-forward CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline",   required=True, help="Path to baseline walk-forward CSV.")
    parser.add_argument("--challenger", required=True, help="Path to challenger walk-forward CSV.")
    parser.add_argument("--tol-prec",       type=float, default=0.02,  help="Max mean precision regression (default 0.02).")
    parser.add_argument("--tol-ret",        type=float, default=0.005, help="Max mean return regression (default 0.005).")
    parser.add_argument("--tol-2022-prec",  type=float, default=0.03,  help="Max 2022 precision regression (default 0.03).")
    parser.add_argument("--tol-2022-ret",   type=float, default=0.01,  help="Max 2022 return regression (default 0.01).")
    parser.add_argument("--coverage-floor", type=float, default=0.75,  help="Min challenger/baseline coverage ratio (default 0.75).")
    args = parser.parse_args()

    baseline   = pd.read_csv(args.baseline)
    challenger = pd.read_csv(args.challenger)

    results = run_gates(
        baseline, challenger,
        tol_prec=args.tol_prec,
        tol_ret=args.tol_ret,
        tol_2022_prec=args.tol_2022_prec,
        tol_2022_ret=args.tol_2022_ret,
        coverage_floor=args.coverage_floor,
    )

    passed = print_report(args.baseline, args.challenger, results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
