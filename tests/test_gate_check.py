import numpy as np
import pandas as pd
import pytest

from src.gate_check import GateResult, run_gates


def _make_df(
    years=(2020, 2021, 2022, 2023, 2024),
    p60=0.55, p70=0.65, top10=0.60, top20=0.55,
    top10_ret=0.04, p60_ret=0.03, p60_cov=0.15,
) -> pd.DataFrame:
    n = len(years)
    return pd.DataFrame({
        "year":              list(years),
        "p60_precision":     [p60]    * n,
        "p70_precision":     [p70]    * n,
        "top10_precision":   [top10]  * n,
        "top20_precision":   [top20]  * n,
        "top10_mean_return": [top10_ret] * n,
        "p60_mean_return":   [p60_ret]   * n,
        "p60_coverage":      [p60_cov]   * n,
    })


def test_identical_csvs_all_pass():
    df = _make_df()
    results = run_gates(df, df)
    assert all(r.passed for r in results)


def test_small_improvement_passes():
    base = _make_df(p60=0.55, top10=0.60, top10_ret=0.04)
    chal = _make_df(p60=0.57, top10=0.62, top10_ret=0.045)
    results = run_gates(base, chal)
    assert all(r.passed for r in results)


def test_regression_within_tolerance_passes():
    base = _make_df(p60=0.55)
    chal = _make_df(p60=0.54)  # -1pp — within default 2pp tolerance
    results = run_gates(base, chal)
    prec_gates = [r for r in results if "p60_prec" in r.label and r.group == "A: mean all years"]
    assert all(r.passed for r in prec_gates)


def test_regression_beyond_tolerance_fails():
    base = _make_df(p60=0.55)
    chal = _make_df(p60=0.52)  # -3pp — beyond default 2pp tolerance
    results = run_gates(base, chal)
    prec_gates = [r for r in results if r.label == "p60_prec" and r.group == "A: mean all years"]
    assert len(prec_gates) == 1
    assert not prec_gates[0].passed


def test_2022_hard_floor_fires_independently():
    """Mean gate passes but 2022 regresses beyond tol_2022_prec."""
    years = [2020, 2021, 2022, 2023, 2024]
    base_p60 = [0.60, 0.60, 0.60, 0.60, 0.60]
    # challenger improves most years but drops 5pp in 2022
    chal_p60 = [0.65, 0.65, 0.55, 0.65, 0.65]

    base = pd.DataFrame({"year": years, "p60_precision": base_p60,
                         "p70_precision": [0.7]*5, "top10_precision": [0.6]*5,
                         "top20_precision": [0.55]*5, "top10_mean_return": [0.04]*5,
                         "p60_mean_return": [0.03]*5, "p60_coverage": [0.15]*5})
    chal = pd.DataFrame({"year": years, "p60_precision": chal_p60,
                         "p70_precision": [0.7]*5, "top10_precision": [0.6]*5,
                         "top20_precision": [0.55]*5, "top10_mean_return": [0.04]*5,
                         "p60_mean_return": [0.03]*5, "p60_coverage": [0.15]*5})

    results = run_gates(base, chal)
    group_a = [r for r in results if r.group == "A: mean all years" and r.label == "p60_prec"]
    group_b = [r for r in results if r.group == "B: 2022 hard floor" and "p60" in r.label]
    assert group_a[0].passed        # mean improved overall
    assert not group_b[0].passed    # 2022 regression fails the hard floor


def test_coverage_floor_fails_when_signals_collapse():
    base = _make_df(p60_cov=0.20)
    chal = _make_df(p60_cov=0.10)  # dropped to 50% of baseline — below 75% floor
    results = run_gates(base, chal)
    cov_gate = [r for r in results if r.group == "C: coverage floor"]
    assert len(cov_gate) == 1
    assert not cov_gate[0].passed


def test_coverage_floor_passes_when_adequate():
    base = _make_df(p60_cov=0.20)
    chal = _make_df(p60_cov=0.16)  # 80% of baseline — above 75% floor
    results = run_gates(base, chal)
    cov_gate = [r for r in results if r.group == "C: coverage floor"]
    assert cov_gate[0].passed


def test_nan_gate_skipped_not_failed():
    base = _make_df(p70=float("nan"))
    chal = _make_df(p70=float("nan"))
    results = run_gates(base, chal, fail_on_missing=False)
    p70_gates = [r for r in results if r.label == "p70_prec"]
    assert len(p70_gates) == 0  # skipped, not counted as failure


def test_missing_2022_skips_hard_floor():
    base = _make_df(years=(2020, 2021, 2023, 2024))
    chal = _make_df(years=(2020, 2021, 2023, 2024))
    results = run_gates(base, chal, fail_on_missing=False)
    group_b = [r for r in results if r.group == "B: 2022 hard floor"]
    assert len(group_b) == 0  # no 2022 row → gates skipped


def test_nan_gate_fails_by_default():
    base = _make_df(p70=float("nan"))
    chal = _make_df(p70=float("nan"))
    results = run_gates(base, chal)  # default fail_on_missing=True
    p70_gates = [r for r in results if r.label == "p70_prec"]
    assert len(p70_gates) == 1
    assert not p70_gates[0].passed


def test_missing_2022_fails_by_default():
    base = _make_df(years=(2020, 2021, 2023, 2024))
    chal = _make_df(years=(2020, 2021, 2023, 2024))
    results = run_gates(base, chal)  # default fail_on_missing=True
    group_b = [r for r in results if r.group == "B: 2022 hard floor"]
    assert len(group_b) == 3
    assert all(not r.passed for r in group_b)
