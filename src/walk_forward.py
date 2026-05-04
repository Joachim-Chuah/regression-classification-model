"""
Walk-forward cross-validation for the 3-class classifier.

Expanding-window: trains on all data up to (test_year - 1), evaluates on
test_year. Each fold is a full calendar year so the market regime is clearly
labelled. Five folds cover COVID crash (2020), low-rate bull (2021), bear /
rate-shock (2022), recovery (2023), and bull (2024).

Consistent performance across all five means the model has real signal.
Strong performance in only one or two means it learned a regime-specific
pattern that will fail in production.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score
from xgboost import XGBClassifier

from src.constants import (
    DEFAULT_HORIZON, DEFAULT_TICKERS, NEUTRAL_THRESHOLD,
    RANDOM_STATE, TICKER_SECTOR_ETF,
)
from src.data import pull, pull_earnings_dates, pull_macro
from src.features import compute_features
from src.labels import compute_sample_weights, make_3class_labels, make_3class_labels_vol_scaled, make_returns
from src.evaluate import signal_quality_table, topk_precision_table
from src.models import CalibratedXGB3Class

_REGIME_LABELS = {
    2020: "COVID crash",
    2021: "low-rate bull",
    2022: "bear/rate shock",
    2023: "recovery",
    2024: "bull",
}

_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    objective="multi:softprob",
    num_class=3,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
    verbosity=0,
)


def walk_forward_cv(
    tickers: list[str] = DEFAULT_TICKERS,
    horizon: int = DEFAULT_HORIZON,
    test_years: list[int] | None = None,
    export_csv_path: str | None = None,
    label_mode: str = "fixed",  # "fixed" | "vol_scaled"
    vol_k: float = 1.0,
    vol_min_threshold: float = 0.005,
    calibrate: bool = False,
    cal_window_years: int = 1,
    weight_clip: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation of the 3-class classifier.

    Builds the full dataset once (parquet cache makes this fast after the
    first run), then for each test year trains on all prior data and
    evaluates on that year alone. Returns a summary DataFrame.

    calibrate=False (default) uses raw XGBoost probabilities on the full
    training window. This is more stable across folds because per-fold
    isotonic calibration is highly sensitive to the regime of the single
    calibration year (e.g. calibrating on a bear year suppresses P(up)
    for the following recovery year, causing near-zero coverage).

    calibrate=True holds out the last cal_window_years of each training
    window as an isotonic calibration set, mirroring the production path.
    Use it for direct Brier score comparison against production, but
    interpret coverage/precision metrics with caution due to regime sensitivity.

    If export_csv_path is provided, writes the summary DataFrame to CSV.
    """
    if test_years is None:
        test_years = [2020, 2021, 2022, 2023, 2024]

    print("=== Walk-Forward Cross-Validation ===")
    print(f"  Tickers: {len(tickers)}  |  Horizon: {horizon}d  |  Years: {test_years}")
    if label_mode == "vol_scaled":
        print(f"  Label mode: vol_scaled (k={vol_k:.2f}, min={vol_min_threshold:.2%})")
    else:
        print(f"  Label mode: fixed (±{NEUTRAL_THRESHOLD:.0%})")
    cal_note = f"calibrated (cal_window={cal_window_years}y)" if calibrate else "raw (no calibration)"
    print(f"  Calibration: {cal_note}")
    w_note = f"clip abs(return) to [{weight_clip[0]:.2%}, {weight_clip[1]:.2%}]" if weight_clip else "none"
    print(f"  Sample weights: {w_note}\n")

    # Build the full 2015-2024 dataset once
    print("  Building full dataset (using parquet cache where available)...")
    macro = pull_macro("2015-01-01", "2024-12-31")
    frames = []
    for ticker in tickers:
        df = pull(ticker, start="2015-01-01", end="2024-12-31")
        sector_etf = TICKER_SECTOR_ETF.get(ticker)
        ed = pull_earnings_dates(ticker)
        X_ticker = compute_features(df, macro=macro, sector_etf=sector_etf, earnings_dates=ed)
        if label_mode == "vol_scaled":
            y_ticker = make_3class_labels_vol_scaled(
                df["close"],
                atr_pct=X_ticker["atr_pct"],
                horizon=horizon,
                k=vol_k,
                min_threshold=vol_min_threshold,
            )
        else:
            y_ticker = make_3class_labels(df["close"], horizon, NEUTRAL_THRESHOLD)
        r_ticker = make_returns(df["close"], horizon).rename("forward_return")
        combined = X_ticker.join(y_ticker.rename("label")).join(r_ticker).dropna()
        frames.append(combined)
    all_data = pd.concat(frames).sort_index()
    X_all = all_data.drop(columns=["label", "forward_return"])
    y_all = all_data["label"]
    ret_all = all_data["forward_return"]
    print(f"  Total rows available: {len(X_all):,}\n")

    results = []
    for test_year in test_years:
        train_end  = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end   = f"{test_year}-12-31"

        X_train = X_all[X_all.index <= train_end]
        y_train = y_all[y_all.index <= train_end]
        X_test  = X_all[(X_all.index >= test_start) & (X_all.index <= test_end)]
        y_test  = y_all[(y_all.index >= test_start) & (y_all.index <= test_end)]
        r_test  = ret_all[(ret_all.index >= test_start) & (ret_all.index <= test_end)]

        if len(X_train) < 2000 or len(X_test) < 200:
            print(f"  {test_year}: insufficient data, skipping")
            continue

        if calibrate:
            xgb_train_end = f"{test_year - 1 - cal_window_years}-12-31"
            cal_start     = f"{test_year - cal_window_years}-01-01"
            cal_end       = f"{test_year - 1}-12-31"
            X_train_xgb   = X_all[X_all.index <= xgb_train_end]
            y_train_xgb   = y_all[y_all.index <= xgb_train_end]
            X_cal         = X_all[(X_all.index >= cal_start) & (X_all.index <= cal_end)]
            y_cal         = y_all[(y_all.index >= cal_start) & (y_all.index <= cal_end)]
            if len(X_train_xgb) < 1000 or len(X_cal) < 100:
                print(f"  {test_year}: insufficient data for calibration split, skipping")
                continue
            xgb = XGBClassifier(**_XGB_PARAMS)
            w_train = compute_sample_weights(ret_all[ret_all.index <= xgb_train_end].values, weight_clip[0], weight_clip[1]) if weight_clip else None
            xgb.fit(X_train_xgb, y_train_xgb, sample_weight=w_train)
            raw_cal = xgb.predict_proba(X_cal)
            calibrators = []
            for k in range(3):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_cal[:, k], (y_cal == k).astype(float))
                calibrators.append(iso)
            fitted_model = CalibratedXGB3Class(xgb, calibrators)
        else:
            xgb = XGBClassifier(**_XGB_PARAMS)
            w_train = compute_sample_weights(ret_all[ret_all.index <= train_end].values, weight_clip[0], weight_clip[1]) if weight_clip else None
            xgb.fit(X_train, y_train, sample_weight=w_train)
            fitted_model = xgb

        y_pred  = fitted_model.predict(X_test)
        y_proba = fitted_model.predict_proba(X_test)

        y_up_bin       = (y_test == 2).astype(int)
        p_up           = y_proba[:, 2]
        baseline_brier = brier_score_loss(y_up_bin, np.full(len(y_up_bin), y_up_bin.mean()))
        model_brier    = brier_score_loss(y_up_bin, p_up)
        brier_skill    = 1 - model_brier / baseline_brier if baseline_brier > 0 else float("nan")

        up_prec = precision_score(y_test == 2, y_pred == 2, zero_division=0)
        up_rec  = recall_score(y_test == 2, y_pred == 2, zero_division=0)
        acc     = accuracy_score(y_test, y_pred)
        thresh_tbl = signal_quality_table(
            y=y_test,
            p_up=p_up,
            forward_return=r_test,
            thresholds=[0.60, 0.70],
        )
        topk_tbl = topk_precision_table(
            y=y_test,
            p_up=p_up,
            forward_return=r_test,
            topk_fracs=[0.10, 0.20],
        )

        p60 = thresh_tbl[thresh_tbl["threshold"] == 0.60].iloc[0]
        p70 = thresh_tbl[thresh_tbl["threshold"] == 0.70].iloc[0]
        t10 = topk_tbl[topk_tbl["topk_frac"] == 0.10].iloc[0]
        t20 = topk_tbl[topk_tbl["topk_frac"] == 0.20].iloc[0]

        regime = _REGIME_LABELS.get(test_year, "")
        print(
            f"  {test_year} ({regime:15s}):  "
            f"acc={acc:.1%}  up_prec={up_prec:.1%}  up_rec={up_rec:.1%}  "
            f"brier={brier_skill:+.4f}  "
            f"actual_up={y_up_bin.mean():.1%}  pred_up={(y_pred == 2).mean():.1%}  "
            f"P60_prec={p60['precision_up']:.1%}  P70_prec={p70['precision_up']:.1%}  "
            f"Top10_prec={t10['precision_up']:.1%}"
        )

        results.append({
            "year":        test_year,
            "regime":      regime,
            "train_rows":  len(X_train_xgb) if calibrate else len(X_train),
            "cal_rows":    len(X_cal) if calibrate else 0,
            "test_rows":   len(X_test),
            "accuracy":    acc,
            "up_precision": up_prec,
            "up_recall":   up_rec,
            "brier_skill": brier_skill,
            "actual_up":   float(y_up_bin.mean()),
            "pred_up":     float((y_pred == 2).mean()),
            "p60_precision": float(p60["precision_up"]),
            "p60_coverage": float(p60["coverage"]),
            "p60_mean_return": float(p60["mean_forward_return"]),
            "p70_precision": float(p70["precision_up"]),
            "p70_coverage": float(p70["coverage"]),
            "p70_mean_return": float(p70["mean_forward_return"]),
            "top10_precision": float(t10["precision_up"]),
            "top10_mean_return": float(t10["mean_forward_return"]),
            "top20_precision": float(t20["precision_up"]),
            "top20_mean_return": float(t20["mean_forward_return"]),
        })

    summary = pd.DataFrame(results)

    print(f"\n  {'Metric':<20} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * 58)
    for col in [
        "accuracy", "up_precision", "up_recall", "brier_skill",
        "p60_precision", "p70_precision", "top10_precision",
    ]:
        print(
            f"  {col:<20} {summary[col].mean():>8.3f}  {summary[col].std():>8.3f}"
            f"  {summary[col].min():>8.3f}  {summary[col].max():>8.3f}"
        )

    if export_csv_path:
        out_path = Path(export_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\n  Saved walk-forward summary to: {out_path.resolve()}")

    return summary


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--output", default="artifacts/metrics/experiments/walk_forward_summary.csv",
                    help="Path to write the CSV results.")
    _p.add_argument("--no-calibrate", action="store_true",
                    help="Skip per-fold isotonic calibration (use raw XGBoost probabilities).")
    _args = _p.parse_args()
    results = walk_forward_cv(export_csv_path=_args.output, calibrate=not _args.no_calibrate)
    print("\nFull results:")
    print(results.to_string(index=False))
