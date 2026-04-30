import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent))

from src.constants import DEFAULT_HORIZON, DEFAULT_TICKERS, TRAIN_END, VAL_END
from src.data import pull, pull_macro
from src.features import compute_features
from src.labels import make_labels, make_returns
from src.splits import time_split
from src.train import CalibratedModel  # noqa: F401 — required for joblib deserialisation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="r-c-model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("r-c-model Dashboard")
st.caption("Direction classification · Return magnitude regression · Combined confidence signal")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS)
    horizon = st.radio("Horizon (trading days)", [5, 10, 20], index=2, horizontal=True)
    st.divider()
    st.info("Train: 2015–2022\nVal: 2023\nTest: 2024")


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def get_ohlcv(tkr: str) -> pd.DataFrame:
    return pull(tkr, start="2015-01-01", end="2024-12-31")


@st.cache_data
def get_macro() -> pd.DataFrame:
    return pull_macro(start="2015-01-01", end="2024-12-31")


@st.cache_data
def get_features(tkr: str) -> pd.DataFrame:
    return compute_features(get_ohlcv(tkr), macro=get_macro())


@st.cache_resource
def load_models():
    artifacts = Path("artifacts")
    clf_files = sorted(artifacts.glob("*_xgb_clf.pkl"))
    reg_files = sorted(artifacts.glob("*_xgb_reg.pkl"))
    if not clf_files or not reg_files:
        return None, None, None, None
    clf_artifact = joblib.load(clf_files[-1])
    reg_artifact = joblib.load(reg_files[-1])
    return (
        clf_artifact["model"],
        reg_artifact["model"],
        clf_artifact["feature_names"],
        reg_artifact["feature_names"],
    )


@st.cache_data
def get_test_predictions(tkr: str, h: int) -> pd.DataFrame | None:
    clf, reg, clf_feats, _ = load_models()
    if clf is None:
        return None
    df = get_ohlcv(tkr)
    X = get_features(tkr)
    y_clf = make_labels(df["close"], horizon=h)
    y_reg = make_returns(df["close"], horizon=h)
    combined = X.join(y_clf.rename("label")).join(y_reg.rename("ret")).dropna()
    _, _, test = time_split(combined, TRAIN_END, VAL_END)
    if test.empty:
        return None
    X_test = test[clf_feats]
    dir_prob = clf.predict_proba(X_test)[:, 1]
    exp_ret = reg.predict(X_test)
    return pd.DataFrame(
        {
            "close": df.loc[test.index, "close"],
            "actual_label": test["label"],
            "actual_return": test["ret"],
            "dir_prob": dir_prob,
            "exp_return": exp_ret,
            "confidence": dir_prob * np.abs(exp_ret),
        },
        index=test.index,
    )


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📈 Price & Features", "🎯 Model Performance", "🔮 Confidence Signal"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Features
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{ticker} — Price & Technical Features")
    df = get_ohlcv(ticker)
    feats = get_features(ticker)

    years = sorted(df.index.year.unique())
    year_range = st.select_slider("Year range", options=years, value=(years[0], years[-1]))
    mask = (df.index.year >= year_range[0]) & (df.index.year <= year_range[1])
    df_v = df[mask]
    feats_v = feats[mask]

    # Bollinger Bands
    mid = df_v["close"].rolling(20).mean()
    std = df_v["close"].rolling(20).std()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.04,
    )

    # Price + Bollinger Bands
    fig.add_trace(go.Scatter(x=df_v.index, y=mid + 2 * std, name="BB Upper",
                             line=dict(color="rgba(150,150,150,0.3)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_v.index, y=mid - 2 * std, name="BB Lower",
                             fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
                             line=dict(color="rgba(150,150,150,0.3)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v["close"], name="Close",
                             line=dict(color="#1f77b4", width=1.5)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=feats_v.index, y=feats_v["rsi_14"], name="RSI 14",
                             line=dict(color="#ff7f0e", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.4, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.4, row=2, col=1)

    # MACD histogram + line
    colors = np.where(feats_v["macd_hist"] >= 0, "#2ca02c", "#d62728")
    fig.add_trace(go.Bar(x=feats_v.index, y=feats_v["macd_hist"], name="MACD Hist",
                         marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=feats_v.index, y=feats_v["macd_line"], name="MACD Line",
                             line=dict(color="#9467bd", width=1.2)), row=3, col=1)

    fig.update_layout(height=580, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Latest feature values (last 10 rows)"):
        st.dataframe(feats.dropna().tail(10).style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    clf, reg, clf_feats, reg_feats = load_models()

    if clf is None:
        st.warning("No trained artifacts found. Run `python -m src.train` first.")
    else:
        pred_df = get_test_predictions(ticker, horizon)

        if pred_df is None or pred_df.empty:
            st.warning("Not enough test data for this ticker and horizon.")
        else:
            st.subheader(f"{ticker} — Test Set Metrics (2024, {horizon}-day horizon)")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Classifier — Direction**")
                y_true = pred_df["actual_label"]
                y_prob = pred_df["dir_prob"]
                y_pred = (y_prob >= 0.5).astype(int)
                bl = brier_score_loss(y_true, np.full(len(y_true), y_true.mean()))
                bs = brier_score_loss(y_true, y_prob)
                clf_metrics = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "Brier Score": bs,
                    "Brier Skill": 1 - bs / bl,
                }
                st.dataframe(
                    pd.DataFrame.from_dict(clf_metrics, orient="index", columns=["Value"])
                    .style.format("{:.4f}"),
                    use_container_width=True,
                )

            with col2:
                st.markdown("**Regressor — Magnitude**")
                y_ret = pred_df["actual_return"]
                y_reg = pred_df["exp_return"]
                reg_metrics = {
                    "MAE": mean_absolute_error(y_ret, y_reg),
                    "RMSE": mean_squared_error(y_ret, y_reg) ** 0.5,
                    "R²": r2_score(y_ret, y_reg),
                    "Directional Accuracy": float((np.sign(y_reg) == np.sign(y_ret)).mean()),
                }
                st.dataframe(
                    pd.DataFrame.from_dict(reg_metrics, orient="index", columns=["Value"])
                    .style.format("{:.4f}"),
                    use_container_width=True,
                )

            st.divider()

            # Feature importance side by side
            st.subheader("Feature Importance — Classifier vs Regressor")
            col3, col4 = st.columns(2)

            with col3:
                fi_clf = pd.Series(clf.base.feature_importances_, index=clf_feats).sort_values()
                fig_fi = go.Figure(go.Bar(
                    x=fi_clf.values, y=fi_clf.index,
                    orientation="h", marker_color="#1f77b4",
                ))
                fig_fi.update_layout(title="Classifier", height=430,
                                     margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_fi, use_container_width=True)

            with col4:
                fi_reg = pd.Series(reg.feature_importances_, index=reg_feats).sort_values()
                fig_fi2 = go.Figure(go.Bar(
                    x=fi_reg.values, y=fi_reg.index,
                    orientation="h", marker_color="#ff7f0e",
                ))
                fig_fi2.update_layout(title="Regressor", height=430,
                                      margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_fi2, use_container_width=True)

            # Predicted vs actual return scatter
            st.subheader("Regressor — Predicted vs Actual Return")
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=pred_df["actual_return"], y=pred_df["exp_return"],
                mode="markers",
                marker=dict(color=pred_df["dir_prob"], colorscale="RdYlGn",
                            size=5, opacity=0.6, showscale=True,
                            colorbar=dict(title="Dir Prob")),
                text=pred_df.index.strftime("%Y-%m-%d"),
            ))
            lim = max(abs(pred_df["actual_return"]).max(), abs(pred_df["exp_return"]).max()) * 1.1
            fig_scat.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                               line=dict(color="gray", dash="dash"))
            fig_scat.update_layout(
                xaxis_title="Actual Return",
                yaxis_title="Predicted Return",
                height=380,
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig_scat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Confidence Signal
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    clf, reg, clf_feats, _ = load_models()

    if clf is None:
        st.warning("No trained artifacts found. Run `python -m src.train` first.")
    else:
        pred_df = get_test_predictions(ticker, horizon)

        if pred_df is None or pred_df.empty:
            st.warning("Not enough test data.")
        else:
            st.subheader(f"{ticker} — {horizon}-Day Forward Signal (Test: 2024)")

            # Latest signal metrics
            latest = pred_df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Date", str(pred_df.index[-1].date()))
            c2.metric("Direction Probability", f"{latest['dir_prob']:.3f}",
                      help="P(price higher in N days)")
            c3.metric("Expected Return", f"{latest['exp_return'] * 100:.2f}%",
                      help="Regressor predicted N-day return")
            c4.metric("Confidence Signal", f"{latest['confidence']:.4f}",
                      help="dir_prob × |expected_return| — high = strong, high-confidence move expected")

            st.divider()

            # Signal over time chart
            fig3 = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.38, 0.31, 0.31],
                vertical_spacing=0.06,
                subplot_titles=(
                    "Close Price",
                    "Direction Probability (classifier)",
                    "Confidence Signal = dir_prob × |expected_return|",
                ),
            )

            fig3.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["close"],
                name="Close", line=dict(color="#1f77b4"),
            ), row=1, col=1)

            fig3.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["dir_prob"],
                name="Dir Prob", line=dict(color="#2ca02c"),
            ), row=2, col=1)
            fig3.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

            bar_colors = np.where(pred_df["exp_return"] >= 0, "#2ca02c", "#d62728")
            fig3.add_trace(go.Bar(
                x=pred_df.index, y=pred_df["confidence"],
                name="Confidence", marker_color=bar_colors,
            ), row=3, col=1)

            fig3.update_layout(height=600, showlegend=False, margin=dict(t=40, b=20))
            fig3.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig3.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
            fig3.update_yaxes(title_text="Signal", row=3, col=1)
            st.plotly_chart(fig3, use_container_width=True)

            # Full table
            with st.expander("Full signal table"):
                display = pred_df[["close", "dir_prob", "exp_return", "confidence",
                                   "actual_label", "actual_return"]].copy()
                display["actual_label"] = display["actual_label"].map(
                    lambda x: "↑ Up" if x == 1.0 else "↓ Down"
                )
                st.dataframe(
                    display.sort_index(ascending=False).style.format({
                        "close": "{:.2f}",
                        "dir_prob": "{:.3f}",
                        "exp_return": "{:.2%}",
                        "confidence": "{:.4f}",
                        "actual_return": "{:.2%}",
                    }),
                    use_container_width=True,
                )
