"""
Kronos + Reddit Signal Dashboard

Run with:
    venv\Scripts\python.exe -m streamlit run dashboard/app.py

Sections:
  1. Signal Table    — today's signals across all tickers with labels + scores
  2. Ticker Detail   — deep-dive into one ticker: Kronos, sentiment, technicals
  3. Pipeline Health — recent run metrics, error rates, duration trends
  4. Accuracy        — Kronos directional accuracy vs actual next-day returns
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when run via `streamlit run dashboard/app.py`
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # loads .env and injects kronos_src into sys.path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from storage.db import init_db
from storage.store import SignalStore

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Kronos Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Shared state                                                        #
# ------------------------------------------------------------------ #

@st.cache_resource
def get_store() -> SignalStore:
    init_db()
    return SignalStore()


LABEL_COLORS = {
    "STRONG_BUY":  "#00c853",
    "BUY":         "#69f0ae",
    "HOLD":        "#ffd740",
    "SELL":        "#ff6d00",
    "STRONG_SELL": "#d50000",
}

LABEL_ORDER = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]


def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#888888")
    return f'<span style="background:{color};color:#000;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em">{label}</span>'


# ------------------------------------------------------------------ #
#  Data loaders (cached per run)                                      #
# ------------------------------------------------------------------ #

@st.cache_data(ttl=300)
def load_signals(days: int = 30) -> pd.DataFrame:
    store = get_store()
    rows = store.get_recent_signals(days=days)
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        data.append({
            "ticker": r.ticker,
            "date": r.signal_date,
            "label": r.classifier_label,
            "confluence": r.confluence_score,
            "direction": r.kronos_direction,
            "kronos_conf": r.kronos_confidence,
            "kronos_pct": r.kronos_pct_change,
            "reddit_sentiment": r.reddit_sentiment,
            "reddit_score": r.reddit_score,
            "reddit_posts": r.reddit_post_count,
            "rsi": r.rsi_14,
            "macd": r.macd_signal,
            "bb": r.bb_position,
            "adx": r.adx_14,
            "vol_ratio": r.avg_volume_ratio,
            "price_at_signal": r.price_at_signal,
            "price_next_day": r.price_next_day,
            "reasoning": r.classifier_reasoning,
        })
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def load_pipeline_runs(limit: int = 30) -> pd.DataFrame:
    store = get_store()
    runs = store.get_pipeline_runs(limit=limit)
    if not runs:
        return pd.DataFrame()
    data = []
    for r in runs:
        data.append({
            "run_at": r.run_at,
            "attempted": r.tickers_attempted,
            "succeeded": r.tickers_succeeded,
            "failed": r.tickers_failed,
            "kronos_errors": r.kronos_errors,
            "reddit_errors": r.reddit_errors,
            "catalyst_dead": bool(r.catalyst_api_dead),
            "catalyst_degraded": bool(r.catalyst_api_degraded),
            "duration_s": r.duration_seconds,
            "notes": r.notes,
        })
    df = pd.DataFrame(data)
    df["run_at"] = pd.to_datetime(df["run_at"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_accuracy() -> dict:
    store = get_store()
    return store.get_accuracy_stats()


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("Kronos Signal")
    st.caption("Kronos + Reddit confluence engine")
    st.divider()

    lookback_days = st.slider("Lookback (days)", min_value=1, max_value=90, value=30)
    st.divider()

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    accuracy = load_accuracy()
    if accuracy["total"] > 0:
        st.metric(
            "Kronos accuracy",
            f"{accuracy['accuracy']:.0%}",
            f"{accuracy['correct']}/{accuracy['total']} signals",
        )
    else:
        st.caption("No accuracy data yet (need price_next_day backfill)")

    st.divider()
    st.caption("Run pipeline:")
    st.code("python main.py --once", language="bash")
    st.caption("Start scheduler:")
    st.code("python main.py --schedule", language="bash")


# ------------------------------------------------------------------ #
#  Main content                                                        #
# ------------------------------------------------------------------ #

df = load_signals(days=lookback_days)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Signal Table",
    "🔍 Ticker Detail",
    "⚙️ Pipeline Health",
    "🎯 Accuracy",
])

# ================================================================== #
#  Tab 1: Signal Table                                                #
# ================================================================== #

with tab1:
    st.header("Signal Table")

    if df.empty:
        st.info("No signals yet. Run the pipeline first: `python main.py --once`")
    else:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            label_filter = st.multiselect(
                "Filter by label",
                options=LABEL_ORDER,
                default=LABEL_ORDER,
            )
        with col2:
            direction_filter = st.multiselect(
                "Kronos direction",
                options=["bullish", "bearish", "neutral"],
                default=["bullish", "bearish", "neutral"],
            )
        with col3:
            min_confluence = st.slider("Min confluence score", 0.0, 1.0, 0.0, 0.01)

        filtered = df[
            df["label"].isin(label_filter) &
            df["direction"].isin(direction_filter) &
            (df["confluence"] >= min_confluence)
        ].copy()

        # Summary metrics
        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        today_df = filtered[filtered["date"] == today_str]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total signals", len(filtered))
        m2.metric("Today's signals", len(today_df))
        m3.metric(
            "Strong buys today",
            len(today_df[today_df["label"] == "STRONG_BUY"])
        )
        m4.metric(
            "Strong sells today",
            len(today_df[today_df["label"] == "STRONG_SELL"])
        )

        st.divider()

        # Label distribution bar chart
        if not filtered.empty:
            label_counts = (
                filtered.groupby(["date", "label"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                label_counts,
                x="date",
                y="count",
                color="label",
                color_discrete_map=LABEL_COLORS,
                category_orders={"label": LABEL_ORDER},
                title="Signal labels over time",
                height=300,
            )
            fig.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Signal table
        display_cols = ["date", "ticker", "label", "confluence", "direction",
                        "kronos_conf", "reddit_sentiment", "rsi", "reddit_posts"]
        display_df = filtered[display_cols].copy()
        display_df["confluence"] = display_df["confluence"].round(3)
        display_df["kronos_conf"] = display_df["kronos_conf"].round(2)
        display_df["rsi"] = display_df["rsi"].round(1)

        st.dataframe(
            display_df.sort_values(["date", "confluence"], ascending=[False, False]),
            use_container_width=True,
            height=400,
            column_config={
                "confluence": st.column_config.ProgressColumn(
                    "Confluence", min_value=0, max_value=1, format="%.3f"
                ),
                "kronos_conf": st.column_config.ProgressColumn(
                    "Kronos conf", min_value=0, max_value=1, format="%.2f"
                ),
            },
        )


# ================================================================== #
#  Tab 2: Ticker Detail                                               #
# ================================================================== #

with tab2:
    st.header("Ticker Detail")

    if df.empty:
        st.info("No signals yet.")
    else:
        tickers = sorted(df["ticker"].unique().tolist())
        selected_ticker = st.selectbox("Select ticker", tickers)

        ticker_df = df[df["ticker"] == selected_ticker].sort_values("date", ascending=False)

        if ticker_df.empty:
            st.warning(f"No signals for {selected_ticker}")
        else:
            latest = ticker_df.iloc[0]

            # Header row
            col_label, col_conf, col_dir = st.columns(3)
            with col_label:
                color = LABEL_COLORS.get(latest["label"], "#888")
                st.markdown(
                    f"<div style='background:{color};padding:12px;border-radius:8px;"
                    f"text-align:center;font-size:1.4em;font-weight:bold'>"
                    f"{latest['label']}</div>",
                    unsafe_allow_html=True,
                )
            with col_conf:
                st.metric("Confluence score", f"{latest['confluence']:.3f}")
            with col_dir:
                st.metric(
                    "Kronos direction",
                    latest["direction"] or "—",
                    f"{(latest['kronos_pct'] or 0)*100:+.1f}% predicted",
                )

            st.divider()

            # Three columns: Kronos / Reddit / Technicals
            c1, c2, c3 = st.columns(3)

            with c1:
                st.subheader("Kronos")
                st.metric("Confidence", f"{latest['kronos_conf']:.0%}" if latest['kronos_conf'] else "—")
                st.metric("Predicted move", f"{(latest['kronos_pct'] or 0)*100:+.2f}%")

            with c2:
                st.subheader("Reddit")
                st.metric("Sentiment", latest["reddit_sentiment"] or "—")
                st.metric("Score", f"{latest['reddit_score']:+.2f}" if latest['reddit_score'] is not None else "—")
                st.metric("Posts", int(latest["reddit_posts"] or 0))

            with c3:
                st.subheader("Technicals")
                st.metric("RSI (14)", f"{latest['rsi']:.1f}" if latest['rsi'] else "—")
                st.metric("MACD", latest["macd"] or "—")
                st.metric("Bollinger", latest["bb"] or "—")
                st.metric("ADX", f"{latest['adx']:.1f}" if latest['adx'] else "—")
                st.metric("Vol ratio", f"{latest['vol_ratio']:.2f}x" if latest['vol_ratio'] else "—")

            # Reasoning
            if latest["reasoning"]:
                st.divider()
                st.subheader("Reasoning")
                for line in str(latest["reasoning"]).split("\n"):
                    if line.strip():
                        st.markdown(f"- {line.strip()}")

            # Confluence history chart
            if len(ticker_df) > 1:
                st.divider()
                st.subheader("Confluence history")
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["confluence"],
                    mode="lines+markers",
                    name="Confluence",
                    line=dict(color="#4fc3f7", width=2),
                    marker=dict(size=6),
                ))
                hist_fig.add_hline(y=0.72, line_dash="dot", line_color="#00c853",
                                   annotation_text="STRONG_BUY")
                hist_fig.add_hline(y=0.58, line_dash="dot", line_color="#69f0ae",
                                   annotation_text="BUY")
                hist_fig.add_hline(y=0.42, line_dash="dot", line_color="#ff6d00",
                                   annotation_text="SELL")
                hist_fig.add_hline(y=0.28, line_dash="dot", line_color="#d50000",
                                   annotation_text="STRONG_SELL")
                hist_fig.update_layout(
                    yaxis=dict(range=[0, 1], title="Confluence"),
                    xaxis_title="Date",
                    height=300,
                    margin=dict(t=20, b=0),
                )
                st.plotly_chart(hist_fig, use_container_width=True)


# ================================================================== #
#  Tab 3: Pipeline Health                                             #
# ================================================================== #

with tab3:
    st.header("Pipeline Health")

    runs_df = load_pipeline_runs(limit=30)

    if runs_df.empty:
        st.info("No pipeline runs recorded yet.")
    else:
        latest_run = runs_df.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last run", latest_run["run_at"].strftime("%Y-%m-%d %H:%M"))
        m2.metric(
            "Success rate",
            f"{latest_run['succeeded']}/{latest_run['attempted']}",
        )
        m3.metric("Duration", f"{latest_run['duration_s']:.1f}s")
        m4.metric(
            "Errors (Kronos/Reddit)",
            f"{latest_run['kronos_errors']} / {latest_run['reddit_errors']}",
        )

        if latest_run["catalyst_dead"]:
            st.error("Catalyst API is DOWN — Claude integration not working")
        elif latest_run["catalyst_degraded"]:
            st.warning("Catalyst API is degraded (>50% of calls failing)")

        st.divider()

        # Duration trend
        dur_fig = px.line(
            runs_df.sort_values("run_at"),
            x="run_at",
            y="duration_s",
            title="Pipeline duration (seconds)",
            markers=True,
            height=250,
        )
        dur_fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(dur_fig, use_container_width=True)

        # Success rate trend
        runs_df["success_rate"] = runs_df["succeeded"] / runs_df["attempted"].clip(lower=1)
        sr_fig = px.line(
            runs_df.sort_values("run_at"),
            x="run_at",
            y="success_rate",
            title="Ticker success rate",
            markers=True,
            height=250,
        )
        sr_fig.update_layout(yaxis=dict(range=[0, 1]), margin=dict(t=40, b=0))
        st.plotly_chart(sr_fig, use_container_width=True)

        # Error breakdown
        error_cols = ["kronos_errors", "reddit_errors"]
        err_df = runs_df[["run_at"] + error_cols].sort_values("run_at")
        err_fig = px.bar(
            err_df.melt(id_vars="run_at", value_vars=error_cols,
                        var_name="error_type", value_name="count"),
            x="run_at",
            y="count",
            color="error_type",
            title="Errors per run",
            height=250,
            barmode="stack",
        )
        err_fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(err_fig, use_container_width=True)

        # Raw runs table
        with st.expander("Raw run log"):
            st.dataframe(runs_df.sort_values("run_at", ascending=False), use_container_width=True)


# ================================================================== #
#  Tab 4: Accuracy                                                    #
# ================================================================== #

with tab4:
    st.header("Kronos Accuracy")
    st.caption(
        "Directional accuracy: did Kronos correctly predict bullish/bearish/neutral? "
        "Requires price_next_day to be backfilled — see TODOS.md."
    )

    accuracy = load_accuracy()

    if accuracy["total"] == 0:
        st.info(
            "No accuracy data yet. Signals need `price_next_day` filled in "
            "before accuracy can be computed. This happens automatically after "
            "the next-day price backfill job runs (see TODOS.md)."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total evaluated", accuracy["total"])
        c2.metric("Correct direction", accuracy["correct"])
        c3.metric(
            "Accuracy",
            f"{accuracy['accuracy']:.1%}",
            f"{'above' if accuracy['accuracy'] > 0.5 else 'at or below'} random (50%)",
        )

        # Accuracy gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy["accuracy"] * 100,
            title={"text": "Directional accuracy (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4fc3f7"},
                "steps": [
                    {"range": [0, 50], "color": "#d50000"},
                    {"range": [50, 65], "color": "#ffd740"},
                    {"range": [65, 100], "color": "#00c853"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Per-ticker accuracy
        if not df.empty:
            evaluated = df[df["price_next_day"].notna() & df["price_at_signal"].notna()].copy()
            if not evaluated.empty:
                evaluated["actual_dir"] = evaluated.apply(
                    lambda r: "bullish" if r["price_next_day"] > r["price_at_signal"]
                    else ("bearish" if r["price_next_day"] < r["price_at_signal"] else "neutral"),
                    axis=1,
                )
                evaluated["correct"] = evaluated["direction"] == evaluated["actual_dir"]
                per_ticker = (
                    evaluated.groupby("ticker")["correct"]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "correct", "count": "total"})
                    .assign(accuracy=lambda d: d["correct"] / d["total"])
                    .sort_values("accuracy", ascending=False)
                    .reset_index()
                )
                st.subheader("Per-ticker accuracy")
                st.dataframe(per_ticker, use_container_width=True)
