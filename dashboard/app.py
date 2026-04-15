"""
Kronos + Reddit Signal Dashboard

Run with:
    venv\Scripts\python.exe -m streamlit run dashboard/app.py

Sections:
  1. Signal Table    — today's signals across all tickers with labels + scores
  2. Ticker Detail   — deep-dive into one ticker: Kronos, sentiment, technicals
  3. Pipeline Health — recent run metrics, error rates, duration trends
  4. Accuracy        — Kronos directional accuracy vs actual next-day returns
  5. Paper Trading   — Alpaca paper portfolio, open positions, trade history
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

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
    page_title="Kronos Signal",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Design system injection                                             #
# ------------------------------------------------------------------ #

COLORS = {
    "bg":          "#020617",
    "surface":     "#0F172A",
    "surface2":    "#1E293B",
    "border":      "#334155",
    "muted":       "#475569",
    "text":        "#F8FAFC",
    "text_muted":  "#94A3B8",
    "green":       "#22C55E",
    "green_dim":   "#86EFAC",
    "amber":       "#FCD34D",
    "orange":      "#F97316",
    "red":         "#EF4444",
    "blue":        "#38BDF8",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {COLORS['surface']} !important;
    border-right: 1px solid {COLORS['border']};
}}
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {COLORS['text_muted']};
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['surface']} !important;
    border-bottom: 1px solid {COLORS['border']};
    gap: 4px;
    padding: 4px 4px 0;
    border-radius: 8px 8px 0 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {COLORS['text_muted']} !important;
    border-radius: 6px 6px 0 0;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 16px;
    border: none !important;
    transition: color 150ms ease;
}}
.stTabs [aria-selected="true"] {{
    background: {COLORS['bg']} !important;
    color: {COLORS['text']} !important;
    border-bottom: 2px solid {COLORS['green']} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 24px;
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {COLORS['surface']} !important;
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 16px 20px !important;
}}
[data-testid="metric-container"] label {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {COLORS['text']} !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    overflow: hidden;
}}

/* Buttons */
.stButton button {{
    background: {COLORS['surface2']} !important;
    color: {COLORS['text']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: border-color 150ms ease, background 150ms ease !important;
}}
.stButton button:hover {{
    border-color: {COLORS['green']} !important;
    background: {COLORS['surface']} !important;
}}

/* Dividers */
hr {{
    border-color: {COLORS['border']} !important;
    margin: 20px 0 !important;
}}

/* Section headers */
h2 {{
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: {COLORS['text']} !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 16px !important;
}}
h3 {{
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: {COLORS['text_muted']} !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
}}

/* Info/warning/error boxes */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border: 1px solid {COLORS['border']} !important;
    font-size: 0.85rem !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
    background: {COLORS['surface']} !important;
}}

/* Sliders */
[data-testid="stSlider"] label {{
    font-size: 0.8rem !important;
    color: {COLORS['text_muted']} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}

/* Selectbox / multiselect */
[data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] label {{
    font-size: 0.8rem !important;
    color: {COLORS['text_muted']} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}

/* Code blocks in sidebar */
code {{
    font-size: 0.78rem !important;
    background: {COLORS['surface2']} !important;
    color: {COLORS['green']} !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}}
</style>
""", unsafe_allow_html=True)

# Shared plotly layout defaults
PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(t=40, b=20, l=16, r=16),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
)

# ------------------------------------------------------------------ #
#  Shared state                                                        #
# ------------------------------------------------------------------ #

@st.cache_resource
def get_store() -> SignalStore:
    init_db()
    return SignalStore()


LABEL_COLORS = {
    "STRONG_BUY":  COLORS["green"],
    "BUY":         COLORS["green_dim"],
    "HOLD":        COLORS["amber"],
    "SELL":        COLORS["orange"],
    "STRONG_SELL": COLORS["red"],
}

LABEL_TEXT_COLORS = {
    "STRONG_BUY":  "#000",
    "BUY":         "#000",
    "HOLD":        "#000",
    "SELL":        "#000",
    "STRONG_SELL": "#fff",
}

LABEL_ORDER = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]


def label_badge(label: str) -> str:
    bg = LABEL_COLORS.get(label, COLORS["muted"])
    fg = LABEL_TEXT_COLORS.get(label, "#fff")
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:5px;'
        f'font-weight:600;font-size:0.78rem;letter-spacing:0.04em">{label}</span>'
    )


def card(content_html: str) -> str:
    return (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:10px;padding:16px 20px;margin-bottom:8px">{content_html}</div>'
    )


# ------------------------------------------------------------------ #
#  Data loaders                                                        #
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
    st.markdown(
        f'<div style="font-size:1.25rem;font-weight:700;color:{COLORS["text"]};'
        f'letter-spacing:-0.02em;margin-bottom:2px">Kronos Signal</div>'
        f'<div style="font-size:0.75rem;color:{COLORS["text_muted"]};margin-bottom:16px">'
        f'Kronos + Reddit confluence engine</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    lookback_days = st.slider("Lookback (days)", min_value=1, max_value=90, value=30)
    st.divider()

    if st.button("Refresh data", use_container_width=True):
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
        st.markdown(
            f'<span style="font-size:0.78rem;color:{COLORS["text_muted"]}">'
            f'No accuracy data yet</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        f'<span style="font-size:0.75rem;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:0.05em">Run pipeline</span>',
        unsafe_allow_html=True,
    )
    st.code("python main.py --once", language="bash")
    st.markdown(
        f'<span style="font-size:0.75rem;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:0.05em">Scheduler</span>',
        unsafe_allow_html=True,
    )
    st.code("python main.py --schedule", language="bash")


# ------------------------------------------------------------------ #
#  Main content                                                        #
# ------------------------------------------------------------------ #

df = load_signals(days=lookback_days)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Signal Table",
    "Ticker Detail",
    "Pipeline Health",
    "Accuracy",
    "Paper Trading",
])

# ================================================================== #
#  Tab 1: Signal Table                                                #
# ================================================================== #

with tab1:
    st.header("Signal Table")

    if df.empty:
        st.info("No signals yet. Run the pipeline: `python main.py --once`")
    else:
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

        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        today_df = filtered[filtered["date"] == today_str]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total signals", len(filtered))
        m2.metric("Today's signals", len(today_df))
        m3.metric("Strong buys today", len(today_df[today_df["label"] == "STRONG_BUY"]))
        m4.metric("Strong sells today", len(today_df[today_df["label"] == "STRONG_SELL"]))

        st.divider()

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
                height=260,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11),
                ),
                bargap=0.15,
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        display_cols = ["date", "ticker", "label", "confluence", "direction",
                        "kronos_conf", "reddit_sentiment", "rsi", "reddit_posts"]
        display_df = filtered[display_cols].copy()
        display_df["confluence"] = display_df["confluence"].round(3)
        display_df["kronos_conf"] = display_df["kronos_conf"].round(2)
        display_df["rsi"] = display_df["rsi"].round(1)

        st.dataframe(
            display_df.sort_values(["date", "confluence"], ascending=[False, False]),
            use_container_width=True,
            height=380,
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

            # Signal banner
            color = LABEL_COLORS.get(latest["label"], COLORS["muted"])
            text_color = LABEL_TEXT_COLORS.get(latest["label"], "#fff")
            st.markdown(
                f'<div style="background:{color};color:{text_color};padding:14px 20px;'
                f'border-radius:10px;display:flex;align-items:center;justify-content:space-between;'
                f'margin-bottom:20px">'
                f'<span style="font-size:1.3rem;font-weight:700;letter-spacing:0.02em">'
                f'{latest["label"]}</span>'
                f'<span style="font-size:0.85rem;opacity:0.75">{selected_ticker} &middot; {latest["date"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_conf, col_dir, col_price = st.columns(3)
            with col_conf:
                st.metric("Confluence score", f"{latest['confluence']:.3f}")
            with col_dir:
                st.metric(
                    "Kronos direction",
                    latest["direction"] or "—",
                    f"{(latest['kronos_pct'] or 0)*100:+.1f}% predicted",
                )
            with col_price:
                st.metric(
                    "Price at signal",
                    f"${latest['price_at_signal']:,.2f}" if latest["price_at_signal"] else "—",
                )

            st.divider()

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

            if latest["reasoning"]:
                st.divider()
                st.subheader("Reasoning")
                lines = [l.strip() for l in str(latest["reasoning"]).split("\n") if l.strip()]
                bullets_html = "".join(
                    f'<div style="display:flex;gap:8px;margin-bottom:6px;font-size:0.85rem;'
                    f'color:{COLORS["text_muted"]}">'
                    f'<span style="color:{COLORS["green"]};flex-shrink:0">&#8250;</span>'
                    f'<span>{line}</span></div>'
                    for line in lines
                )
                st.markdown(
                    card(bullets_html),
                    unsafe_allow_html=True,
                )

            if len(ticker_df) > 1:
                st.divider()
                st.subheader("Confluence history")
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["confluence"],
                    mode="lines+markers",
                    name="Confluence",
                    line=dict(color=COLORS["blue"], width=2),
                    marker=dict(size=5, color=COLORS["blue"]),
                    fill="tozeroy",
                    fillcolor=f'{COLORS["blue"]}18',
                ))
                # Threshold lines matching current thresholds
                for y, color, label in [
                    (0.68, COLORS["green"],      "STRONG BUY"),
                    (0.54, COLORS["green_dim"],  "BUY"),
                    (0.46, COLORS["orange"],     "SELL"),
                    (0.32, COLORS["red"],        "STRONG SELL"),
                ]:
                    hist_fig.add_hline(
                        y=y, line_dash="dot", line_color=color, line_width=1,
                        annotation_text=label,
                        annotation_font=dict(size=10, color=color),
                        annotation_position="right",
                    )
                hist_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    yaxis=dict(range=[0, 1], title="Confluence",
                               gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
                    xaxis_title=None,
                    height=280,
                    showlegend=False,
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
        m2.metric("Success rate", f"{latest_run['succeeded']}/{latest_run['attempted']}")
        m3.metric("Duration", f"{latest_run['duration_s']:.1f}s")
        m4.metric(
            "Errors (Kronos / Reddit)",
            f"{latest_run['kronos_errors']} / {latest_run['reddit_errors']}",
        )

        if latest_run["catalyst_dead"]:
            st.error("Catalyst API is DOWN")
        elif latest_run["catalyst_degraded"]:
            st.warning("Catalyst API is degraded")

        st.divider()

        sorted_runs = runs_df.sort_values("run_at")

        dur_fig = px.line(
            sorted_runs, x="run_at", y="duration_s",
            title="Pipeline duration (seconds)",
            markers=True, height=220,
        )
        dur_fig.update_traces(
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=5, color=COLORS["blue"]),
        )
        dur_fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(dur_fig, use_container_width=True)

        sorted_runs = sorted_runs.copy()
        sorted_runs["success_rate"] = sorted_runs["succeeded"] / sorted_runs["attempted"].clip(lower=1)
        sr_fig = px.line(
            sorted_runs, x="run_at", y="success_rate",
            title="Ticker success rate",
            markers=True, height=220,
        )
        sr_fig.update_traces(
            line=dict(color=COLORS["green"], width=2),
            marker=dict(size=5, color=COLORS["green"]),
        )
        sr_fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(range=[0, 1], gridcolor=COLORS["border"]))
        st.plotly_chart(sr_fig, use_container_width=True)

        err_df = runs_df[["run_at", "kronos_errors", "reddit_errors"]].sort_values("run_at")
        err_fig = px.bar(
            err_df.melt(id_vars="run_at", value_vars=["kronos_errors", "reddit_errors"],
                        var_name="error_type", value_name="count"),
            x="run_at", y="count", color="error_type",
            color_discrete_map={
                "kronos_errors": COLORS["orange"],
                "reddit_errors": COLORS["red"],
            },
            title="Errors per run",
            height=220,
            barmode="stack",
        )
        err_fig.update_layout(**PLOTLY_LAYOUT)
        err_fig.update_traces(marker_line_width=0)
        st.plotly_chart(err_fig, use_container_width=True)

        with st.expander("Raw run log"):
            st.dataframe(runs_df.sort_values("run_at", ascending=False), use_container_width=True)


# ================================================================== #
#  Tab 4: Accuracy                                                    #
# ================================================================== #

with tab4:
    st.header("Kronos Accuracy")
    st.markdown(
        f'<p style="font-size:0.82rem;color:{COLORS["text_muted"]};margin-bottom:20px">'
        f'Directional accuracy — did Kronos correctly predict bullish / bearish / neutral? '
        f'Requires price_next_day to be backfilled (runs nightly).</p>',
        unsafe_allow_html=True,
    )

    accuracy = load_accuracy()

    if accuracy["total"] == 0:
        st.info(
            "No accuracy data yet. The next-day price backfill job populates this "
            "automatically after market close."
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

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy["accuracy"] * 100,
            number=dict(suffix="%", font=dict(size=40, color=COLORS["text"])),
            title={"text": "Directional accuracy", "font": {"size": 14, "color": COLORS["text_muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["border"]},
                "bar": {"color": COLORS["blue"], "thickness": 0.25},
                "bgcolor": COLORS["surface2"],
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 50],  "color": f'{COLORS["red"]}33'},
                    {"range": [50, 65], "color": f'{COLORS["amber"]}33'},
                    {"range": [65, 100],"color": f'{COLORS["green"]}33'},
                ],
                "threshold": {
                    "line": {"color": COLORS["text_muted"], "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor=COLORS["surface"],
            font=dict(family="Inter, sans-serif", color=COLORS["text"]),
            height=280,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

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


# ================================================================== #
#  Tab 5: Paper Trading                                               #
# ================================================================== #

with tab5:
    st.header("Paper Trading")

    try:
        from trading.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
    except Exception as e:
        st.error(f"Failed to load AlpacaTrader: {e}")
        trader = None

    if trader is None or not trader.enabled:
        st.info(
            "Paper trading is disabled. Add to `.env`:\n\n"
            "```\n"
            "ALPACA_API_KEY=your_key_id\n"
            "ALPACA_SECRET_KEY=your_secret\n"
            "PAPER_TRADING_ENABLED=true\n"
            "POSITION_SIZE_USD=1000\n"
            "```"
        )
    else:
        portfolio = trader.get_portfolio_summary()

        if "error" in portfolio:
            st.error(f"Alpaca API error: {portfolio['error']}")
        else:
            st.subheader("Portfolio")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio value", f"${portfolio.get('portfolio_value', 0):,.2f}")
            c2.metric("Equity", f"${portfolio.get('equity', 0):,.2f}")
            c3.metric("Cash", f"${portfolio.get('cash', 0):,.2f}")
            c4.metric("Buying power", f"${portfolio.get('buying_power', 0):,.2f}")

        st.divider()
        st.subheader("Open Positions")
        positions = portfolio.get("positions", [])
        if not positions:
            st.markdown(
                f'<p style="color:{COLORS["text_muted"]};font-size:0.85rem">No open positions.</p>',
                unsafe_allow_html=True,
            )
        else:
            pos_df = pd.DataFrame(positions)
            pos_df["unrealized_pnl_pct"] = pos_df["unrealized_pnl_pct"].map(lambda x: f"{x:.2f}%")
            pos_df["unrealized_pnl"] = pos_df["unrealized_pnl"].map(lambda x: f"${x:,.2f}")
            pos_df["market_value"] = pos_df["market_value"].map(lambda x: f"${x:,.2f}")
            pos_df["avg_entry_price"] = pos_df["avg_entry_price"].map(lambda x: f"${x:,.2f}")
            pos_df["current_price"] = pos_df["current_price"].map(lambda x: f"${x:,.2f}")
            st.dataframe(
                pos_df.rename(columns={
                    "ticker": "Ticker", "qty": "Qty",
                    "market_value": "Mkt Value", "avg_entry_price": "Entry",
                    "current_price": "Current",
                    "unrealized_pnl": "Unrealized P&L", "unrealized_pnl_pct": "P&L %",
                }),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Trade Statistics")
        store = get_store()
        stats = store.get_trade_stats()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total trades", stats["total_trades"])
        sc2.metric("Open positions", stats["open_positions"])
        sc3.metric("Closed trades", stats["closed_trades"])
        total_pnl = stats.get("total_pnl_usd", 0.0) or 0.0
        sc4.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"${total_pnl:,.2f}")

        if stats["win_rate"] is not None:
            wc1, wc2 = st.columns(2)
            wc1.metric("Win rate", f"{stats['win_rate']:.1%}")
            wc2.metric("Avg P&L per trade", f"${stats['avg_pnl_usd']:,.2f}")

        st.divider()
        st.subheader("Trade History")
        trades = store.get_trades(limit=100)
        if not trades:
            st.markdown(
                f'<p style="color:{COLORS["text_muted"]};font-size:0.85rem">No trades recorded yet.</p>',
                unsafe_allow_html=True,
            )
        else:
            trade_df = pd.DataFrame(trades)
            display_cols = [
                "ticker", "signal_date", "signal_label", "side",
                "qty", "entry_price", "exit_price", "pnl_usd", "pnl_pct", "status", "opened_at",
            ]
            display_cols = [c for c in display_cols if c in trade_df.columns]
            trade_df = trade_df[display_cols].copy()

            st.dataframe(
                trade_df.rename(columns={
                    "ticker": "Ticker", "signal_date": "Date",
                    "signal_label": "Signal", "side": "Side", "qty": "Qty",
                    "entry_price": "Entry", "exit_price": "Exit",
                    "pnl_usd": "P&L ($)", "pnl_pct": "P&L %",
                    "status": "Status", "opened_at": "Opened",
                }),
                use_container_width=True,
            )

            closed_df = trade_df[trade_df["status"] == "closed"].copy() if "status" in trade_df.columns else pd.DataFrame()
            if not closed_df.empty and "pnl_usd" in closed_df.columns:
                closed_df = closed_df.dropna(subset=["pnl_usd"]).copy()
                if not closed_df.empty:
                    closed_df["cumulative_pnl"] = closed_df["pnl_usd"].cumsum()
                    pnl_fig = px.line(
                        closed_df.reset_index(drop=True),
                        y="cumulative_pnl",
                        title="Cumulative P&L",
                        labels={"index": "Trade #", "cumulative_pnl": "P&L ($)"},
                        height=260,
                    )
                    final_pnl = closed_df["cumulative_pnl"].iloc[-1]
                    line_color = COLORS["green"] if final_pnl >= 0 else COLORS["red"]
                    pnl_fig.update_traces(line=dict(color=line_color, width=2))
                    pnl_fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(pnl_fig, use_container_width=True)
