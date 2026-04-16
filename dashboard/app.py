"""
Kronos + Reddit Signal Dashboard

Run with:
    venv\Scripts\python.exe -m streamlit run dashboard/app.py

Sections:
  1. Discovery      — active discovered tickers by priority (HIGH / MEDIUM / NEW / COOLING)
  2. Ticker Detail  — deep-dive: company info, Reddit context, Claude analysis, model inputs
  3. Pipeline Health — recent run metrics, error rates, duration trends
  4. Accuracy        — Kronos directional accuracy vs actual next-day returns
  5. Paper Trading   — Alpaca paper portfolio, open positions, trade history
"""
import json
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
    "purple":      "#A78BFA",
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
)

# Grid style to apply per-axis when customising individual charts
_AXIS_STYLE = dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])

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

PRIORITY_COLORS = {
    "HIGH":    COLORS["green"],
    "MEDIUM":  COLORS["blue"],
    "NEW":     COLORS["amber"],
    "COOLING": COLORS["muted"],
    "DROPPED": COLORS["red"],
}

PRIORITY_BG = {
    "HIGH":    "#052e16",
    "MEDIUM":  "#0c1a2e",
    "NEW":     "#2d1f00",
    "COOLING": "#1a1a1a",
    "DROPPED": "#2d0000",
}


def label_badge(label: str) -> str:
    bg = LABEL_COLORS.get(label, COLORS["muted"])
    fg = LABEL_TEXT_COLORS.get(label, "#fff")
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:5px;'
        f'font-weight:600;font-size:0.78rem;letter-spacing:0.04em">{label}</span>'
    )


def priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, COLORS["muted"])
    icons = {"HIGH": "🔥", "MEDIUM": "📈", "NEW": "✨", "COOLING": "❄️", "DROPPED": "💀"}
    icon = icons.get(priority, "")
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.72rem;'
        f'letter-spacing:0.05em">{icon} {priority}</span>'
    )


def card(content_html: str, border_color: str = None) -> str:
    border = border_color or COLORS["border"]
    return (
        f'<div style="background:{COLORS["surface"]};border:1px solid {border};'
        f'border-radius:10px;padding:16px 20px;margin-bottom:8px">{content_html}</div>'
    )


def quality_bar(score: int, max_score: int = 10) -> str:
    pct = max(0, min(100, (score or 0) / max_score * 100))
    color = COLORS["green"] if pct >= 70 else COLORS["amber"] if pct >= 50 else COLORS["red"]
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:{COLORS["surface2"]};border-radius:4px;height:6px">'
        f'<div style="width:{pct:.0f}%;background:{color};border-radius:4px;height:6px"></div>'
        f'</div>'
        f'<span style="font-size:0.75rem;color:{color};font-weight:600;min-width:24px">{score}/10</span>'
        f'</div>'
    )


# ------------------------------------------------------------------ #
#  Data loaders                                                        #
# ------------------------------------------------------------------ #

@st.cache_data(ttl=300)
def load_discovered_tickers(limit: int = 200) -> list:
    store = get_store()
    return store.get_all_discovered_tickers(limit=limit)


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
        f'Reddit discovery + confluence engine</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    lookback_days = st.slider("Signal lookback (days)", min_value=1, max_value=90, value=30)
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
        f'text-transform:uppercase;letter-spacing:0.05em">Discovery + pipeline</span>',
        unsafe_allow_html=True,
    )
    st.code("python main.py --once", language="bash")
    st.markdown(
        f'<span style="font-size:0.75rem;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:0.05em">Discovery only</span>',
        unsafe_allow_html=True,
    )
    st.code("python main.py --discover", language="bash")
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
discovered = load_discovered_tickers()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Discovery",
    "Ticker Detail",
    "Pipeline Health",
    "Accuracy",
    "Paper Trading",
])


# ================================================================== #
#  Tab 1: Discovery                                                    #
# ================================================================== #

with tab1:
    st.header("Reddit Discovery Watchlist")

    if not discovered:
        st.info(
            "No tickers discovered yet. Run discovery: `python main.py --discover`"
        )
    else:
        # Summary metrics
        active = [t for t in discovered if t.get("status") == "active"]
        high = [t for t in active if t.get("priority") == "HIGH"]
        medium = [t for t in active if t.get("priority") == "MEDIUM"]
        new = [t for t in active if t.get("priority") == "NEW"]
        cooling = [t for t in discovered if t.get("priority") == "COOLING"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Active tickers", len(active))
        m2.metric("HIGH priority", len(high), delta=f"🔥 {len(high)} trending")
        m3.metric("MEDIUM priority", len(medium))
        m4.metric("NEW this cycle", len(new))
        m5.metric("Cooling off", len(cooling))

        st.divider()

        def _ticker_card(t, accent_color: str) -> str:
            """Build HTML for a single discovered ticker card."""
            ticker = t.get("ticker", "")
            name = t.get("company_name") or ticker
            sector = t.get("sector") or ""
            industry = t.get("industry") or ""
            sector_line = " · ".join(filter(None, [sector, industry]))

            cap_str = ""
            cap_val = t.get("market_cap")
            if cap_val:
                if cap_val >= 1e12:
                    cap_str = f"${cap_val/1e12:.1f}T"
                elif cap_val >= 1e9:
                    cap_str = f"${cap_val/1e9:.1f}B"
                else:
                    cap_str = f"${cap_val/1e6:.0f}M"

            streak = t.get("consecutive_days", 1)
            streak_icon = "🔥" if streak >= 4 else "📈" if streak >= 2 else "✨"
            streak_str = f"{streak_icon} Day {streak} streak"

            buzz = t.get("last_buzz_score")
            buzz_str = f"Buzz {buzz:.1f}" if buzz else ""
            mentions = t.get("mention_count")
            mention_str = f"{mentions} mentions" if mentions else ""

            summary = t.get("layman_summary") or ""
            if len(summary) > 200:
                summary = summary[:200] + "..."

            bull = t.get("bull_case") or ""
            if len(bull) > 120:
                bull = bull[:120] + "..."
            bear = t.get("bear_case") or ""
            if len(bear) > 120:
                bear = bear[:120] + "..."

            quality_html = ""
            thesis_quality = t.get("thesis_quality")
            if thesis_quality is not None:
                pct = max(0, min(100, thesis_quality / 10 * 100))
                qcolor = COLORS["green"] if pct >= 70 else COLORS["amber"] if pct >= 50 else COLORS["red"]
                quality_html = (
                    f'<div style="display:flex;align-items:center;gap:6px;margin-top:4px">'
                    f'<span style="font-size:0.72rem;color:{COLORS["text_muted"]}">QUALITY</span>'
                    f'<div style="flex:1;background:{COLORS["surface2"]};border-radius:3px;height:5px">'
                    f'<div style="width:{pct:.0f}%;background:{qcolor};border-radius:3px;height:5px"></div>'
                    f'</div>'
                    f'<span style="font-size:0.72rem;color:{qcolor};font-weight:600">{thesis_quality}/10</span>'
                    f'</div>'
                )

            url_html = ""
            post_url = t.get("triggering_post_url")
            if post_url:
                url_html = (
                    f'<a href="{post_url}" target="_blank" '
                    f'style="font-size:0.75rem;color:{COLORS["blue"]};text-decoration:none;">'
                    f'View triggering post ↗</a>'
                )

            first_seen = t.get("first_seen", "")
            priority = t.get("priority", "")
            meta_parts = [p for p in [buzz_str, mention_str, cap_str, f"First seen {first_seen}"] if p]
            meta_line = " &nbsp;·&nbsp; ".join(meta_parts)

            bull_html = (
                f'<div style="margin-top:8px;font-size:0.78rem;color:{COLORS["green_dim"]}">'
                f'<span style="font-weight:600;color:{COLORS["green"]}">Bull: </span>{bull}</div>'
            ) if bull else ""

            bear_html = (
                f'<div style="margin-top:4px;font-size:0.78rem;color:{COLORS["text_muted"]}">'
                f'<span style="font-weight:600;color:{COLORS["red"]}">Bear: </span>{bear}</div>'
            ) if bear else ""

            return f"""
<div style="background:{COLORS['surface']};border:1px solid {accent_color}44;
border-left:3px solid {accent_color};border-radius:10px;padding:14px 18px;margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
    <div>
      <span style="font-size:1.1rem;font-weight:700;color:{COLORS['text']}">{ticker}</span>
      <span style="font-size:0.85rem;color:{COLORS['text_muted']};margin-left:8px">{name}</span>
      {f'<span style="font-size:0.75rem;color:{COLORS["text_muted"]};margin-left:8px">{sector_line}</span>' if sector_line else ''}
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span style="background:{accent_color}22;color:{accent_color};border:1px solid {accent_color}44;
      padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.72rem">{priority}</span>
      <span style="font-size:0.78rem;color:{COLORS['text_muted']}">{streak_str}</span>
    </div>
  </div>
  <div style="font-size:0.72rem;color:{COLORS['muted']};margin-top:4px">{meta_line}</div>
  {quality_html}
  {f'<div style="font-size:0.82rem;color:{COLORS["text_muted"]};margin-top:10px;line-height:1.5">{summary}</div>' if summary else ''}
  {bull_html}
  {bear_html}
  {f'<div style="margin-top:10px">{url_html}</div>' if url_html else ''}
</div>
"""

        # HIGH priority section
        if high:
            st.markdown(
                f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["green"]};'
                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px">'
                f'🔥 High Priority ({len(high)} tickers)</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for i, t in enumerate(high):
                with cols[i % 2]:
                    st.markdown(_ticker_card(t, COLORS["green"]), unsafe_allow_html=True)

        # MEDIUM + NEW (building) section
        building = medium + new
        if building:
            st.divider()
            st.markdown(
                f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["blue"]};'
                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px">'
                f'📈 Building ({len(building)} tickers)</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for i, t in enumerate(building):
                accent = COLORS["blue"] if t.priority == "MEDIUM" else COLORS["amber"]
                with cols[i % 2]:
                    st.markdown(_ticker_card(t, accent), unsafe_allow_html=True)

        # COOLING section
        if cooling:
            st.divider()
            st.markdown(
                f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["muted"]};'
                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px">'
                f'❄️ Cooling Off ({len(cooling)} tickers)</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for i, t in enumerate(cooling):
                with cols[i % 3]:
                    st.markdown(_ticker_card(t, COLORS["muted"]), unsafe_allow_html=True)


# ================================================================== #
#  Tab 2: Ticker Detail                                               #
# ================================================================== #

with tab2:
    st.header("Ticker Detail")

    # Build list from discovered tickers first, then fall back to signal tickers
    disc_tickers = {t["ticker"]: t for t in discovered} if discovered else {}
    signal_tickers = sorted(df["ticker"].unique().tolist()) if not df.empty else []
    all_tickers = sorted(set(list(disc_tickers.keys()) + signal_tickers))

    if not all_tickers:
        st.info("No data yet. Run the pipeline: `python main.py --once`")
    else:
        selected_ticker = st.selectbox("Select ticker", all_tickers)
        disc = disc_tickers.get(selected_ticker)
        ticker_df = df[df["ticker"] == selected_ticker].sort_values("date", ascending=False) if not df.empty else pd.DataFrame()

        # Company snapshot banner
        if disc:
            company_name = disc.get("company_name") or selected_ticker
            priority_color = PRIORITY_COLORS.get(disc.get("priority", ""), COLORS["muted"])
            cap_str = ""
            mkt_cap = disc.get("market_cap")
            if mkt_cap:
                cap_str = f"${mkt_cap/1e12:.1f}T" if mkt_cap >= 1e12 else f"${mkt_cap/1e9:.1f}B" if mkt_cap >= 1e9 else f"${mkt_cap/1e6:.0f}M"

            sector_parts = [p for p in [disc.get("sector"), disc.get("industry")] if p]

            website_link = (
                f' &nbsp;<a href="{disc["website"]}" target="_blank" '
                f'style="color:{COLORS["blue"]};font-size:0.75rem;text-decoration:none">'
                f'website ↗</a>'
            ) if disc.get("website") else ""

            priority = disc.get("priority", "")
            consec = disc.get("consecutive_days", 1)
            peak = disc.get("peak_streak", 1)
            first_seen = disc.get("first_seen", "")

            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {priority_color}44;'
                f'border-left:4px solid {priority_color};border-radius:10px;padding:16px 20px;margin-bottom:16px">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">'
                f'<div>'
                f'<span style="font-size:1.5rem;font-weight:700;color:{COLORS["text"]}">{selected_ticker}</span>'
                f'<span style="font-size:1rem;color:{COLORS["text_muted"]};margin-left:10px">{company_name}</span>'
                f'{website_link}'
                f'</div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">'
                f'<span style="background:{priority_color}22;color:{priority_color};border:1px solid {priority_color}44;'
                f'padding:3px 10px;border-radius:5px;font-weight:600;font-size:0.78rem">{priority}</span>'
                f'<span style="font-size:0.78rem;color:{COLORS["text_muted"]}">Day {consec} streak &nbsp;·&nbsp; Peak {peak} days</span>'
                f'</div>'
                f'</div>'
                f'<div style="font-size:0.78rem;color:{COLORS["muted"]};margin-top:6px">'
                f'{" &nbsp;·&nbsp; ".join(sector_parts)}'
                f'{f" &nbsp;·&nbsp; Mkt cap {cap_str}" if cap_str else ""}'
                f' &nbsp;·&nbsp; First seen {first_seen}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Description
            description = disc.get("description") or ""
            if description:
                st.markdown(
                    f'<p style="font-size:0.83rem;color:{COLORS["text_muted"]};'
                    f'line-height:1.6;margin-bottom:16px">{description[:400]}{"..." if len(description) > 400 else ""}</p>',
                    unsafe_allow_html=True,
                )

        # Latest signal (if any)
        if not ticker_df.empty:
            latest = ticker_df.iloc[0]
            color = LABEL_COLORS.get(latest["label"], COLORS["muted"])
            text_color = LABEL_TEXT_COLORS.get(latest["label"], "#fff")
            st.markdown(
                f'<div style="background:{color};color:{text_color};padding:12px 18px;'
                f'border-radius:10px;display:flex;align-items:center;justify-content:space-between;'
                f'margin-bottom:16px">'
                f'<span style="font-size:1.1rem;font-weight:700;letter-spacing:0.02em">'
                f'Latest signal: {latest["label"]}</span>'
                f'<span style="font-size:0.82rem;opacity:0.75">{latest["date"]} &nbsp;·&nbsp; '
                f'Confluence {latest["confluence"]:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Discovery analysis section
        layman_summary = disc.get("layman_summary") if disc else None
        bull_case = disc.get("bull_case") if disc else None
        bear_case = disc.get("bear_case") if disc else None
        key_catalyst = disc.get("key_catalyst") if disc else None

        if disc and (layman_summary or bull_case or bear_case or key_catalyst):
            st.subheader("Claude's Analysis")

            if layman_summary:
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-radius:8px;padding:14px 18px;margin-bottom:12px;'
                    f'font-size:0.85rem;color:{COLORS["text"]};line-height:1.6">'
                    f'<span style="font-size:0.72rem;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.06em;font-weight:600">Summary</span>'
                    f'<div style="margin-top:6px">{layman_summary}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            analysis_cols = st.columns(2)
            with analysis_cols[0]:
                if bull_case:
                    st.markdown(
                        f'<div style="background:#052e16;border:1px solid {COLORS["green"]}33;'
                        f'border-radius:8px;padding:14px 18px;height:100%">'
                        f'<div style="font-size:0.72rem;color:{COLORS["green"]};'
                        f'text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:6px">Bull Case</div>'
                        f'<div style="font-size:0.83rem;color:{COLORS["green_dim"]};line-height:1.5">{bull_case}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with analysis_cols[1]:
                if bear_case:
                    st.markdown(
                        f'<div style="background:#2d0000;border:1px solid {COLORS["red"]}33;'
                        f'border-radius:8px;padding:14px 18px;height:100%">'
                        f'<div style="font-size:0.72rem;color:{COLORS["red"]};'
                        f'text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:6px">Bear Case</div>'
                        f'<div style="font-size:0.83rem;color:#fca5a5;line-height:1.5">{bear_case}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if key_catalyst:
                st.markdown(
                    f'<div style="background:{COLORS["surface2"]};border:1px solid {COLORS["border"]};'
                    f'border-radius:8px;padding:12px 18px;margin-top:10px;'
                    f'font-size:0.83rem;color:{COLORS["text_muted"]};line-height:1.5">'
                    f'<span style="color:{COLORS["amber"]};font-weight:600">Key catalyst: </span>'
                    f'{key_catalyst}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            confidence_row = []
            analysis_confidence = disc.get("analysis_confidence")
            thesis_quality = disc.get("thesis_quality")
            buzz_score = disc.get("last_buzz_score")
            stocktwits = disc.get("stocktwits_count")
            if analysis_confidence is not None:
                confidence_row.append(f"Analysis confidence: {analysis_confidence}/10")
            if thesis_quality is not None:
                confidence_row.append(f"Thesis quality: {thesis_quality}/10")
            if buzz_score:
                confidence_row.append(f"Buzz score: {buzz_score:.1f}")
            if stocktwits:
                confidence_row.append(f"StockTwits: {stocktwits:,} msgs")

            if confidence_row:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:{COLORS["muted"]};margin-top:8px">'
                    f' &nbsp;·&nbsp; '.join(confidence_row) + '</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

        # Reddit post summaries
        post_summaries_raw = disc.get("post_summaries") if disc else None
        triggering_url = disc.get("triggering_post_url") if disc else None

        if disc and post_summaries_raw:
            st.subheader("Reddit Posts")
            try:
                summaries = json.loads(post_summaries_raw) if isinstance(post_summaries_raw, str) else post_summaries_raw
                if isinstance(summaries, list) and summaries:
                    for i, summary in enumerate(summaries[:6], 1):
                        st.markdown(
                            f'<div style="display:flex;gap:10px;padding:10px 0;'
                            f'border-bottom:1px solid {COLORS["border"]}">'
                            f'<span style="color:{COLORS["green"]};font-weight:600;font-size:0.82rem;'
                            f'flex-shrink:0;min-width:20px">{i}.</span>'
                            f'<span style="font-size:0.82rem;color:{COLORS["text_muted"]};line-height:1.5">'
                            f'{summary}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if triggering_url:
                        st.markdown(
                            f'<a href="{triggering_url}" target="_blank" '
                            f'style="font-size:0.78rem;color:{COLORS["blue"]};text-decoration:none;'
                            f'display:inline-block;margin-top:8px">View top Reddit post ↗</a>',
                            unsafe_allow_html=True,
                        )
            except (json.JSONDecodeError, TypeError):
                pass

            st.divider()

        # Model inputs
        if not ticker_df.empty:
            latest = ticker_df.iloc[0]
            st.subheader("Model Inputs (Latest Signal)")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:10px">Kronos</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Direction", latest["direction"] or "—")
                st.metric("Confidence", f"{latest['kronos_conf']:.0%}" if latest['kronos_conf'] else "—")
                st.metric("Predicted move", f"{(latest['kronos_pct'] or 0)*100:+.2f}%")
                st.metric("Price at signal", f"${latest['price_at_signal']:,.2f}" if latest["price_at_signal"] else "—")

            with c2:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:10px">Reddit Sentiment</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Sentiment", latest["reddit_sentiment"] or "—")
                st.metric("Score", f"{latest['reddit_score']:+.2f}" if latest['reddit_score'] is not None else "—")
                st.metric("Posts analyzed", int(latest["reddit_posts"] or 0))

            with c3:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:10px">Technicals</div>',
                    unsafe_allow_html=True,
                )
                st.metric("RSI (14)", f"{latest['rsi']:.1f}" if latest['rsi'] else "—")
                st.metric("MACD", latest["macd"] or "—")
                st.metric("Bollinger", latest["bb"] or "—")
                st.metric("ADX", f"{latest['adx']:.1f}" if latest['adx'] else "—")
                st.metric("Vol ratio", f"{latest['vol_ratio']:.2f}x" if latest['vol_ratio'] else "—")

            if latest["reasoning"]:
                st.divider()
                st.subheader("Confluence Reasoning")
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
                st.subheader("Confluence History")
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
                for y, color, lbl in [
                    (0.68, COLORS["green"],      "STRONG BUY"),
                    (0.54, COLORS["green_dim"],  "BUY"),
                    (0.46, COLORS["orange"],     "SELL"),
                    (0.32, COLORS["red"],        "STRONG SELL"),
                ]:
                    hist_fig.add_hline(
                        y=y, line_dash="dot", line_color=color, line_width=1,
                        annotation_text=lbl,
                        annotation_font=dict(size=10, color=color),
                        annotation_position="right",
                    )
                hist_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis={**_AXIS_STYLE, "title": None},
                    yaxis={**_AXIS_STYLE, "range": [0, 1], "title": "Confluence"},
                    height=280,
                    showlegend=False,
                )
                st.plotly_chart(hist_fig, use_container_width=True)

        elif not disc:
            st.info(f"No data for {selected_ticker} yet.")


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
        dur_fig.update_layout(**PLOTLY_LAYOUT, xaxis=_AXIS_STYLE, yaxis=_AXIS_STYLE)
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
        sr_fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=_AXIS_STYLE,
            yaxis={**_AXIS_STYLE, "range": [0, 1]},
        )
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
        err_fig.update_layout(**PLOTLY_LAYOUT, xaxis=_AXIS_STYLE, yaxis=_AXIS_STYLE)
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
                    pnl_fig.update_layout(**PLOTLY_LAYOUT, xaxis=_AXIS_STYLE, yaxis=_AXIS_STYLE)
                    st.plotly_chart(pnl_fig, use_container_width=True)
