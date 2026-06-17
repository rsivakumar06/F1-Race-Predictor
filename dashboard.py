"""
F1 Race Predictor Dashboard — Streamlit App

A web-based UI for browsing predictions, visualizations, and race data.

Launch:
    streamlit run dashboard.py

Design: a broadcast timing-screen aesthetic — deep blue-black "pit wall"
surface, team-colour ID bars, monospace timing data, and a single cyan
telemetry signal for live values. Titillium Web (the F1 brand face) for
display, JetBrains Mono for data, Inter for UI copy.

Features (unchanged):
    - Race predictions with win/podium probabilities
    - Elo driver rankings with visual bars
    - Feature importance from the ML model
    - Race analysis plots (lap times, positions, tyre strategy)
    - Telemetry comparisons between drivers
    - Track maps with speed heatmaps
    - Data pipeline controls (collect, train, update)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config

try:
    import fastf1
    import fastf1.plotting
    fastf1.plotting.setup_mpl(color_scheme="fastf1")
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
except Exception:
    pass


# ═══════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════
CARBON  = "#0A0E14"   # base — deep blue-black pit-wall screen
SURFACE = "#121821"   # cards / rows
LINE    = "#1F2733"   # hairline dividers
TEXT    = "#E6EAF0"   # primary text
MUTED   = "#8A94A6"   # secondary text
SIGNAL  = "#00E1FF"   # cyan telemetry accent
GOLD, SILVER, BRONZE = "#F5D77A", "#CBD2DA", "#D08B4E"


# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root{
  --carbon:#0A0E14; --surface:#121821; --line:#1F2733;
  --text:#E6EAF0; --muted:#8A94A6; --signal:#00E1FF;
}

/* base */
.stApp { background-color: var(--carbon); font-family:'Inter',-apple-system,sans-serif; }
.block-container { padding-top: 4.5rem; max-width: 1320px; }
h1,h2,h3,h4 { font-family:'Titillium Web',sans-serif !important; color:var(--text) !important; }
p, span, label, div { color:var(--text); }

/* strip default chrome (selectors target Streamlit internals — may shift across versions) */
[data-testid="stToolbar"] { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

/* sidebar */
[data-testid="stSidebar"] { background:#0C121B; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top:1.4rem; }
.brand { display:flex; align-items:center; gap:11px; margin-bottom:4px; }
.brand-mark { width:4px; height:30px; background:var(--signal); border-radius:2px;
              box-shadow:0 0 12px rgba(0,225,255,.5); }
.brand-text { font-family:'Titillium Web',sans-serif; font-weight:700; font-size:21px;
              color:var(--text); letter-spacing:.01em; line-height:1; }
.brand-sub { font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:.24em;
             color:var(--muted); text-transform:uppercase; margin:6px 0 16px 15px; }

/* page header */
.page-eyebrow { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.24em;
                text-transform:uppercase; color:var(--signal); }
.page-title { font-family:'Titillium Web',sans-serif; font-size:36px; font-weight:700;
              color:var(--text); margin:2px 0 0; line-height:1.05; }
.page-rule { height:2px; background:linear-gradient(90deg,var(--signal),transparent);
             margin:14px 0 18px; border:none; }
.race-line { font-family:'Titillium Web',sans-serif; font-size:22px; font-weight:600;
             color:var(--text); }

/* section header */
.sec-eyebrow { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.22em;
               text-transform:uppercase; color:var(--signal); margin-top:6px; }
.sec-title { font-family:'Titillium Web',sans-serif; font-size:20px; font-weight:600;
             color:var(--text); margin:2px 0 12px; }

/* mode badge */
.badge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:11px;
         letter-spacing:.12em; padding:4px 13px; border-radius:20px; border:1px solid var(--line); }
.badge.forecast { color:#F5D77A; border-color:#3a3320; background:rgba(245,215,122,.07); }
.badge.pre  { color:var(--signal); border-color:#10323b; background:rgba(0,225,255,.07); }
.badge.post { color:#7CE38B; border-color:#16321d; background:rgba(124,227,139,.07); }

/* timing tower — the signature element */
.tt-wrap { display:flex; flex-direction:column; gap:6px; margin-top:4px; }
.tt-row { display:grid; grid-template-columns:6px 50px minmax(110px,1fr) 66px 66px minmax(170px,260px);
          align-items:center; background:var(--surface); border:1px solid var(--line);
          border-radius:10px; overflow:hidden; min-height:50px; transition:border-color .15s; }
.tt-row:hover { border-color:#2c3a4a; }
.tt-id { width:6px; height:50px; }
.tt-pos { font-family:'JetBrains Mono',monospace; font-size:22px; font-weight:600;
          color:var(--text); text-align:center; }
.tt-row.p1 .tt-pos { color:#F5D77A; }
.tt-row.p2 .tt-pos { color:#CBD2DA; }
.tt-row.p3 .tt-pos { color:#D08B4E; }
.tt-driver { display:flex; flex-direction:column; padding:7px 4px; }
.tt-code { font-family:'Titillium Web',sans-serif; font-weight:700; font-size:16px;
           letter-spacing:.03em; color:var(--text); }
.tt-team { font-size:11px; color:var(--muted); }
.tt-meta { display:flex; flex-direction:column; padding:0 6px; }
.tt-label { font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:.12em;
            color:var(--muted); }
.tt-val { font-family:'JetBrains Mono',monospace; font-size:14px; color:var(--text); }
.tt-prob { display:flex; align-items:center; gap:10px; padding:0 14px 0 6px; }
.tt-prob-track { flex:1; height:8px; background:#070A0F; border-radius:4px; overflow:hidden;
                 border:1px solid #131c27; }
.tt-prob-fill { height:100%; background:linear-gradient(90deg,rgba(0,225,255,.45),var(--signal));
                border-radius:4px; box-shadow:0 0 8px rgba(0,225,255,.35); }
.tt-prob-nums { display:flex; flex-direction:column; align-items:flex-end; min-width:54px; }
.tt-prob-num { font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--signal); }
.tt-pod { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--muted); letter-spacing:.05em; }

/* qualifying grid */
.q-wrap { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; margin-top:4px; }
.q-row { display:grid; grid-template-columns:5px 34px 1fr auto; align-items:center;
         background:var(--surface); border:1px solid var(--line); border-radius:8px; min-height:40px; }
.q-pos { font-family:'JetBrains Mono',monospace; font-size:14px; text-align:center; color:var(--muted); }
.q-driver { display:flex; flex-direction:column; padding:5px 4px; }
.q-code { font-family:'Titillium Web',sans-serif; font-weight:700; font-size:14px; color:var(--text); }
.q-team { font-size:10px; color:var(--muted); }
.q-src { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--signal);
         padding-right:10px; letter-spacing:.06em; }

/* stat cards */
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.stat-card { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--signal);
             border-radius:10px; padding:16px 18px; }
.stat-num { font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; color:var(--text); line-height:1.1; }
.stat-sub { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--signal); margin-top:2px; }
.stat-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.14em;
              text-transform:uppercase; color:var(--muted); margin-top:8px; }

/* elo text-bar fallback */
.elo-row { display:grid; grid-template-columns:30px 54px 70px 1fr; align-items:center;
           gap:8px; padding:3px 0; font-family:'JetBrains Mono',monospace; font-size:13px; }
.elo-bar { height:8px; background:linear-gradient(90deg,rgba(0,225,255,.4),var(--signal)); border-radius:4px; }

/* tabs — distinct bordered chips */
.stTabs [data-baseweb="tab-list"] { gap:10px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family:'Titillium Web',sans-serif; font-weight:600; font-size:15px;
    padding:8px 20px; background:var(--surface); border:1px solid var(--line);
    border-radius:8px 8px 0 0; color:var(--muted); margin-bottom:-1px;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--text); border-color:#2c3a4a; }
.stTabs [aria-selected="true"] {
    color:var(--signal) !important; background:#0E1620;
    border-color:#163240; border-bottom:2px solid var(--signal);
}
.stTabs [data-baseweb="tab-highlight"] { background-color:var(--signal) !important; }
.stTabs [data-baseweb="tab-border"] { background-color:var(--line) !important; }

/* buttons */
.stButton button { font-family:'Titillium Web',sans-serif; font-weight:600; border:1px solid var(--line);
                   background:var(--surface); color:var(--text); border-radius:8px; }
.stButton button:hover { border-color:var(--signal); color:var(--signal); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MATPLOTLIB STYLING
# ═══════════════════════════════════════════════

def _style_fig(fig, axes):
    """Apply the dashboard palette to a matplotlib figure + axes."""
    fig.patch.set_facecolor(CARBON)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for a in axes:
        a.set_facecolor(CARBON)
        a.tick_params(colors=MUTED)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_color(LINE)
        a.spines["left"].set_color(LINE)
        a.grid(alpha=0.12, color=MUTED)
        a.xaxis.label.set_color(TEXT)
        a.yaxis.label.set_color(TEXT)
        a.title.set_color(TEXT)


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark"></div>'
    '<div class="brand-text">F1 PREDICTOR</div></div>'
    '<div class="brand-sub">Telemetry &middot; Elo &middot; XGBoost</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["Predictions", "Race Analysis", "Track Visualization",
     "Elo Rankings", "Pipeline Controls"],
)


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

@st.cache_data(ttl=60)
def load_features():
    """Load the features DataFrame (refreshes every 60s)."""
    path = config.DATA_DIR / "features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource
def load_ensemble():
    """Load the trained ensemble model."""
    from models.ensemble import F1Ensemble
    ensemble = F1Ensemble()
    try:
        ensemble.load()
        return ensemble
    except Exception:
        return None


@st.cache_resource
def load_session(year, race, session_type="R"):
    """Load a FastF1 session (cached)."""
    from data.pipeline import F1DataPipeline
    pipeline = F1DataPipeline()
    return pipeline.load_session_for_visualization(year, race, session_type)


@st.cache_data(ttl=3600)
def driver_colour(code):
    """Team/driver colour for ID bars (reuses fastf1.plotting; falls back to signal)."""
    try:
        import fastf1.plotting
        c = fastf1.plotting.get_driver_color(code, session=None)
        if isinstance(c, str) and c.startswith("#"):
            return c
    except Exception:
        pass
    return None


def get_available_races(df):
    """Get list of (year, round, name) tuples from data."""
    if df is None:
        return []
    races = df.groupby(["Year", "RoundNumber", "EventName"]).size().reset_index()
    races = races.sort_values(["Year", "RoundNumber"], ascending=[False, False])
    return [(int(r["Year"]), int(r["RoundNumber"]), r["EventName"])
            for _, r in races.iterrows()]


# ── presentation helpers ──────────────────────────────────────

def page_header(eyebrow, title):
    st.markdown(
        f'<div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        f'<hr class="page-rule"/>',
        unsafe_allow_html=True,
    )


def section_header(eyebrow, title):
    st.markdown(
        f'<div class="sec-eyebrow">{eyebrow}</div>'
        f'<div class="sec-title">{title}</div>',
        unsafe_allow_html=True,
    )


def stat_cards(items):
    """items: list of (label, value) or (label, value, subvalue)."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        sub = f'<div class="stat-sub">{it[2]}</div>' if len(it) > 2 else ""
        cards.append(
            f'<div class="stat-card"><div class="stat-num">{value}</div>'
            f'{sub}<div class="stat-label">{label}</div></div>'
        )
    st.markdown(f'<div class="stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_timing_tower(race_pred):
    rows = []
    for _, row in race_pred.iterrows():
        rank = int(row["FinalRank"])
        driver = str(row["Abbreviation"])
        team = str(row.get("TeamName", ""))
        elo_val = row.get("EloRating", float("nan"))
        win = (row.get("WinProb", 0) or 0) * 100
        pod = (row.get("PodiumProb", 0) or 0) * 100
        grid = f"P{int(row['GridPosition'])}" if pd.notna(row.get("GridPosition")) else "&mdash;"
        col = driver_colour(driver) or SIGNAL
        elo_str = f"{elo_val:.0f}" if pd.notna(elo_val) else "&mdash;"
        cls = "tt-row" + (f" p{rank}" if rank <= 3 else "")
        fillw = max(0.0, min(100.0, win))
        rows.append(
            f'<div class="{cls}">'
            f'<div class="tt-id" style="background:{col}"></div>'
            f'<div class="tt-pos">{rank}</div>'
            f'<div class="tt-driver"><span class="tt-code">{driver}</span>'
            f'<span class="tt-team">{team}</span></div>'
            f'<div class="tt-meta"><span class="tt-label">GRID</span>'
            f'<span class="tt-val">{grid}</span></div>'
            f'<div class="tt-meta"><span class="tt-label">ELO</span>'
            f'<span class="tt-val">{elo_str}</span></div>'
            f'<div class="tt-prob"><div class="tt-prob-track">'
            f'<div class="tt-prob-fill" style="width:{fillw:.1f}%"></div></div>'
            f'<div class="tt-prob-nums"><span class="tt-prob-num">{win:.1f}%</span>'
            f'<span class="tt-pod">POD {pod:.0f}%</span></div></div>'
            f'</div>'
        )
    st.markdown('<div class="tt-wrap">' + "".join(rows) + '</div>', unsafe_allow_html=True)


def render_quali_grid(quali_pred):
    if quali_pred is None or quali_pred.empty:
        st.info("Qualifying prediction unavailable.")
        return
    has_src = "Source" in quali_pred.columns
    cells = []
    for _, r in quali_pred.iterrows():
        pos = (int(r["PredictedQualiRank"])
               if "PredictedQualiRank" in r and pd.notna(r["PredictedQualiRank"]) else "&mdash;")
        code = str(r.get("Abbreviation", ""))
        team = str(r.get("TeamName", ""))
        col = driver_colour(code) or SIGNAL
        src = f'<span class="q-src">{r["Source"]}</span>' if has_src else "<span></span>"
        cells.append(
            f'<div class="q-row"><div class="tt-id" style="background:{col};height:40px"></div>'
            f'<div class="q-pos">{pos}</div>'
            f'<div class="q-driver"><span class="q-code">{code}</span>'
            f'<span class="q-team">{team}</span></div>'
            f'{src}</div>'
        )
    st.markdown('<div class="q-wrap">' + "".join(cells) + '</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PREDICTIONS PAGE
# ═══════════════════════════════════════════════

def page_predictions():
    page_header("F1 Predictor / 2026", "Race Predictions")

    df = load_features()
    ensemble = load_ensemble()

    if df is None:
        st.error("No data found. Run `python main.py --step collect` and `--step features` first.")
        return

    if ensemble is None:
        st.warning("Model not trained yet. Run `python main.py --step train` first.")
        st.info("Showing Elo ratings only (no XGBoost predictions).")
        from models.elo_rating import F1EloRating
        elo = F1EloRating()
        elo.process_historical(df)
        _show_elo_only(elo, df)
        return

    from data.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(df)

    # ── Round selector ──────────────────────────────────────────
    latest_year = int(df["Year"].max())
    year_data = df[df["Year"] == latest_year]
    available_rounds = sorted(year_data["RoundNumber"].unique())
    latest_available = int(available_rounds[-1])

    all_rounds = list(range(1, 25))  # up to round 24
    round_labels = {}
    for r in all_rounds:
        rd = year_data[year_data["RoundNumber"] == r]
        if not rd.empty:
            round_labels[r] = f"Round {r} — {rd['EventName'].iloc[0]}"
        else:
            round_labels[r] = f"Round {r} — (forecast)"

    selected_round = st.sidebar.selectbox(
        "Select Round",
        all_rounds,
        index=latest_available - 1,
        format_func=lambda r: round_labels[r],
    )

    # ── Build race_data (with fallback for future rounds) ───────
    race_data = year_data[year_data["RoundNumber"] == selected_round].copy()
    is_forecast = False

    if race_data.empty:
        is_forecast = True
        race_data = year_data[year_data["RoundNumber"] == latest_available].copy()
        race_data["RoundNumber"] = selected_round
        race_data["IsPreRace"] = True
        result_cols = [c for c in race_data.columns if any(
            x in c.lower() for x in ["finishpos", "finish_pos", "race_pos", "laps_led", "race_lap"]
        )]
        race_data[result_cols] = None
        try:
            import fastf1
            event = fastf1.get_event(latest_year, selected_round)
            race_data["EventName"] = event["EventName"]
        except Exception:
            race_data["EventName"] = f"Round {selected_round}"

    race_name = race_data["EventName"].iloc[0] if "EventName" in race_data.columns else f"Round {selected_round}"
    is_pre_race = is_forecast or race_data.get("IsPreRace", pd.Series([False])).any()
    mode_label = "FORECAST (no session data yet)" if is_forecast else (
        "PRE-RACE" if is_pre_race else "POST-RACE"
    )
    mode_cls = "forecast" if is_forecast else ("pre" if is_pre_race else "post")

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:6px;">'
        f'<span class="race-line">{race_name} {latest_year} &nbsp;'
        f'<span style="color:var(--muted);font-size:15px;">Round {selected_round}</span></span>'
        f'<span class="badge {mode_cls}">{mode_label}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="page-rule"/>', unsafe_allow_html=True)

    # ── Stage 1: Qualifying prediction ──────────────────────────
    section_header("Stage 01", "Predicted Qualifying Grid")
    try:
        quali_pred = ensemble.predict_quali(race_data, verbose=False)
        render_quali_grid(quali_pred)
    except Exception as e:
        st.warning(f"Could not generate qualifying prediction: {e}")
        quali_pred = None

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Stage 2: Race prediction ─────────────────────────────────
    section_header("Stage 02", "Predicted Race Result")
    try:
        race_pred = ensemble.predict_race(race_data)
    except Exception as e:
        st.error(f"Could not generate race prediction: {e}")
        return

    render_timing_tower(race_pred)

    # Podium snapshot
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    podium = race_pred.head(3)
    labels = ["P1 — Win Favourite", "P2", "P3"]
    items = []
    for i, (_, row) in enumerate(podium.iterrows()):
        win = (row.get("WinProb", 0) or 0) * 100
        items.append((labels[i], row["Abbreviation"], f"{win:.1f}% win"))
    if items:
        stat_cards(items)

    # ── Model diagnostics ────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("Model details & diagnostics"):
        tree_w, elo_w = ensemble._get_blend_weights()
        st.markdown(f"**Blend weights:** XGBoost {tree_w:.0%} / Elo {elo_w:.0%}")
        st.markdown(f"**Races in current era:** {ensemble.elo_model._races_in_era}")
        st.markdown(f"**Training data:** {len(df)} rows, {len(feature_cols)} features")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Win Probability (Top 10)")
            top10 = race_pred.head(10)
            fig, ax = plt.subplots(figsize=(6, 7))
            ax.barh(range(len(top10) - 1, -1, -1), top10["WinProb"] * 100,
                    color=SIGNAL, alpha=0.85, height=0.6)
            ax.set_yticks(range(len(top10) - 1, -1, -1))
            ax.set_yticklabels(top10["Abbreviation"], color=TEXT, fontfamily="monospace")
            ax.set_xlabel("Win Probability (%)")
            _style_fig(fig, ax)
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("##### Top 15 Feature Importance")
            importance = ensemble.tree_model.get_feature_importance(15)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.barh(range(len(importance) - 1, -1, -1), importance["Importance"],
                    color=SIGNAL, alpha=0.85)
            ax.set_yticks(range(len(importance) - 1, -1, -1))
            ax.set_yticklabels(importance["Feature"], color=TEXT, fontsize=9)
            _style_fig(fig, ax)
            st.pyplot(fig)
            plt.close()


def _show_elo_only(elo, df):
    """Show Elo rankings when no trained model is available."""
    rankings = elo.get_current_rankings()
    active = df[df["Year"] == df["Year"].max()]["Abbreviation"].unique()
    rankings = rankings[rankings["Abbreviation"].isin(active)]

    section_header("Fallback", "Current Elo Rankings")
    rows = []
    top = rankings.head(22).reset_index(drop=True)
    max_above = max((top["EloRating"] - 1300).max(), 1)
    for i, row in top.iterrows():
        elo_val = row["EloRating"]
        pct = max(2.0, (elo_val - 1300) / max_above * 100)
        col = driver_colour(row["Abbreviation"]) or SIGNAL
        rows.append(
            f'<div class="elo-row"><span style="color:var(--muted)">{i+1:02d}</span>'
            f'<span class="tt-code">{row["Abbreviation"]}</span>'
            f'<span style="color:var(--signal)">{elo_val:.0f}</span>'
            f'<span class="elo-bar" style="width:{pct:.0f}%;'
            f'background:linear-gradient(90deg,{col}55,{col})"></span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# RACE ANALYSIS PAGE
# ═══════════════════════════════════════════════

def page_race_analysis():
    page_header("Session Telemetry", "Race Analysis")

    df = load_features()
    if df is None:
        st.error("No data found.")
        return

    races = get_available_races(df)
    if not races:
        st.warning("No races available.")
        return

    race_labels = [f"{name} {year}" for year, rnd, name in races]
    selected_idx = st.sidebar.selectbox("Select Race", range(len(race_labels)),
                                         format_func=lambda i: race_labels[i])
    year, rnd, name = races[selected_idx]

    st.markdown(f'<div class="race-line">{name} {year}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with st.spinner(f"Loading {name} {year}..."):
        try:
            session = load_session(year, rnd, "R")
        except Exception as e:
            st.error(f"Could not load session: {e}")
            return

    from visualizations.race_plots import RacePlots
    plots = RacePlots()

    tab1, tab2, tab3, tab4 = st.tabs(["Lap Times", "Position Chart", "Tyre Strategy", "Pace Comparison"])

    with tab1:
        section_header("Distribution", "Lap Time Distribution")
        fig = plots.lap_time_distribution(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab2:
        section_header("Order", "Position Chart")
        fig = plots.position_chart(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab3:
        section_header("Strategy", "Tyre Strategy")
        fig = plots.tyre_strategy(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab4:
        section_header("Head to Head", "Race Pace Comparison")
        results = session.results.sort_values("Position")
        drivers = results["Abbreviation"].tolist()

        col1, col2 = st.columns(2)
        d1 = col1.selectbox("Driver 1", drivers, index=0)
        d2 = col2.selectbox("Driver 2", drivers, index=min(1, len(drivers) - 1))

        if d1 != d2:
            fig = plots.race_pace_comparison(session, d1, d2, save=False)
            st.pyplot(fig)
            plt.close()


# ═══════════════════════════════════════════════
# TRACK VISUALIZATION PAGE
# ═══════════════════════════════════════════════

def page_track_viz():
    page_header("Circuit", "Track Visualization")

    df = load_features()
    if df is None:
        st.error("No data found.")
        return

    races = get_available_races(df)
    race_labels = [f"{name} {year}" for year, rnd, name in races]
    selected_idx = st.sidebar.selectbox("Select Race", range(len(race_labels)),
                                         format_func=lambda i: race_labels[i],
                                         key="track_race")
    year, rnd, name = races[selected_idx]

    st.markdown(f'<div class="race-line">{name} {year}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with st.spinner(f"Loading {name} {year}..."):
        try:
            session = load_session(year, rnd, "R")
        except Exception as e:
            st.error(f"Could not load session: {e}")
            return

    from visualizations.track_viz import TrackViz
    from visualizations.telemetry_plots import TelemetryPlots

    track = TrackViz()
    tel_plots = TelemetryPlots()

    results = session.results.sort_values("Position")
    drivers = results["Abbreviation"].tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["Track Map", "Speed Heatmap", "Gear Map", "Telemetry"])

    with tab1:
        section_header("Layout", "Circuit Map")
        fig = track.draw_track_map(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab2:
        driver = st.selectbox("Driver", drivers, index=0, key="speed_driver")
        section_header("Speed", f"Speed Heatmap — {driver}")
        fig = track.speed_heatmap(session, driver, save=False)
        st.pyplot(fig)
        plt.close()

    with tab3:
        driver = st.selectbox("Driver", drivers, index=0, key="gear_driver")
        section_header("Gears", f"Gear Map — {driver}")
        fig = track.gear_map(session, driver, save=False)
        st.pyplot(fig)
        plt.close()

    with tab4:
        section_header("Comparison", "Telemetry Comparison")
        col1, col2 = st.columns(2)
        d1 = col1.selectbox("Driver 1", drivers, index=0, key="tel_d1")
        d2 = col2.selectbox("Driver 2", drivers, index=min(1, len(drivers) - 1), key="tel_d2")

        session_type = st.radio("Session", ["Qualifying", "Race"], horizontal=True)
        ses_code = "Q" if session_type == "Qualifying" else "R"

        if d1 != d2:
            with st.spinner("Loading telemetry..."):
                try:
                    tel_session = load_session(year, rnd, ses_code)
                    fig = tel_plots.speed_trace_comparison(tel_session, d1, d2, save=False)
                    st.pyplot(fig)
                    plt.close()

                    fig = tel_plots.delta_time_on_track(tel_session, d1, d2, save=False)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Could not load telemetry: {e}")


# ═══════════════════════════════════════════════
# ELO RANKINGS PAGE
# ═══════════════════════════════════════════════

def page_elo_rankings():
    page_header("Rating System", "Elo Driver Rankings")

    df = load_features()
    if df is None:
        st.error("No data found.")
        return

    from models.elo_rating import F1EloRating
    elo = F1EloRating()
    pre_race = elo.process_historical(df)

    rankings = elo.get_current_rankings()
    active = df[df["Year"] == df["Year"].max()]["Abbreviation"].unique()
    rankings_active = rankings[rankings["Abbreviation"].isin(active)].reset_index(drop=True)

    section_header("Standings", "Current Active Driver Rankings")

    fig, ax = plt.subplots(figsize=(12, 10))
    n = len(rankings_active)
    colors = []
    for i in range(n):
        if i == 0:
            colors.append(GOLD)
        elif i == 1:
            colors.append(SILVER)
        elif i == 2:
            colors.append(BRONZE)
        else:
            colors.append(SIGNAL)

    ax.barh(range(n - 1, -1, -1), rankings_active["EloRating"] - 1300,
            color=colors, alpha=0.88, height=0.7)
    ax.set_yticks(range(n - 1, -1, -1))
    ax.set_yticklabels(
        [f"{r['Abbreviation']}  ({r['EloRating']:.0f})"
         for _, r in rankings_active.iterrows()],
        color=TEXT, fontfamily="monospace", fontsize=11,
    )
    ax.set_xlabel("Elo Rating (above 1300 baseline)")
    _style_fig(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Win probabilities for next race
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section_header("Next Race", "Win Probabilities")

    probs = elo.predict_probabilities(active.tolist())

    rank_labels = ["1st", "2nd", "3rd"]
    items = []
    for i, (_, row) in enumerate(probs.head(3).iterrows()):
        items.append((f"{rank_labels[i]} — {row['Abbreviation']}",
                      f"{row['WinProb']*100:.1f}%",
                      f"Elo {row['EloRating']:.0f}"))
    stat_cards(items)

    with st.expander("Full win probability table"):
        display_df = probs[["Abbreviation", "EloRating", "WinProb", "PodiumProb", "ExpectedPosition"]].copy()
        display_df["WinProb"] = (display_df["WinProb"] * 100).round(1).astype(str) + "%"
        display_df["PodiumProb"] = (display_df["PodiumProb"] * 100).round(1).astype(str) + "%"
        display_df["EloRating"] = display_df["EloRating"].round(0).astype(int)
        display_df["ExpectedPosition"] = display_df["ExpectedPosition"].round(1)
        display_df.columns = ["Driver", "Elo", "Win %", "Podium %", "Expected Pos"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Elo history chart
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section_header("Trajectory", "Elo Rating History")

    history = pd.DataFrame(elo.rating_history, columns=["Year", "Round", "Driver", "Rating"])
    history = history[history["Driver"].isin(active)]

    race_index = history.groupby(["Year", "Round"]).ngroup()
    history["RaceIndex"] = race_index

    fig, ax = plt.subplots(figsize=(14, 7))
    for driver in rankings_active.head(10)["Abbreviation"]:
        dh = history[history["Driver"] == driver]
        if dh.empty:
            continue
        try:
            color = fastf1.plotting.get_driver_color(driver, session=None)
        except Exception:
            color = "#AAAAAA"
        ax.plot(dh["RaceIndex"], dh["Rating"], label=driver, linewidth=1.5, alpha=0.9, color=color)

    ax.set_xlabel("Race (chronological)")
    ax.set_ylabel("Elo Rating")
    leg = ax.legend(loc="upper left", fontsize=9, ncol=2, facecolor=SURFACE, edgecolor=LINE)
    for txt in leg.get_texts():
        txt.set_color(TEXT)
    _style_fig(fig, ax)

    era_changes = history.groupby("Year")["RaceIndex"].min()
    for year_val in [2026]:
        if year_val in era_changes.index:
            ax.axvline(era_changes[year_val], color=SIGNAL, linestyle="--",
                       alpha=0.6, label="2026 Reg Change")

    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# PIPELINE CONTROLS PAGE
# ═══════════════════════════════════════════════

def page_pipeline():
    page_header("Operations", "Pipeline Controls")

    st.markdown("Update your data and retrain the model. After each race weekend, run these steps in order.")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    df = load_features()
    if df is not None:
        feat_count = len([c for c in df.columns if c.startswith("Driver_") or c.startswith("Team_")])
        stat_cards([
            ("Total Rows", f"{len(df):,}"),
            ("Seasons", f"{len(df['Year'].unique())}"),
            ("Drivers", f"{df['Abbreviation'].nunique()}"),
            ("Features", f"{feat_count}"),
        ])
        st.markdown(
            f"<div style='margin-top:12px;color:var(--muted);font-family:JetBrains Mono,monospace;font-size:12px;'>"
            f"LATEST &nbsp; {df['EventName'].iloc[-1]} {int(df['Year'].iloc[-1])}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No data found yet.")

    st.markdown('<hr class="page-rule"/>', unsafe_allow_html=True)

    section_header("Terminal", "Run Pipeline Steps")
    st.code("""
# Step 1: Collect latest race data
python main.py --step collect --year 2026

# Step 2: Rebuild features
python main.py --step features

# Step 3: Retrain model
python main.py --step train

# Step 4: Check predictions
python main.py --step predict --year 2026

# Or run everything at once:
python main.py --step all
    """, language="bash")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    section_header("Actions", "Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh data view", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    with col2:
        if st.button("Launch race animation", use_container_width=True):
            st.info("Run from terminal: `python main.py --step animate --race Australia --year 2025`")

    if df is not None:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        with st.expander("Raw data preview"):
            display_cols = ["Year", "RoundNumber", "EventName", "Abbreviation", "TeamName",
                            "GridPosition", "FinishPosition", "Points"]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[available].sort_values(["Year", "RoundNumber", "FinishPosition"],
                                          ascending=[False, False, True]).head(100),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════

if page == "Predictions":
    page_predictions()
elif page == "Race Analysis":
    page_race_analysis()
elif page == "Track Visualization":
    page_track_viz()
elif page == "Elo Rankings":
    page_elo_rankings()
elif page == "Pipeline Controls":
    page_pipeline()
