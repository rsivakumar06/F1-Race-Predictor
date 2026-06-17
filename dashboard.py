"""
F1 Race Predictor Dashboard — Streamlit App

A web-based UI for browsing predictions, visualizations, and race data.

Launch:
    streamlit run dashboard.py

Features:
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
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0D0D0D; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #E0E0E0 !important; }
    .metric-card {
        background: #1A1A1A;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #FF1801;
    }
    .gold { color: #FFD700; font-weight: bold; }
    .silver { color: #C0C0C0; font-weight: bold; }
    .bronze { color: #CD7F32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
st.sidebar.title("F1 Predictor")
st.sidebar.markdown("---")

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


def get_available_races(df):
    """Get list of (year, round, name) tuples from data."""
    if df is None:
        return []
    races = df.groupby(["Year", "RoundNumber", "EventName"]).size().reset_index()
    races = races.sort_values(["Year", "RoundNumber"], ascending=[False, False])
    return [(int(r["Year"]), int(r["RoundNumber"]), r["EventName"])
            for _, r in races.iterrows()]


# ═══════════════════════════════════════════════
# PREDICTIONS PAGE
# ═══════════════════════════════════════════════

def page_predictions():
    st.title("Race Predictions")

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

    # Sidebar: pick any round in 2026, including future ones
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

    st.markdown(f"### {race_name} {latest_year} — Round {selected_round}")
    st.markdown(f"**Mode:** {mode_label}")
    st.markdown("---")

    # ── Stage 1: Qualifying prediction ──────────────────────────
    st.markdown("#### Stage 1: Predicted Qualifying Grid")
    try:
        quali_pred = ensemble.predict_quali(race_data, verbose=False)
        q_cols = ["PredictedQualiRank", "Abbreviation", "TeamName"]
        q_cols = [c for c in q_cols if c in quali_pred.columns]
        if "Source" in quali_pred.columns:
            q_cols.append("Source")
        st.dataframe(
            quali_pred[q_cols].rename(columns={"PredictedQualiRank": "Predicted Grid"}),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.warning(f"Could not generate qualifying prediction: {e}")
        quali_pred = None

    st.markdown("---")

    # ── Stage 2: Race prediction ─────────────────────────────────
    st.markdown("#### Stage 2: Predicted Race Result")
    try:
        race_pred = ensemble.predict_race(race_data)
    except Exception as e:
        st.error(f"Could not generate race prediction: {e}")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        for _, row in race_pred.iterrows():
            rank = int(row["FinalRank"])
            driver = row["Abbreviation"]
            team = row["TeamName"]
            elo_val = row["EloRating"]
            win_prob = row.get("WinProb", 0) * 100
            podium_prob = row.get("PodiumProb", 0) * 100
            grid = f"P{int(row['GridPosition'])}" if pd.notna(row.get("GridPosition")) else "-"

            pos_label = f"P{rank}"

            st.markdown(
                f"**{pos_label} {driver}** — {team} "
                f"&nbsp;&nbsp; Grid: {grid} "
                f"&nbsp;&nbsp; Elo: {elo_val:.0f} "
                f"&nbsp;&nbsp; Win: {win_prob:.1f}% "
                f"&nbsp;&nbsp; Podium: {podium_prob:.1f}%"
            )

    with col2:
        st.markdown("#### Win Probability")
        top10 = race_pred.head(10)
        fig, ax = plt.subplots(figsize=(6, 8), facecolor="#0D0D0D")
        ax.set_facecolor("#0D0D0D")
        ax.barh(
            range(len(top10) - 1, -1, -1),
            top10["WinProb"] * 100,
            color="#FF1801",
            alpha=0.8,
            height=0.6,
        )
        ax.set_yticks(range(len(top10) - 1, -1, -1))
        ax.set_yticklabels(top10["Abbreviation"], color="#E0E0E0", fontfamily="monospace")
        ax.set_xlabel("Win Probability (%)", color="#E0E0E0")
        ax.tick_params(colors="#777777")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.grid(axis="x", alpha=0.15)
        st.pyplot(fig)
        plt.close()

    # Model info
    st.markdown("---")
    with st.expander("Model Details"):
        tree_w, elo_w = ensemble._get_blend_weights()
        st.markdown(f"**Blend weights:** XGBoost {tree_w:.0%} / Elo {elo_w:.0%}")
        st.markdown(f"**Races in current era:** {ensemble.elo_model._races_in_era}")
        st.markdown(f"**Training data:** {len(df)} rows, {len(feature_cols)} features")

        st.markdown("#### Top 15 Feature Importance")
        importance = ensemble.tree_model.get_feature_importance(15)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0D0D0D")
        ax.set_facecolor("#0D0D0D")
        ax.barh(range(len(importance) - 1, -1, -1), importance["Importance"],
                color="#FF1801", alpha=0.8)
        ax.set_yticks(range(len(importance) - 1, -1, -1))
        ax.set_yticklabels(importance["Feature"], color="#E0E0E0", fontsize=9)
        ax.tick_params(colors="#777777")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        st.pyplot(fig)
        plt.close()


def _show_elo_only(elo, df):
    """Show Elo rankings when no trained model is available."""
    rankings = elo.get_current_rankings()
    active = df[df["Year"] == df["Year"].max()]["Abbreviation"].unique()
    rankings = rankings[rankings["Abbreviation"].isin(active)]

    st.markdown("#### Current Elo Rankings")
    for i, row in rankings.head(22).iterrows():
        elo_val = row["EloRating"]
        bar = "█" * int((elo_val - 1300) / 8)
        st.text(f"  {i+1:2d}. {row['Abbreviation']:4s}  {elo_val:7.0f}  {bar}")


# ═══════════════════════════════════════════════
# RACE ANALYSIS PAGE
# ═══════════════════════════════════════════════

def page_race_analysis():
    st.title("Race Analysis")

    df = load_features()
    if df is None:
        st.error("No data found.")
        return

    races = get_available_races(df)
    if not races:
        st.warning("No races available.")
        return

    # Race selector
    race_labels = [f"{name} {year}" for year, rnd, name in races]
    selected_idx = st.sidebar.selectbox("Select Race", range(len(race_labels)),
                                         format_func=lambda i: race_labels[i])
    year, rnd, name = races[selected_idx]

    st.markdown(f"### {name} {year}")

    # Load session
    with st.spinner(f"Loading {name} {year}..."):
        try:
            session = load_session(year, rnd, "R")
        except Exception as e:
            st.error(f"Could not load session: {e}")
            return

    from visualizations.race_plots import RacePlots
    plots = RacePlots()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Lap Times", "Position Chart", "Tyre Strategy", "Pace Comparison"])

    with tab1:
        st.markdown("#### Lap Time Distribution")
        fig = plots.lap_time_distribution(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("#### Position Chart")
        fig = plots.position_chart(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown("#### Tyre Strategy")
        fig = plots.tyre_strategy(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("#### Race Pace Comparison")
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
    st.title("Track Visualization")

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
        st.markdown("#### Circuit Map")
        fig = track.draw_track_map(session, save=False)
        st.pyplot(fig)
        plt.close()

    with tab2:
        driver = st.selectbox("Driver", drivers, index=0, key="speed_driver")
        st.markdown(f"#### Speed Heatmap — {driver}")
        fig = track.speed_heatmap(session, driver, save=False)
        st.pyplot(fig)
        plt.close()

    with tab3:
        driver = st.selectbox("Driver", drivers, index=0, key="gear_driver")
        st.markdown(f"#### Gear Map — {driver}")
        fig = track.gear_map(session, driver, save=False)
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("#### Telemetry Comparison")
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
    st.title("Elo Driver Rankings")

    df = load_features()
    if df is None:
        st.error("No data found.")
        return

    from models.elo_rating import F1EloRating
    elo = F1EloRating()
    pre_race = elo.process_historical(df)

    # Current rankings
    rankings = elo.get_current_rankings()
    active = df[df["Year"] == df["Year"].max()]["Abbreviation"].unique()
    rankings_active = rankings[rankings["Abbreviation"].isin(active)].reset_index(drop=True)

    st.markdown("### Current Active Driver Rankings")

    # Visual bar chart
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    n = len(rankings_active)
    colors = []
    for i in range(n):
        if i == 0:
            colors.append("#FFD700")
        elif i == 1:
            colors.append("#C0C0C0")
        elif i == 2:
            colors.append("#CD7F32")
        else:
            colors.append("#FF1801")

    ax.barh(range(n - 1, -1, -1), rankings_active["EloRating"] - 1300,
            color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(n - 1, -1, -1))
    ax.set_yticklabels(
        [f"{r['Abbreviation']}  ({r['EloRating']:.0f})"
         for _, r in rankings_active.iterrows()],
        color="#E0E0E0", fontfamily="monospace", fontsize=11,
    )
    ax.set_xlabel("Elo Rating (above 1300 baseline)", color="#E0E0E0")
    ax.tick_params(colors="#777777")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.grid(axis="x", alpha=0.15)

    st.pyplot(fig)
    plt.close()

    # Win probabilities for next race
    st.markdown("---")
    st.markdown("### Win Probabilities (Next Race)")

    probs = elo.predict_probabilities(active.tolist())

    col1, col2, col3 = st.columns(3)
    for i, (_, row) in enumerate(probs.head(3).iterrows()):
        col = [col1, col2, col3][i]
        rank_label = ["1st", "2nd", "3rd"][i]
        with col:
            st.metric(
                label=f"{rank_label} {row['Abbreviation']}",
                value=f"{row['WinProb']*100:.1f}%",
                delta=f"Elo: {row['EloRating']:.0f}",
            )

    # Full probability table
    with st.expander("Full Win Probability Table"):
        display_df = probs[["Abbreviation", "EloRating", "WinProb", "PodiumProb", "ExpectedPosition"]].copy()
        display_df["WinProb"] = (display_df["WinProb"] * 100).round(1).astype(str) + "%"
        display_df["PodiumProb"] = (display_df["PodiumProb"] * 100).round(1).astype(str) + "%"
        display_df["EloRating"] = display_df["EloRating"].round(0).astype(int)
        display_df["ExpectedPosition"] = display_df["ExpectedPosition"].round(1)
        display_df.columns = ["Driver", "Elo", "Win %", "Podium %", "Expected Pos"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Elo history chart
    st.markdown("---")
    st.markdown("### Elo Rating History")

    history = pd.DataFrame(elo.rating_history, columns=["Year", "Round", "Driver", "Rating"])
    history = history[history["Driver"].isin(active)]

    # Create race index for x-axis
    race_index = history.groupby(["Year", "Round"]).ngroup()
    history["RaceIndex"] = race_index

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0D0D0D")
    ax.set_facecolor("#0D0D0D")

    for driver in rankings_active.head(10)["Abbreviation"]:
        dh = history[history["Driver"] == driver]
        if dh.empty:
            continue
        try:
            color = fastf1.plotting.get_driver_color(driver, session=None)
        except Exception:
            color = "#AAAAAA"
        ax.plot(dh["RaceIndex"], dh["Rating"], label=driver, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Race (chronological)", color="#E0E0E0")
    ax.set_ylabel("Elo Rating", color="#E0E0E0")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.tick_params(colors="#777777")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.grid(alpha=0.15)

    # Mark regulation change
    era_changes = history.groupby("Year")["RaceIndex"].min()
    for year_val in [2026]:
        if year_val in era_changes.index:
            ax.axvline(era_changes[year_val], color="#FF1801", linestyle="--",
                      alpha=0.5, label="2026 Reg Change")

    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# PIPELINE CONTROLS PAGE
# ═══════════════════════════════════════════════

def page_pipeline():
    st.title("Pipeline Controls")

    st.markdown("""
    Use these controls to update your data and retrain the model.
    After each race weekend, run these steps in order.
    """)

    st.markdown("---")

    # Status
    df = load_features()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", len(df))
        col2.metric("Seasons", len(df["Year"].unique()))
        col3.metric("Drivers", df["Abbreviation"].nunique())
        col4.metric("Features", len([c for c in df.columns if c.startswith("Driver_") or c.startswith("Team_")]))

        st.markdown(f"**Latest data:** {df['EventName'].iloc[-1]} {int(df['Year'].iloc[-1])}")
    else:
        st.warning("No data found yet.")

    st.markdown("---")

    # Pipeline steps
    st.markdown("### Run Pipeline Steps")
    st.markdown("Run these from your terminal:")

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

    st.markdown("---")

    # Quick actions
    st.markdown("### Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Refresh Data View", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with col2:
        if st.button("Launch Race Animation", use_container_width=True):
            st.info("Run from terminal: `python main.py --step animate --race Australia --year 2025`")

    # Data preview
    if df is not None:
        st.markdown("---")
        with st.expander("Raw Data Preview"):
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
