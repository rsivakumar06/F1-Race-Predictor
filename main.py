#!/usr/bin/env python3
"""
F1 Race Winner Predictor — Main Entry Point

Usage:
    python main.py --step collect                     # Download all seasons
    python main.py --step collect --year 2026          # Next uncollected race only
    python main.py --step collect --year 2026 --round 9    # One specific round, then stop
    python main.py --step collect --year 2026 --full   # Whole season (resumable)
    python main.py --step features             # Run feature engineering
    python main.py --step train                # Train ML model (Phase 3)
    python main.py --step predict --race Australia --year 2026
    python main.py --step visualize --race Australia --year 2026
    python main.py --step animate --race Australia --year 2026 --driver VER
    python main.py --step all                  # Full pipeline
"""
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config


def step_collect(args):
    """Step 1: Collect data from FastF1 API."""
    from data.pipeline import F1DataPipeline

    pipeline = F1DataPipeline()

    if args.year and args.round:
        # One specific round, persisted into the season + combined files, then stop.
        event_name = _lookup_event_name(args.year, args.round)
        print(f"\nCollecting {args.year} Round {args.round} ({event_name or '...'})...")
        df = pipeline.collect_round(args.year, args.round, event_name or f"Round {args.round}")
        if df.empty:
            print(f"   No session data available for Round {args.round} yet.")
        else:
            _print_entries(df, f"Collected {len(df)} driver entries")

    elif args.year and args.full:
        # Whole season — resumable (skips completed rounds); --force redoes them.
        print(f"\nCollecting full season {args.year}...")
        pipeline.collect_season(args.year, force=args.force)

    elif args.year:
        # DEFAULT: collect only the next race that still needs data, then stop.
        print(f"\nCollecting next uncollected race of {args.year}...")
        df = pipeline.collect_next(args.year, force=args.force)
        if df.empty:
            print("   Up to date, or the next race hasn't run yet — nothing collected.")
        else:
            rnd = int(df["RoundNumber"].iloc[0])
            nm = df["EventName"].iloc[0] if "EventName" in df.columns else f"Round {rnd}"
            _print_entries(df, f"Collected Round {rnd} — {nm} ({len(df)} rows)")

    else:
        print(f"\nCollecting all seasons: {config.ALL_SEASONS}")
        pipeline.collect_all_seasons()

    print("\nData collection complete!")
    print(f"   Cache: {config.CACHE_DIR}")
    print(f"   Data:  {config.DATA_DIR}")


def _lookup_event_name(year, round_number):
    """Best-effort human-readable event name from the FastF1 schedule."""
    try:
        import fastf1
        sched = fastf1.get_event_schedule(year)
        row = sched[sched["RoundNumber"] == round_number]
        if not row.empty:
            return str(row["EventName"].iloc[0])
    except Exception:
        pass
    return ""


def _print_entries(df, header):
    """Print a short summary of collected driver entries (columns that exist)."""
    print(f"\n{header}")
    cols = [c for c in ["Abbreviation", "TeamName", "GridPosition", "FinishPosition"]
            if c in df.columns]
    if cols:
        print(df[cols].to_string(index=False))


def step_features(args):
    """Step 2: Run feature engineering."""
    from data.feature_engineering import FeatureEngineer

    print("\nRunning feature engineering...")
    engineer = FeatureEngineer()
    features = engineer.build_features()

    feature_cols = engineer.get_feature_columns(features)
    print(f"\nFeatures built!")
    print(f"   Total rows:    {len(features)}")
    print(f"   Feature count: {len(feature_cols)}")
    print(f"   Seasons:       {sorted(features['Year'].unique())}")
    print(f"   Drivers:       {features['Abbreviation'].nunique()}")


def step_train(args):
    """Step 3: Train the ML model."""
    from models.ensemble import F1Ensemble
    from data.feature_engineering import FeatureEngineer
    import pandas as pd

    print("\nTraining ensemble model (2-stage: quali + race)...")

    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    engineer = FeatureEngineer()
    feature_cols = engineer.get_race_feature_columns(df)

    ensemble = F1Ensemble()
    metrics = ensemble.train(df, feature_cols, engineer=engineer)
    ensemble.save()

    print("\nModel trained and saved!")


def step_predict(args):
    """Step 4: Predict race results."""
    if not args.year:
        print("Please specify --year for prediction")
        return

    from models.ensemble import F1Ensemble
    from data.feature_engineering import FeatureEngineer
    import pandas as pd

    print(f"\nLoading model and predicting...")

    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    engineer = FeatureEngineer()

    ensemble = F1Ensemble()
    ensemble.load()

    # Filter to requested year
    year_data = df[df["Year"] == args.year]
    if year_data.empty:
        print(f"No data for {args.year}")
        return

    # Select specific round or latest
    if args.round:
        target_round = args.round
        race_data = year_data[year_data["RoundNumber"] == target_round]

        # ── Fallback: synthesize rows from latest available round ──
        if race_data.empty:
            available = sorted(year_data["RoundNumber"].unique())
            latest_round = available[-1]
            print(f"No data for Round {target_round} yet.")
            print(f"   Building forecast using latest data (Round {latest_round})...")

            race_data = year_data[year_data["RoundNumber"] == latest_round].copy()
            race_data["RoundNumber"] = target_round
            race_data["IsPreRace"] = True

            # Clear any race-result-leaking features
            result_cols = [c for c in race_data.columns if any(
                x in c.lower() for x in ["finishpos", "finish_pos", "race_pos", "laps_led", "race_lap"]
            )]
            race_data[result_cols] = None

            # Try to get the event name for the target round
            try:
                import fastf1
                event = fastf1.get_event(args.year, target_round)
                race_data["EventName"] = event["EventName"]
            except Exception:
                race_data["EventName"] = f"Round {target_round}"
    else:
        target_round = int(year_data["RoundNumber"].max())
        race_data = year_data[year_data["RoundNumber"] == target_round]

    # Get race name
    race_name = race_data["EventName"].iloc[0] if "EventName" in race_data.columns else f"Round {target_round}"
    is_pre_race = race_data.get("IsPreRace", pd.Series([False])).any()
    mode = "PRE-RACE (quali + practice only)" if is_pre_race else "POST-RACE (full data)"

    print(f"\n  {race_name} {args.year} — Round {target_round}")
    print(f"  Mode: {mode}")
    print(f"  {len(race_data)} drivers")

    # ── Stage 1: Qualifying prediction (pre-quali) ──
    print(f"\n{'='*60}")
    print("  STAGE 1: PREDICTED QUALIFYING GRID (pre-quali)")
    print(f"{'='*60}")
    quali_pred = ensemble.predict_quali(race_data, verbose=True)
    print(f"\n  {'Pos':>3s}  {'Driver':4s}  {'Team':20s}  {'Source'}")
    print(f"  {'---':>3s}  {'----':4s}  {'----':20s}  {'------'}")
    for _, row in quali_pred.iterrows():
        print(
            f"  P{row['PredictedQualiRank']:2d}  {row['Abbreviation']:4s}  "
            f"{row['TeamName']:20s}  {row.get('Source', '')}"
        )

    # ── Stage 2: Race prediction (post-quali, using predicted grid) ──
    print(f"\n{'='*60}")
    print("  STAGE 2: PREDICTED RACE RESULT (post-quali)")
    print(f"{'='*60}")
    race_pred = ensemble.predict_race(race_data)
    print(f"\n  {'Pos':>3s}  {'Driver':4s}  {'Team':20s}  {'Grid':>4s}  {'Elo':>5s}  {'Win%':>5s}")
    print(f"  {'---':>3s}  {'----':4s}  {'----':20s}  {'----':>4s}  {'---':>5s}  {'----':>5s}")
    for _, row in race_pred.iterrows():
        grid = f"P{int(row['GridPosition'])}" if pd.notna(row.get('GridPosition')) else "  -"
        print(
            f"  P{row['FinalRank']:2d}  {row['Abbreviation']:4s}  "
            f"{row['TeamName']:20s}  {grid:>4s}  "
            f"{row['EloRating']:5.0f}  "
            f"{row.get('WinProb', 0)*100:4.1f}%"
        )


def step_visualize(args):
    """Step 5: Generate visualizations."""
    if not args.race or not args.year:
        print("Please specify --race and --year for visualization")
        return

    from data.pipeline import F1DataPipeline
    from visualizations.race_plots import RacePlots
    from visualizations.telemetry_plots import TelemetryPlots
    from visualizations.track_viz import TrackViz

    pipeline = F1DataPipeline()
    print(f"\nLoading {args.year} {args.race}...")
    session = pipeline.load_session_for_visualization(args.year, args.race, "R")

    plots = RacePlots()
    tel_plots = TelemetryPlots()
    track = TrackViz()

    print("\n  Generating race plots...")
    plots.lap_time_distribution(session)
    plots.position_chart(session)
    plots.tyre_strategy(session)

    print("\n  Generating track visualizations...")
    track.draw_track_map(session)

    # Speed heatmap for the winner
    winner = session.results.sort_values("Position").iloc[0]["Abbreviation"]
    track.speed_heatmap(session, winner)

    # Telemetry comparison for top 2
    top2 = session.results.sort_values("Position").head(2)["Abbreviation"].tolist()
    if len(top2) == 2:
        print(f"\n  Generating telemetry comparison: {top2[0]} vs {top2[1]}...")
        # Load qualifying for telemetry comparison
        try:
            quali = pipeline.load_session_for_visualization(args.year, args.race, "Q")
            tel_plots.speed_trace_comparison(quali, top2[0], top2[1])
            tel_plots.delta_time_on_track(quali, top2[0], top2[1])
        except Exception as e:
            print(f"    Could not load qualifying telemetry: {e}")

    print(f"\nPlots saved to: {config.PLOT_DIR}")


def step_animate(args):
    """Step 6: Animate driver positions on track."""
    if not args.race or not args.year:
        print("Please specify --race and --year for animation")
        return

    from data.pipeline import F1DataPipeline
    from visualizations.track_animation import TrackAnimation

    pipeline = F1DataPipeline()
    print(f"\nLoading {args.year} {args.race}...")
    session = pipeline.load_session_for_visualization(args.year, args.race, "R")

    anim = TrackAnimation()
    print(f"\n  Launching full race replay (60fps, 20x speed)...")
    print(f"  Close the window to exit.\n")
    anim.animate_race(session, speed=20, fps=60)
    print(f"\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Winner Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --step collect              Download all seasons
  python main.py --step collect --year 2026  Download 2026 only
  python main.py --step features             Build features from collected data
  python main.py --step all                  Full pipeline (collect + features)
        """,
    )
    parser.add_argument(
        "--step",
        choices=["collect", "features", "train", "predict", "visualize", "animate", "all"],
        required=True,
        help="Pipeline step to run",
    )
    parser.add_argument("--year", type=int, help="Season year")
    parser.add_argument("--round", type=int, help="Round number (requires --year)")
    parser.add_argument("--race", type=str, help="Race name (e.g., 'Australia')")
    parser.add_argument("--driver", type=str, help="Driver abbreviation (e.g., 'VER')")
    parser.add_argument("--force", action="store_true",
                        help="Re-collect rounds already on disk (e.g. to upgrade "
                             "results-only rounds once the API is healthy)")
    parser.add_argument("--full", action="store_true",
                        help="With --year: collect/refresh the whole season instead "
                             "of just the next uncollected race")

    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Race Winner Predictor")
    print("=" * 60)

    start = time.time()

    if args.step == "all":
        step_collect(args)
        step_features(args)
        step_train(args)
    elif args.step == "collect":
        step_collect(args)
    elif args.step == "features":
        step_features(args)
    elif args.step == "train":
        step_train(args)
    elif args.step == "predict":
        step_predict(args)
    elif args.step == "visualize":
        step_visualize(args)
    elif args.step == "animate":
        step_animate(args)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()