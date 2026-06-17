"""
Feature Engineering — Full Phase 2 Implementation.

Transforms raw race data into a comprehensive ML-ready feature set including:

DRIVER FEATURES (regulation-agnostic — carry across eras):
  - Rolling pace, consistency, finish position (3/5/10 race windows)
  - First-lap performance (positions gained on lap 1 — elite starters)
  - Overtaking rate (positions gained per race after lap 1)
  - Wet weather delta (performance difference in rain vs dry)
  - Qualifying conversion (quali position → race finish tendency)
  - Season momentum (winning streak, points slope, form trajectory)
  - Teammate head-to-head (separates driver skill from car performance)

TEAM FEATURES:
  - Rolling team form (avg finish, points trajectory)
  - Pit stop speed and consistency
  - Reliability rate (mechanical DNF history)
  - Constructor championship position (proxy for car quality)

CIRCUIT FEATURES:
  - Circuit type encoding (street/technical/high-speed/balanced)
  - Corner count, straight ratio
  - Driver track-specific history (some drivers dominate certain tracks)

2026-SPECIFIC FEATURES (computed when 2026 data available):
  - Energy management proxy (braking-zone-to-straight ratio per circuit)
  - Tyre degradation profiles per compound
  - Active aero straight speed effectiveness (approximated from telemetry)

Usage:
    from data.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    features_df = engineer.build_features("all_races_raw.parquet")
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class FeatureEngineer:
    """Builds comprehensive ML features from raw race data."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or config.DATA_DIR

    def build_features(self, input_file: str = "all_races_raw.parquet") -> pd.DataFrame:
        """
        Full feature engineering pipeline — runs all feature groups in order.

        Args:
            input_file: Name of raw data parquet file in data_dir.

        Returns:
            DataFrame with all features, one row per driver per race.
        """
        raw = pd.read_parquet(self.data_dir / input_file)

        # Sort chronologically — critical for rolling features
        raw = raw.sort_values(["Year", "RoundNumber", "Abbreviation"]).reset_index(drop=True)
        df = raw.copy()

        print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
        print(f"Seasons: {sorted(df['Year'].unique())}")
        print(f"Drivers: {df['Abbreviation'].nunique()}")

        # ── Step 1: Driver rolling performance ──
        print("  [1/10] Driver rolling performance...")
        df = self._add_driver_rolling_features(df)

        # ── Step 2: First-lap and overtaking ──
        print("  [2/10] First-lap and overtaking features...")
        df = self._add_first_lap_features(df)

        # ── Step 3: Wet weather performance ──
        print("  [3/10] Wet weather delta...")
        df = self._add_wet_weather_features(df)

        # ── Step 4: Teammate head-to-head ──
        print("  [4/10] Teammate head-to-head...")
        df = self._add_teammate_features(df)

        # ── Step 5: Season momentum ──
        print("  [5/10] Season momentum...")
        df = self._add_momentum_features(df)

        # ── Step 6: Team features ──
        print("  [6/10] Team rolling features...")
        df = self._add_team_rolling_features(df)

        # ── Step 7: Qualifying features ──
        print("  [7/12] Qualifying features...")
        df = self._add_qualifying_features(df)

        # ── Step 8: Circuit encoding ──
        print("  [8/12] Circuit features...")
        df = self._encode_circuit_features(df)
        df = self._add_track_history_features(df)

        # ── Step 9: Grid position features ──
        print("  [9/12] Grid position features...")
        df = self._add_grid_features(df)

        # ── Step 10: 2026-specific features ──
        print("  [10/12] Era-specific features...")
        df = self._add_era_specific_features(df)

        # ── Step 11: Practice-based features ──
        print("  [11/12] Practice session features...")
        df = self._add_practice_features(df)

        # ── Step 12: Quali-to-race adjustment features ──
        print("  [12/12] Quali-to-race adjustment features...")
        df = self._add_quali_race_adjustment_features(df)

        # ── Target variables ──
        # Qualifying target
        df["Target_QualifyingPosition"] = df["QualifyingPosition"]

        # Race targets
        df["Target_FinishPosition"] = df["FinishPosition"]
        # Race delta = finish - grid. Modelling this (instead of absolute finish)
        # forces the model to learn deviation from the grid order, which is the
        # only thing that can beat the grid baseline. Predicted finish is then
        # reconstructed as grid + predicted_delta. NaN for pit-lane/grid-0 starts
        # so they don't poison the target or the reconstruction.
        valid_grid = df["GridPosition"] > 0
        df["Target_RaceDelta"] = np.where(
            valid_grid, df["FinishPosition"] - df["GridPosition"], np.nan
        )
        df["Target_IsWinner"] = (df["FinishPosition"] == 1).astype(int)
        df["Target_IsPodium"] = (df["FinishPosition"] <= 3).astype(int)
        df["Target_IsPointsFinish"] = (df["FinishPosition"] <= 10).astype(int)

        # ── Save ──
        output_path = self.data_dir / "features.parquet"
        df.to_parquet(output_path, index=False)

        quali_cols = self.get_quali_feature_columns(df)
        race_cols = self.get_race_feature_columns(df)
        print(f"\nFeatures saved: {output_path}")
        print(f"  Total rows:       {len(df)}")
        print(f"  Total columns:    {len(df.columns)}")
        print(f"  Quali features:   {len(quali_cols)}")
        print(f"  Race features:    {len(race_cols)}")

        return df

    # ═══════════════════════════════════════════
    # 1. DRIVER ROLLING PERFORMANCE
    # ═══════════════════════════════════════════

    def _add_driver_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling driver performance statistics at 3 window sizes.

        All features are SHIFTED by 1 — they represent the driver's form
        going INTO the race, not including the current race's result.
        This prevents data leakage.
        """
        for window_name, window_size in config.ROLLING_WINDOWS.items():
            prefix = f"Driver_Roll{window_name}"

            rolling_specs = [
                ("FinishPosition", "AvgFinish", "mean"),
                ("FinishPosition", "BestFinish", "min"),
                ("FinishPosition", "WorstFinish", "max"),
                ("AvgLapTime_s", "AvgPace", "mean"),
                ("LapTimeConsistency", "Consistency", "mean"),
                ("IsFinished", "CompletionRate", "mean"),
                ("Points", "AvgPoints", "mean"),
                ("Points", "TotalPoints", "sum"),
                ("PaceToMedianRatio", "RelativePace", "mean"),
                ("TyreDegPerLap", "AvgTyreDeg", "mean"),
            ]

            for col, agg_name, agg_func in rolling_specs:
                if col not in df.columns:
                    continue
                feature_name = f"{prefix}_{agg_name}"
                df[feature_name] = (
                    df.groupby("Abbreviation")[col]
                    .transform(
                        lambda x: x.rolling(window_size, min_periods=1).agg(agg_func).shift(1)
                    )
                )

        # ── Positions gained (grid to finish) — overall ──
        df["PositionsGained"] = df["GridPosition"] - df["FinishPosition"]
        df["Driver_AvgPositionsGained"] = (
            df.groupby("Abbreviation")["PositionsGained"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )

        # ── Finish position variance (high variance = unpredictable driver) ──
        df["Driver_FinishVariance"] = (
            df.groupby("Abbreviation")["FinishPosition"]
            .transform(lambda x: x.rolling(5, min_periods=2).var().shift(1))
        )

        return df

    # ═══════════════════════════════════════════
    # 2. FIRST-LAP & OVERTAKING
    # ═══════════════════════════════════════════

    def _add_first_lap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        First-lap ability is a distinct skill — some drivers are elite starters.
        Overtaking rate after lap 1 captures racecraft separately from start ability.
        """
        if "FirstLapPositionsGained" in df.columns:
            df["Driver_RollFirstLapGain"] = (
                df.groupby("Abbreviation")["FirstLapPositionsGained"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )
            # Consistency of first-lap performance (std dev)
            df["Driver_FirstLapConsistency"] = (
                df.groupby("Abbreviation")["FirstLapPositionsGained"]
                .transform(lambda x: x.rolling(5, min_periods=2).std().shift(1))
            )

        if "OvertakesAfterLap1" in df.columns:
            df["Driver_RollOvertakeRate"] = (
                df.groupby("Abbreviation")["OvertakesAfterLap1"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        return df

    # ═══════════════════════════════════════════
    # 3. WET WEATHER PERFORMANCE
    # ═══════════════════════════════════════════

    def _add_wet_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some drivers excel in the rain (Hamilton, Verstappen historically).
        Compute separate dry/wet rolling averages, then the delta.
        """
        if "WeatherHadRain" not in df.columns:
            return df

        # Flag wet races
        df["IsWetRace"] = df["WeatherHadRain"].astype(int)

        # Separate dry and wet finish histories per driver
        # We compute an expanding (all-history) average since wet races are rare
        df["_DryFinish"] = df["FinishPosition"].where(~df["WeatherHadRain"])
        df["_WetFinish"] = df["FinishPosition"].where(df["WeatherHadRain"])

        df["Driver_DryAvgFinish"] = (
            df.groupby("Abbreviation")["_DryFinish"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )
        df["Driver_WetAvgFinish"] = (
            df.groupby("Abbreviation")["_WetFinish"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )

        # Wet delta: negative = better in wet, positive = worse in wet
        df["Driver_WetDelta"] = df["Driver_WetAvgFinish"] - df["Driver_DryAvgFinish"]

        # Fill NaN (driver hasn't raced in rain yet) with 0 (assume neutral)
        df["Driver_WetDelta"] = df["Driver_WetDelta"].fillna(0)

        # Clean up temp columns
        df.drop(columns=["_DryFinish", "_WetFinish"], inplace=True)

        return df

    # ═══════════════════════════════════════════
    # 4. TEAMMATE HEAD-TO-HEAD
    # ═══════════════════════════════════════════

    def _add_teammate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Teammate comparison is the best way to separate driver from car.
        If a driver consistently beats their teammate, they're extracting
        more from the car than the machinery alone provides.

        Vectorized: self-merge on (Year, Round, Team) and keep the other
        driver. Only teams with exactly two drivers in a race get a teammate
        (mid-season swaps producing 3 names in one weekend are left NaN).
        """
        keys = ["Year", "RoundNumber", "TeamName"]
        nunique = df.groupby(keys)["Abbreviation"].transform("nunique")

        sub = df[keys + ["Abbreviation", "FinishPosition",
                         "BestQualiTime_s", "AvgLapTime_s"]].copy()
        sub["_pair_ok"] = (nunique == 2).values

        pair = sub.merge(sub, on=keys, suffixes=("", "_tm"))
        pair = pair[(pair["Abbreviation"] != pair["Abbreviation_tm"]) & pair["_pair_ok"]]
        pair = pair.drop_duplicates(subset=keys + ["Abbreviation"])

        tm = df[keys + ["Abbreviation"]].merge(
            pair[keys + ["Abbreviation", "FinishPosition_tm",
                         "BestQualiTime_s_tm", "AvgLapTime_s_tm"]],
            on=keys + ["Abbreviation"], how="left",
        )

        both_fin = df["FinishPosition"].notna().values & tm["FinishPosition_tm"].notna().values
        df["BeatTeammate"] = np.where(
            both_fin,
            (df["FinishPosition"].values < tm["FinishPosition_tm"].values).astype(float),
            np.nan,
        )
        df["QualiGapToTeammate_s"] = df["BestQualiTime_s"].values - tm["BestQualiTime_s_tm"].values
        df["PaceGapToTeammate_s"] = df["AvgLapTime_s"].values - tm["AvgLapTime_s_tm"].values

        # Rolling teammate-beat rate (how often does this driver beat their teammate?)
        df["Driver_TeammateWinRate"] = (
            df.groupby("Abbreviation")["BeatTeammate"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )

        # Rolling quali advantage over teammate
        df["Driver_RollQualiVsTeammate"] = (
            df.groupby("Abbreviation")["QualiGapToTeammate_s"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )

        # Rolling race pace advantage over teammate
        df["Driver_RollPaceVsTeammate"] = (
            df.groupby("Abbreviation")["PaceGapToTeammate_s"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )

        return df

    # ═══════════════════════════════════════════
    # 5. SEASON MOMENTUM
    # ═══════════════════════════════════════════

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Captures whether a driver/team is on an upward or downward trajectory.
        Winning streaks, podium streaks, and the slope of recent points.
        """
        # ── Winning / podium streaks (consecutive results going INTO this race) ──
        df["_IsWin"] = (df["FinishPosition"] == 1).astype(int)
        df["_IsPodium"] = (df["FinishPosition"] <= 3).astype(int)

        def _trailing_streak(flag: pd.Series) -> pd.Series:
            # run = length of the current consecutive run of 1s ending at each row;
            # shift(1) turns it into "streak coming into this race" (0 at season/career start).
            run = flag.groupby((flag != flag.shift()).cumsum()).cumcount() + 1
            run = run.where(flag == 1, 0)
            return run.shift(1).fillna(0)

        df = df.sort_values(["Abbreviation", "Year", "RoundNumber"])
        df["Driver_WinStreak"] = (
            df.groupby("Abbreviation")["_IsWin"].transform(_trailing_streak).astype(int)
        )
        df["Driver_PodiumStreak"] = (
            df.groupby("Abbreviation")["_IsPodium"].transform(_trailing_streak).astype(int)
        )
        df = df.sort_values(["Year", "RoundNumber", "Abbreviation"])

        # ── Points trajectory slope ──
        def _rolling_slope(series, window=5):
            """Compute slope of a rolling linear regression."""
            slopes = []
            for i in range(len(series)):
                if i < 2:
                    slopes.append(0.0)
                    continue
                start = max(0, i - window)
                window_data = series.iloc[start:i].dropna()
                if len(window_data) < 2:
                    slopes.append(0.0)
                    continue
                x = np.arange(len(window_data))
                y = window_data.values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            return slopes

        slopes_list = []
        for driver, group in df.groupby("Abbreviation"):
            group = group.sort_values(["Year", "RoundNumber"])
            slopes = _rolling_slope(group["Points"], window=5)
            for idx, slope in zip(group.index, slopes):
                slopes_list.append((idx, slope))

        slope_series = pd.Series(
            {idx: val for idx, val in slopes_list}, name="Driver_PointsSlope"
        )
        df["Driver_PointsSlope"] = slope_series

        # ── Recent form vs season average ──
        df["Driver_RecentVsSeasonForm"] = (
            df.groupby("Abbreviation")["FinishPosition"]
            .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        ) - (
            df.groupby("Abbreviation")["FinishPosition"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )

        df.drop(columns=["_IsWin", "_IsPodium"], inplace=True)
        return df

    # ═══════════════════════════════════════════
    # 6. TEAM FEATURES
    # ═══════════════════════════════════════════

    def _add_team_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Team-level rolling statistics across both drivers."""

        # Team average finish per race
        team_race_avg = (
            df.groupby(["Year", "RoundNumber", "TeamName"])["FinishPosition"]
            .mean().reset_index()
            .rename(columns={"FinishPosition": "TeamRaceAvgFinish"})
        )
        df = df.merge(team_race_avg, on=["Year", "RoundNumber", "TeamName"], how="left")

        # Team points per race
        team_race_points = (
            df.groupby(["Year", "RoundNumber", "TeamName"])["Points"]
            .sum().reset_index()
            .rename(columns={"Points": "TeamRacePoints"})
        )
        df = df.merge(team_race_points, on=["Year", "RoundNumber", "TeamName"], how="left")

        # Rolling team form
        df["Team_RollAvgFinish"] = (
            df.groupby("Abbreviation")["TeamRaceAvgFinish"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )
        df["Team_RollAvgPoints"] = (
            df.groupby("Abbreviation")["TeamRacePoints"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )

        # Team reliability rate
        team_reliability = (
            df.groupby(["Year", "RoundNumber", "TeamName"])["IsMechanicalDNF"]
            .mean().reset_index()
            .rename(columns={"IsMechanicalDNF": "TeamDNFRate"})
        )
        df = df.merge(team_reliability, on=["Year", "RoundNumber", "TeamName"], how="left")
        df["Team_RollReliability"] = (
            df.groupby("Abbreviation")["TeamDNFRate"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )

        # Team pit stop performance
        if "TotalPitTime_s" in df.columns and "NumPitStops" in df.columns:
            df["AvgPitStopDuration"] = df["TotalPitTime_s"] / df["NumPitStops"].replace(0, np.nan)
            team_pit = (
                df.groupby(["Year", "RoundNumber", "TeamName"])["AvgPitStopDuration"]
                .mean().reset_index()
                .rename(columns={"AvgPitStopDuration": "TeamAvgPitDuration"})
            )
            df = df.merge(team_pit, on=["Year", "RoundNumber", "TeamName"], how="left")
            df["Team_RollPitSpeed"] = (
                df.groupby("Abbreviation")["TeamAvgPitDuration"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )
            df["Team_RollPitConsistency"] = (
                df.groupby("Abbreviation")["TeamAvgPitDuration"]
                .transform(lambda x: x.rolling(5, min_periods=2).std().shift(1))
            )

        # Cumulative team points within season
        team_cumpoints = df.groupby(["Year", "TeamName"])["TeamRacePoints"].cumsum()
        df["Team_CumulativePoints"] = team_cumpoints

        return df

    # ═══════════════════════════════════════════
    # 7. QUALIFYING FEATURES
    # ═══════════════════════════════════════════

    def _add_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Qualifying performance and conversion rate."""

        if "QualifyingPosition" in df.columns:
            df["Driver_RollQualiPosition"] = (
                df.groupby("Abbreviation")["QualifyingPosition"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )
            df["Driver_QualiConsistency"] = (
                df.groupby("Abbreviation")["QualifyingPosition"]
                .transform(lambda x: x.rolling(5, min_periods=2).std().shift(1))
            )

        if "QualiGapToPole_s" in df.columns:
            df["Driver_RollQualiGapToPole"] = (
                df.groupby("Abbreviation")["QualiGapToPole_s"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )
            # Percentage gap to pole (normalizes across circuit lengths)
            if "BestQualiTime_s" in df.columns:
                pole_times = df.groupby(["Year", "RoundNumber"])["BestQualiTime_s"].transform("min")
                df["QualiGapToPole_pct"] = (
                    (df["BestQualiTime_s"] - pole_times) / pole_times * 100
                )
                df["Driver_RollQualiGapPct"] = (
                    df.groupby("Abbreviation")["QualiGapToPole_pct"]
                    .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
                )

        if "QualifyingPosition" in df.columns:
            df["QualiToRaceGain"] = df["QualifyingPosition"] - df["FinishPosition"]
            df["Driver_RollQualiConversion"] = (
                df.groupby("Abbreviation")["QualiToRaceGain"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        return df

    # ═══════════════════════════════════════════
    # 8. CIRCUIT FEATURES & TRACK HISTORY
    # ═══════════════════════════════════════════

    def _encode_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode circuit type and normalize numeric circuit features."""

        if "CircuitType" in df.columns:
            circuit_dummies = pd.get_dummies(df["CircuitType"], prefix="Circuit")
            df = pd.concat([df, circuit_dummies], axis=1)

        if "CircuitCorners" in df.columns:
            mean_c = df["CircuitCorners"].mean()
            std_c = df["CircuitCorners"].std()
            df["CircuitCorners_norm"] = (df["CircuitCorners"] - mean_c) / max(std_c, 0.001)

        if "CircuitStraightRatio" in df.columns:
            mean_s = df["CircuitStraightRatio"].mean()
            std_s = df["CircuitStraightRatio"].std()
            df["CircuitStraightRatio_norm"] = (df["CircuitStraightRatio"] - mean_s) / max(std_s, 0.001)

        return df

    def _add_track_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Driver's historical performance at this specific track.
        Some drivers dominate certain circuits (e.g., Hamilton at Silverstone).
        """
        if "EventName" not in df.columns:
            return df

        df["Driver_TrackAvgFinish"] = (
            df.groupby(["Abbreviation", "EventName"])["FinishPosition"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )
        df["Driver_TrackExperience"] = (
            df.groupby(["Abbreviation", "EventName"]).cumcount()
        )
        df["Driver_TrackBestFinish"] = (
            df.groupby(["Abbreviation", "EventName"])["FinishPosition"]
            .transform(lambda x: x.expanding(min_periods=1).min().shift(1))
        )

        return df

    # ═══════════════════════════════════════════
    # 9. GRID POSITION FEATURES
    # ═══════════════════════════════════════════

    def _add_grid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features derived from starting grid position."""

        df["IsPoleSitter"] = (df["GridPosition"] == 1).astype(int)
        df["IsFrontRow"] = (df["GridPosition"] <= 2).astype(int)
        df["IsTopThreeGrid"] = (df["GridPosition"] <= 3).astype(int)
        df["IsTopTenGrid"] = (df["GridPosition"] <= 10).astype(int)

        # Grid position relative to field size (normalized 0-1)
        max_grid = df.groupby(["Year", "RoundNumber"])["GridPosition"].transform("max")
        df["GridPosition_norm"] = df["GridPosition"] / max_grid.clip(lower=1)

        # Grid position bucket
        df["GridBucket"] = pd.cut(
            df["GridPosition"],
            bins=[0, 3, 6, 10, 15, 22],
            labels=["front", "mid_front", "midfield", "mid_back", "back"],
        )
        grid_dummies = pd.get_dummies(df["GridBucket"], prefix="GridBucket")
        df = pd.concat([df, grid_dummies], axis=1)

        return df

    # ═══════════════════════════════════════════
    # 10. ERA-SPECIFIC FEATURES (2026)
    # ═══════════════════════════════════════════

    def _add_era_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features specific to the 2026 regulation era.

        Energy management is the defining challenge of 2026:
        - MGU-K tripled in power, but battery capacity is limited
        - Active aero changes drag profile on straights vs corners
        - Circuits with more braking zones offer more harvest opportunities
        """
        # Era flag
        df["IsNewEra2026"] = (df["Year"] >= 2026).astype(int)

        # Energy management proxy: braking-to-straight ratio per circuit
        if "CircuitCorners" in df.columns and "CircuitStraightRatio" in df.columns:
            df["Circuit_BrakingToStraightRatio"] = (
                df["CircuitCorners"] * (1 - df["CircuitStraightRatio"])
                / df["CircuitStraightRatio"].clip(lower=0.01)
            )

        # Tyre degradation profile
        if "TyreDegPerLap" in df.columns:
            df["Driver_RollTyreDeg"] = (
                df.groupby("Abbreviation")["TyreDegPerLap"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        # Stint management
        if "NumStints" in df.columns:
            df["Driver_RollAvgStints"] = (
                df.groupby("Abbreviation")["NumStints"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        return df

    # ═══════════════════════════════════════════
    # 11. PRACTICE SESSION FEATURES
    # ═══════════════════════════════════════════

    def _add_practice_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features derived from free practice sessions.

        Practice data gives us THIS WEEKEND's pace before quali/race.
        Short runs approximate qualifying speed, long runs approximate race pace.
        The delta between them captures tyre degradation characteristics.
        """
        # ── Practice short-run gap to best (proxy for raw car speed) ──
        if "Practice_ShortRunGapToBest_s" in df.columns:
            df["Practice_ShortRunGapNorm"] = (
                df["Practice_ShortRunGapToBest_s"]
                / df["Practice_BestShortRun_s"].clip(lower=60) * 100  # as percentage
            )

        # ── Practice long-run gap to best (proxy for race pace) ──
        if "Practice_LongRunGapToBest_s" in df.columns:
            df["Practice_LongRunGapNorm"] = (
                df["Practice_LongRunGapToBest_s"]
                / df["Practice_BestLongRun_s"].clip(lower=60) * 100
            )

        # ── Short vs long delta: how much pace a car loses on long runs ──
        # High delta = good quali car but poor race pace (tyre deg / fuel load)
        # Low delta = consistent pace, good tyre management
        if "Practice_ShortLongDelta_s" in df.columns:
            field_avg_delta = df.groupby(["Year", "RoundNumber"])["Practice_ShortLongDelta_s"].transform("mean")
            df["Practice_ShortLongDeltaVsField"] = df["Practice_ShortLongDelta_s"] - field_avg_delta

        # ── Rolling practice performance (are they consistently fast in practice?) ──
        if "Practice_ShortRunGapToBest_s" in df.columns:
            df["Driver_RollPracticeShortRun"] = (
                df.groupby("Abbreviation")["Practice_ShortRunGapToBest_s"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        if "Practice_LongRunGapToBest_s" in df.columns:
            df["Driver_RollPracticeLongRun"] = (
                df.groupby("Abbreviation")["Practice_LongRunGapToBest_s"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        # ── Practice-to-quali conversion (do they typically improve from FP to Q?) ──
        if "Practice_BestShortRun_s" in df.columns and "BestQualiTime_s" in df.columns:
            df["PracticeToQualiImprovement_s"] = (
                df["Practice_BestShortRun_s"] - df["BestQualiTime_s"]
            )
            df["Driver_RollPracticeToQuali"] = (
                df.groupby("Abbreviation")["PracticeToQualiImprovement_s"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        return df

    # ═══════════════════════════════════════════
    # 12. QUALI-TO-RACE ADJUSTMENT FEATURES
    # ═══════════════════════════════════════════

    def _add_quali_race_adjustment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features that capture how qualifying results translate to race results.

        Some cars are fast in quali but degrade in races (e.g. one-lap wonders).
        Others qualify poorly but have great race pace (e.g. strong tyre management).
        The Ferrari turbo example: slower in quali but great off the line.
        """
        # ── Historical quali-to-race delta per driver ──
        # Positive = typically finishes ahead of their grid position (race car)
        # Negative = typically drops back from their grid position (quali car)
        if "GridPosition" in df.columns and "FinishPosition" in df.columns:
            df["QualiRaceDelta"] = df["GridPosition"] - df["FinishPosition"]

            # Rolling average: is this driver typically a race-day gainer or loser?
            df["Driver_RollQualiRaceDelta"] = (
                df.groupby("Abbreviation")["QualiRaceDelta"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

            # Team-level: does this car tend to be better in race trim vs quali?
            team_delta = (
                df.groupby(["Year", "RoundNumber", "TeamName"])["QualiRaceDelta"]
                .mean().reset_index()
                .rename(columns={"QualiRaceDelta": "Team_QualiRaceDelta"})
            )
            df = df.merge(team_delta, on=["Year", "RoundNumber", "TeamName"], how="left")
            df["Team_RollQualiRaceDelta"] = (
                df.groupby("Abbreviation")["Team_QualiRaceDelta"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

        # ── Practice long-run vs short-run as predictor of race vs quali strength ──
        if "Practice_ShortLongDelta_s" in df.columns:
            # High delta = car loses more pace over long runs = worse race car relative to quali
            df["Practice_RacePaceIndicator"] = -df["Practice_ShortLongDelta_s"]
            # Positive = good race car, negative = drops off

        # ── First-lap gain tendency (matters more when starting further back) ──
        if "Driver_RollFirstLapGain" in df.columns and "GridPosition" in df.columns:
            df["ExpectedFirstLapGain"] = (
                df["Driver_RollFirstLapGain"] * (df["GridPosition"] / 10).clip(upper=2)
            )
            # Drivers starting further back have more opportunity to gain on lap 1

        return df

    # ═══════════════════════════════════════════
    # FEATURE COLUMN SELECTION — SPLIT MODELS
    # ═══════════════════════════════════════════

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Return ALL feature columns (backward compatible)."""
        return self.get_race_feature_columns(df)

    def get_quali_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Features available BEFORE qualifying happens.
        Used to predict the qualifying grid.

        Includes: historical driver/team form, practice data, circuit features,
                  track history, momentum. Excludes: anything from quali or race.
        """
        exclude_prefixes = [
            "Target_", "Abbreviation", "TeamName", "EventName",
            "Status", "DriverNumber", "RegulationEra", "CircuitType",
            "GridBucket",
        ]
        exclude_exact = [
            "Year", "RoundNumber",
            "IsFinished", "IsMechanicalDNF", "IsCrashDNF",
            "LapTime_s", "TeamRaceAvgFinish", "TeamDNFRate",
            "QualiToRaceGain", "PositionsGained",
            "BeatTeammate", "QualiGapToTeammate_s", "PaceGapToTeammate_s",
            "TeamRacePoints", "TeamAvgPitDuration", "Team_CumulativePoints",
            "QualiGapToPole_pct", "QualiRaceDelta", "Team_QualiRaceDelta",
            "PracticeToQualiImprovement_s",
            # In-race weather is observed during the race we're predicting — leakage.
            # Keep only the shifted historical wet-form features (Driver_Wet*).
            "WeatherAvgAirTemp", "WeatherAvgTrackTemp", "WeatherAvgHumidity",
            "WeatherAvgWindSpeed", "WeatherHadRain", "IsWetRace",
            # These are race-day only:
            "GridPosition", "FinishPosition", "Points",
            "QualifyingPosition", "BestQualiTime_s", "QualiGapToPole_s",
            "AvgLapTime_s", "LapTimeStd_s", "LapTimeConsistency",
            "FastestLap_s", "NumLaps", "NumPitStops", "TotalPitTime_s",
            "FirstLapPositionsGained", "OvertakesAfterLap1",
            "PaceToMedianRatio", "TyreDegPerLap", "NumStints",
            "AvgPitStopDuration",
        ]
        # Also exclude any Grid or Quali-derived feature that uses the CURRENT
        # session's actual result (genuine leakage). NOTE: the rolling, shift(1)
        # historical quali-form features (Driver_RollQualiPosition,
        # _QualiConsistency, _RollQualiGapToPole, _RollQualiGapPct,
        # _RollQualiConversion, _RollQualiVsTeammate) use only PRIOR races and are
        # leakage-free + highly predictive of the next quali, so they stay IN.
        exclude_contains = [
            # Current-session grid/quali results:
            "IsPoleSitter", "IsFrontRow", "IsTopThreeGrid", "IsTopTenGrid",
            "GridPosition_norm", "GridBucket_",
            # Current-race-derived:
            "QualiToRaceGain", "ExpectedFirstLapGain",
            # Quali->race conversion (about race movement, not quali pace):
            "Driver_RollQualiRaceDelta", "Team_RollQualiRaceDelta",
        ]

        feature_cols = []
        for col in df.columns:
            if any(col.startswith(p) for p in exclude_prefixes):
                continue
            if col in exclude_exact:
                continue
            if col in exclude_contains:
                continue
            if df[col].dtype in ["float64", "float32", "int64", "int32", "uint8", "bool"]:
                feature_cols.append(col)

        return sorted(feature_cols)

    def get_race_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Features available AFTER qualifying but BEFORE the race.
        Used to predict race finishing positions.

        Includes everything from quali features PLUS:
        actual qualifying position, grid features, quali-race adjustment features.
        Excludes: race results (finish position, lap times, etc.)
        """
        exclude_prefixes = [
            "Target_", "Abbreviation", "TeamName", "EventName",
            "Status", "DriverNumber", "RegulationEra", "CircuitType",
            "GridBucket",
        ]
        exclude_exact = [
            "Year", "RoundNumber",
            "IsFinished", "IsMechanicalDNF", "IsCrashDNF",
            "LapTime_s", "TeamRaceAvgFinish", "TeamDNFRate",
            "QualiToRaceGain", "PositionsGained",
            "BeatTeammate", "QualiGapToTeammate_s", "PaceGapToTeammate_s",
            "TeamRacePoints", "TeamAvgPitDuration", "Team_CumulativePoints",
            "QualiGapToPole_pct", "QualiRaceDelta", "Team_QualiRaceDelta",
            "PracticeToQualiImprovement_s",
            # In-race weather is observed during the race we're predicting — leakage.
            # Keep only the shifted historical wet-form features (Driver_Wet*).
            "WeatherAvgAirTemp", "WeatherAvgTrackTemp", "WeatherAvgHumidity",
            "WeatherAvgWindSpeed", "WeatherHadRain", "IsWetRace",
            # These are post-race only:
            "FinishPosition", "Points",
            "AvgLapTime_s", "LapTimeStd_s", "LapTimeConsistency",
            "FastestLap_s", "NumLaps", "NumPitStops", "TotalPitTime_s",
            "FirstLapPositionsGained", "OvertakesAfterLap1",
            "PaceToMedianRatio", "TyreDegPerLap", "NumStints",
            "AvgPitStopDuration",
        ]

        feature_cols = []
        for col in df.columns:
            if any(col.startswith(p) for p in exclude_prefixes):
                continue
            if col in exclude_exact:
                continue
            if df[col].dtype in ["float64", "float32", "int64", "int32", "uint8", "bool"]:
                feature_cols.append(col)

        return sorted(feature_cols)

    def get_feature_groups(self, df: pd.DataFrame) -> dict:
        """Return feature columns organized by category (useful for analysis)."""
        feature_cols = self.get_feature_columns(df)

        groups = {
            "driver_rolling": [c for c in feature_cols if c.startswith("Driver_Roll")],
            "driver_skill": [
                c for c in feature_cols
                if c.startswith("Driver_") and not c.startswith("Driver_Roll")
            ],
            "team": [c for c in feature_cols if c.startswith("Team_")],
            "circuit": [c for c in feature_cols if c.startswith("Circuit")],
            "grid": [c for c in feature_cols if "Grid" in c or c.startswith("Is") and "Grid" in c],
            "weather": [c for c in feature_cols if "Weather" in c or "Wet" in c],
            "qualifying": [
                c for c in feature_cols
                if "Quali" in c and not c.startswith("Driver_")
            ],
            "era_specific": [
                c for c in feature_cols
                if "2026" in c or "BrakingToStraight" in c or "TyreDeg" in c or "Stints" in c
            ],
        }

        # Catch anything not in a group
        all_grouped = set(sum(groups.values(), []))
        groups["other"] = [c for c in feature_cols if c not in all_grouped]

        return groups


# ─────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engineer = FeatureEngineer()
    features = engineer.build_features()

    print(f"\n{'='*60}")
    print("FEATURE GROUPS")
    print(f"{'='*60}")
    groups = engineer.get_feature_groups(features)
    for group_name, cols in groups.items():
        print(f"\n  {group_name} ({len(cols)} features):")
        for col in cols:
            non_null = features[col].notna().sum()
            print(f"    {col:45s}  ({non_null}/{len(features)} non-null)")
