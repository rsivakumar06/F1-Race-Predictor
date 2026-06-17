"""
Data Pipeline — Ingest F1 data from the FastF1 API.

This module handles:
1. Session loading with caching for fast re-runs
2. Extracting structured DataFrames for laps, results, weather, telemetry
3. Computing circuit characteristics from position data
4. Saving processed data for downstream feature engineering

Usage:
    from data.pipeline import F1DataPipeline
    pipeline = F1DataPipeline()
    pipeline.collect_all_seasons()        # Full download
    pipeline.collect_season(2026)         # Single season
    pipeline.collect_race(2026, 'Australia')  # Single race
"""
import sys
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.helpers import (
    safe_timedelta_col_to_seconds,
    clean_lap_data,
    classify_circuit,
    compute_straight_ratio,
    compute_consistency,
)
from utils.constants import REGULATION_ERAS, MECHANICAL_DNF_STATUSES, CRASH_DNF_STATUSES

# Suppress verbose FastF1 logging during bulk collection
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class F1DataPipeline:
    """
    Collects and processes F1 data from the FastF1 API.

    Attributes:
        cache_dir: Path to FastF1 cache directory.
        output_dir: Path to save processed DataFrames.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.output_dir = output_dir or config.DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collection resilience knobs (override via config if present)
        self._retries = int(getattr(config, "COLLECT_RETRIES", 3))
        self._retry_backoff = float(getattr(config, "COLLECT_RETRY_BACKOFF_S", 3.0))
        self._collect_sleep = float(getattr(config, "COLLECT_SLEEP_S", 1.0))
        self._rate_limited = False  # set when FastF1's hourly call cap is hit

        # Enable FastF1 caching — critical for performance
        fastf1.Cache.enable_cache(str(self.cache_dir))
        logger.info(f"FastF1 cache enabled at: {self.cache_dir}")

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """True if the exception is FastF1's hourly call-budget limiter."""
        return (
            "RateLimit" in type(exc).__name__
            or "calls/h" in str(exc).lower()
            or "rate limit" in str(exc).lower()
        )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def collect_all_seasons(self, seasons: Optional[list] = None) -> pd.DataFrame:
        """
        Collect data for all configured seasons.

        Args:
            seasons: List of years to collect. Defaults to config.ALL_SEASONS.

        Returns:
            Combined DataFrame of all race data.
        """
        seasons = seasons or config.ALL_SEASONS

        for year in seasons:
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting season {year}")
            logger.info(f"{'='*60}")
            self.collect_season(year)  # writes season file + rebuilds combined
            if self._rate_limited:
                logger.warning(
                    "Stopping multi-season collect — hourly API budget exhausted. "
                    "Re-run the same command in ~1 hour to continue where it left off."
                )
                break

        combined = self._rebuild_combined_dataset()
        if combined.empty:
            logger.warning("No data collected!")
        return combined

    def collect_season(self, year: int, force: bool = False) -> pd.DataFrame:
        """
        Collect data for every race in a season.

        Resumable and incremental: rounds already present (with a finishing
        result) in season_{year}_raw.parquet are skipped, and the season file
        is re-saved after every newly collected round. So a re-run only fetches
        the gaps, and an interrupted run keeps the rounds it already got.

        Args:
            year: Season year (e.g., 2026).
            force: Re-collect every round even if already present.

        Returns:
            DataFrame with one row per driver per race.
        """
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            logger.error(f"Could not get schedule for {year}: {e}")
            return pd.DataFrame()

        # Filter to actual race events (exclude testing)
        race_events = schedule[schedule["EventFormat"] != "testing"]
        season_path = self.output_dir / f"season_{year}_raw.parquet"

        # Resume from whatever's already on disk
        season_frames = []
        have_rounds = set()
        if season_path.exists():
            existing = pd.read_parquet(season_path)
            season_frames.append(existing)
            if not force:
                done = existing
                if "IsPreRace" in done.columns:
                    done = done[~done["IsPreRace"].fillna(False)]
                have_rounds = {int(r) for r in done["RoundNumber"].dropna().unique()}

        collected = 0
        for _, event in tqdm(
            race_events.iterrows(),
            total=len(race_events),
            desc=f"Season {year}",
        ):
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]

            if round_num == 0:
                continue  # Skip pre-season testing
            if round_num in have_rounds and not force:
                continue  # already collected — skip the gap-fill re-run

            try:
                race_df = self.collect_race(year, round_num, event_name)
            except Exception as e:
                if self._is_rate_limit(e):
                    self._rate_limited = True
                    logger.warning(
                        f"Hit FastF1's hourly call limit at {year} R{round_num}. "
                        f"Progress saved ({collected} new round(s) this run). "
                        f"Re-run the same collect command in ~1 hour to resume."
                    )
                    break
                raise
            if race_df.empty:
                continue

            season_frames.append(race_df)
            collected += 1

            # Incremental save: persist progress after every round so an
            # interrupted run is never lost and a re-run resumes from here.
            season_df = self._dedupe_entries(pd.concat(season_frames, ignore_index=True))
            season_df.to_parquet(season_path, index=False)
            season_frames = [season_df]  # collapse to keep memory flat

            if self._collect_sleep > 0:
                time.sleep(self._collect_sleep)

        if not season_frames:
            return pd.DataFrame()

        season_df = self._dedupe_entries(pd.concat(season_frames, ignore_index=True))
        season_df.to_parquet(season_path, index=False)
        logger.info(
            f"Saved season {year}: {season_path} "
            f"({len(season_df)} rows, {collected} new round(s) this run)"
        )

        # Rebuild the combined file that feature engineering reads. Without this,
        # a single-season collect updates season_{year}_raw.parquet but never
        # reaches all_races_raw.parquet, so the model trains on stale data.
        self._rebuild_combined_dataset()
        return season_df

    def collect_round(self, year: int, round_number: int, event_name: str = "") -> pd.DataFrame:
        """
        Collect a single round and persist it into the season + combined files.
        Use this to backfill or refresh one race without touching the rest.
        """
        race_df = self.collect_race(year, round_number, event_name)
        if race_df.empty:
            logger.warning(f"    Nothing collected for {year} R{round_number}")
            return race_df

        season_path = self.output_dir / f"season_{year}_raw.parquet"
        frames = []
        if season_path.exists():
            frames.append(pd.read_parquet(season_path))
        frames.append(race_df)
        season_df = self._dedupe_entries(pd.concat(frames, ignore_index=True))
        season_df.to_parquet(season_path, index=False)
        logger.info(f"Saved {year} R{round_number} into {season_path}")

        self._rebuild_combined_dataset()
        return race_df

    @staticmethod
    def _dedupe_entries(df: pd.DataFrame) -> pd.DataFrame:
        """One row per (Year, Round, Driver); post-race row wins over pre-race."""
        if df.empty:
            return df
        if "IsPreRace" not in df.columns:
            df = df.copy()
            df["IsPreRace"] = False
        df["IsPreRace"] = df["IsPreRace"].fillna(False)
        return (
            df.sort_values(
                ["Year", "RoundNumber", "DriverNumber", "IsPreRace"],
                ascending=[True, True, True, False],
            )
            .drop_duplicates(subset=["Year", "RoundNumber", "DriverNumber"], keep="last")
            .reset_index(drop=True)
        )

    def _rebuild_combined_dataset(self) -> pd.DataFrame:
        """
        Merge every season_*_raw.parquet on disk into all_races_raw.parquet.

        Dedupes on (Year, RoundNumber, DriverNumber), preferring the post-race
        row over any earlier pre-race (IsPreRace) row for the same entry.
        """
        season_files = sorted(self.output_dir.glob("season_*_raw.parquet"))
        if not season_files:
            return pd.DataFrame()

        frames = [pd.read_parquet(f) for f in season_files]
        combined = self._dedupe_entries(pd.concat(frames, ignore_index=True))

        output_path = self.output_dir / "all_races_raw.parquet"
        combined.to_parquet(output_path, index=False)
        logger.info(
            f"Rebuilt combined dataset: {output_path} "
            f"({len(combined)} rows, seasons {sorted(combined['Year'].unique())})"
        )
        return combined

    def collect_race(
        self,
        year: int,
        round_number: int,
        event_name: str = "",
    ) -> pd.DataFrame:
        """
        Collect all data for a single race weekend (practice + qualifying + race).

        This is the core method that:
        1. Loads practice sessions (FP1/FP2/FP3) for long-run and short-run pace
        2. Loads qualifying session for grid positions and quali times
        3. Loads race session for results, lap data, weather, circuit info
        4. Merges everything into a single DataFrame

        Args:
            year: Season year.
            round_number: Round number in the season.
            event_name: Human-readable event name (for logging).

        Returns:
            DataFrame with one row per driver, containing all features.
        """
        logger.info(f"  Round {round_number}: {event_name}")

        # ── Load Practice Sessions ──
        practice_data = self._load_practice_sessions(year, round_number)

        # ── Load Qualifying ──
        quali_data = self._load_qualifying(year, round_number)

        # ── Load Race ──
        race_data = self._load_race(year, round_number)

        if race_data is not None:
            # Full weekend — race has happened
            race_results, race_laps, weather_summary, circuit_features = race_data

            # Merge qualifying + race data
            if quali_data is not None and not quali_data.empty:
                merged = race_results.merge(
                    quali_data, on="DriverNumber", how="left",
                    suffixes=("", "_quali"),
                )
            else:
                merged = race_results.copy()
                merged["QualifyingPosition"] = np.nan
                merged["BestQualiTime_s"] = np.nan
                merged["QualiGapToPole_s"] = np.nan

            # Add race lap statistics per driver
            lap_stats = self._compute_driver_lap_stats(race_laps)
            if not lap_stats.empty:
                merged = merged.merge(lap_stats, on="DriverNumber", how="left")

            # Add weather summary
            for col, val in weather_summary.items():
                merged[col] = val

            # Add circuit features
            for col, val in circuit_features.items():
                merged[col] = val

        elif quali_data is not None and not quali_data.empty:
            # Partial weekend — qualifying done but race hasn't happened yet
            # Build a skeleton row per driver from qualifying data
            logger.info(f"    Race not available yet — using qualifying data only (pre-race mode)")

            merged = quali_data.copy()
            merged["GridPosition"] = merged["QualifyingPosition"]

            # Try to get driver info (abbreviation, team) from the qualifying session
            try:
                q_session = fastf1.get_session(year, round_number, "Q")
                q_session.load(laps=False, telemetry=False, weather=False, messages=False)
                q_results = q_session.results
                if q_results is not None and not q_results.empty:
                    driver_info = pd.DataFrame({
                        "DriverNumber": q_results["DriverNumber"].astype(str),
                        "Abbreviation": q_results["Abbreviation"],
                        "TeamName": q_results["TeamName"],
                    }).reset_index(drop=True)
                    merged = merged.merge(driver_info, on="DriverNumber", how="left")
            except Exception:
                pass

            # Fill in missing columns that the race would normally provide
            for col in ["FinishPosition", "Points", "Status", "AvgLapTime_s",
                        "LapTimeStd_s", "LapTimeConsistency", "FastestLap_s",
                        "NumLaps", "NumPitStops", "TotalPitTime_s",
                        "FirstLapPositionsGained", "OvertakesAfterLap1",
                        "PaceToMedianRatio", "TyreDegPerLap", "NumStints"]:
                merged[col] = np.nan

            merged["IsFinished"] = False
            merged["IsMechanicalDNF"] = False
            merged["IsCrashDNF"] = False

            # Weather and circuit — try to get from practice or set NaN
            for col in ["WeatherAvgAirTemp", "WeatherAvgTrackTemp",
                        "WeatherAvgHumidity", "WeatherAvgWindSpeed"]:
                merged[col] = np.nan
            merged["WeatherHadRain"] = False
            merged["CircuitCorners"] = np.nan
            merged["CircuitStraightRatio"] = np.nan
            merged["CircuitType"] = "unknown"

            # Try to get circuit features from practice session telemetry
            try:
                fp_session = fastf1.get_session(year, round_number, "FP1")
                fp_session.load(laps=True, telemetry=True, weather=True, messages=False)
                circuit_features = self._extract_circuit_features(fp_session, fp_session.laps)
                for col, val in circuit_features.items():
                    merged[col] = val
                weather_summary = self._summarize_weather(fp_session)
                for col, val in weather_summary.items():
                    merged[col] = val
            except Exception:
                pass

            # Mark this as a pre-race entry
            merged["IsPreRace"] = True

        else:
            # No qualifying or race data at all — skip this round
            return pd.DataFrame()

        # ── Add practice data ──
        if practice_data is not None and not practice_data.empty:
            merged = merged.merge(practice_data, on="DriverNumber", how="left")

        # ── Add metadata ──
        merged["Year"] = year
        merged["RoundNumber"] = round_number
        merged["EventName"] = event_name
        merged["RegulationEra"] = REGULATION_ERAS.get(year, "unknown")

        # Mark pre-race if not already set
        if "IsPreRace" not in merged.columns:
            merged["IsPreRace"] = False

        return merged

    # ─────────────────────────────────────────
    # Session Loading
    # ─────────────────────────────────────────

    def _load_practice_sessions(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        """
        Load FP1/FP2/FP3 and extract practice pace data.

        Computes per driver:
        - Short-run pace (best single lap — quali simulation proxy)
        - Long-run pace (avg of stint of 5+ consecutive laps — race pace proxy)
        - FP2 long-run pace specifically (most representative of race conditions)
        - Practice consistency
        """
        practice_stats = {}

        for fp_name in ["FP1", "FP2", "FP3"]:
            try:
                session = fastf1.get_session(year, round_number, fp_name)
                session.load(**config.PRACTICE_LOAD_OPTIONS)
                laps = session.laps
                if laps is None or laps.empty:
                    continue
            except Exception:
                continue

            # Clean laps
            clean = clean_lap_data(laps)
            if clean.empty:
                continue

            for driver_num, driver_laps in clean.groupby("DriverNumber"):
                driver_num_str = str(driver_num)
                if driver_num_str not in practice_stats:
                    practice_stats[driver_num_str] = {}

                lap_times = driver_laps["LapTime_s"].dropna()
                if lap_times.empty:
                    continue

                # ── Short run: best single lap (quali sim) ──
                best_lap = lap_times.min()
                key_best = f"{fp_name}_BestLap_s"
                practice_stats[driver_num_str][key_best] = best_lap

                # ── Long run: average of stints >= 5 consecutive laps ──
                # Remove outliers first (pit laps already removed by clean_lap_data)
                all_driver_laps = laps[laps["DriverNumber"] == driver_num].sort_values("LapNumber")

                if "Stint" in all_driver_laps.columns:
                    long_run_times = []
                    for stint_num, stint_laps in driver_laps.groupby(
                        all_driver_laps.loc[driver_laps.index, "Stint"]
                    ):
                        stint_times = stint_laps["LapTime_s"].dropna()
                        if len(stint_times) >= 5:
                            # Drop first lap of stint (outlap effect)
                            long_run_times.extend(stint_times.iloc[1:].tolist())

                    if long_run_times:
                        key_long = f"{fp_name}_LongRunPace_s"
                        practice_stats[driver_num_str][key_long] = np.mean(long_run_times)

                        key_long_std = f"{fp_name}_LongRunStd_s"
                        practice_stats[driver_num_str][key_long_std] = np.std(long_run_times)

        if not practice_stats:
            return None

        # Build DataFrame
        rows = []
        for driver_num, stats in practice_stats.items():
            row = {"DriverNumber": driver_num}
            row.update(stats)

            # ── Composite practice features ──
            # Best short-run across all sessions (closest to quali pace)
            best_laps = [v for k, v in stats.items() if "BestLap_s" in k]
            if best_laps:
                row["Practice_BestShortRun_s"] = min(best_laps)

            # Best long-run pace (prefer FP2 as it's most representative)
            if "FP2_LongRunPace_s" in stats:
                row["Practice_BestLongRun_s"] = stats["FP2_LongRunPace_s"]
                row["Practice_LongRunConsistency"] = stats.get("FP2_LongRunStd_s", np.nan)
            else:
                long_runs = [v for k, v in stats.items() if "LongRunPace_s" in k]
                if long_runs:
                    row["Practice_BestLongRun_s"] = min(long_runs)

            # Short-run vs long-run delta (how much pace a driver loses over a stint)
            if "Practice_BestShortRun_s" in row and "Practice_BestLongRun_s" in row:
                row["Practice_ShortLongDelta_s"] = (
                    row["Practice_BestLongRun_s"] - row["Practice_BestShortRun_s"]
                )

            rows.append(row)

        result = pd.DataFrame(rows)

        # Compute gaps to field best
        if "Practice_BestShortRun_s" in result.columns:
            field_best_short = result["Practice_BestShortRun_s"].min()
            result["Practice_ShortRunGapToBest_s"] = result["Practice_BestShortRun_s"] - field_best_short

        if "Practice_BestLongRun_s" in result.columns:
            field_best_long = result["Practice_BestLongRun_s"].min()
            result["Practice_LongRunGapToBest_s"] = result["Practice_BestLongRun_s"] - field_best_long

        return result

    def _load_session_safe(self, year, round_number, session_type, load_opts, validate=None):
        """
        Load a session with retry + exponential backoff. Returns the loaded
        session or None.

        FastF1's load() often does NOT raise when the upstream API has no data
        for a session — it logs warnings and returns a half-loaded object whose
        .laps/.results then raise DataNotLoadedError on access. So an optional
        `validate(session)` runs after load and should raise if the session is
        unusable; that turns a silent partial-load into a real failure we can
        retry, and ultimately skip cleanly instead of crashing the whole run.
        """
        last_err = None
        for attempt in range(1, self._retries + 1):
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load(**load_opts)
                if validate is not None:
                    validate(session)
                return session
            except Exception as e:
                if self._is_rate_limit(e):
                    # The hourly budget won't clear in seconds — don't burn
                    # retries/backoff on it. Signal the caller to stop the run.
                    raise
                last_err = e
                logger.warning(
                    f"    {session_type} load failed "
                    f"(attempt {attempt}/{self._retries}): {e}"
                )
                self._clear_session_cache(year, round_number, session_type)
                if attempt < self._retries:
                    time.sleep(self._retry_backoff * attempt)
        logger.warning(
            f"    {session_type} unavailable for {year} R{round_number} "
            f"after {self._retries} attempts: {last_err}"
        )
        return None

    @staticmethod
    def _require_results(session):
        """Validator: session must expose non-empty results."""
        results = session.results
        if results is None or results.empty:
            raise RuntimeError("no results available")

    def _load_qualifying(self, year: int, round_number: int) -> Optional[pd.DataFrame]:
        """Load qualifying session and extract grid positions + best times."""
        session = self._load_session_safe(
            year, round_number, "Q",
            {"laps": True, "telemetry": False, "weather": False, "messages": False},
            validate=self._require_results,
        )
        if session is None:
            return None

        try:
            results = session.results
            quali_df = pd.DataFrame({
                "DriverNumber": results["DriverNumber"].astype(str),
                "QualifyingPosition": results["Position"].astype(float),
                "BestQualiTime_s": safe_timedelta_col_to_seconds(
                    results[["Q1", "Q2", "Q3"]].min(axis=1)
                ),
            }).reset_index(drop=True)  # results is DriverNumber-indexed; keep DriverNumber
            # as a column only, so the downstream merge isn't ambiguous
        except Exception as e:
            logger.warning(f"    Quali results unreadable for {year} R{round_number}: {e}")
            return None

        # Compute gap to pole
        pole_time = quali_df["BestQualiTime_s"].min()
        quali_df["QualiGapToPole_s"] = quali_df["BestQualiTime_s"] - pole_time

        return quali_df

    def _load_race(self, year: int, round_number: int):
        """
        Load race session and extract results, laps, weather, circuit info.

        Race telemetry (config.LOAD_OPTIONS) is the heaviest part of collection
        and only feeds the circuit straight-ratio feature. Set telemetry=False
        in config for much faster, more robust bulk collection if needed.

        Returns:
            Tuple of (results_df, laps_df, weather_dict, circuit_dict)
            or None if loading fails.
        """
        session = self._load_session_safe(
            year, round_number, "R", config.LOAD_OPTIONS, validate=self._require_results
        )
        if session is None:
            return None

        # ── Results ── (guaranteed present by the validator, but stay defensive)
        try:
            results = session.results
            results_df = pd.DataFrame({
                "DriverNumber": results["DriverNumber"].astype(str),
                "Abbreviation": results["Abbreviation"],
                "TeamName": results["TeamName"],
                "GridPosition": results["GridPosition"].astype(float),
                "FinishPosition": results["Position"].astype(float),
                "Points": results["Points"].astype(float),
                "Status": results["Status"],
                "IsFinished": results["Status"] == "Finished",
                "IsMechanicalDNF": results["Status"].isin(MECHANICAL_DNF_STATUSES),
                "IsCrashDNF": results["Status"].isin(CRASH_DNF_STATUSES),
            }).reset_index(drop=True)  # session.results is DriverNumber-indexed;
            # drop that so DriverNumber is only a column (avoids merge ambiguity)
        except Exception as e:
            logger.warning(f"    Race results unreadable for {year} R{round_number}: {e}")
            self._clear_session_cache(year, round_number, "R")
            return None

        # ── Laps ── optional: if timing data didn't load (common when the API
        # falls back to a mirror missing old sessions), keep the round on
        # results alone rather than dropping it. Lap-derived features become NaN.
        try:
            laps = session.laps
            if laps is None or laps.empty:
                laps = pd.DataFrame()
        except Exception as e:
            logger.warning(f"    Lap data unavailable for {year} R{round_number}: {e} "
                           f"(keeping results-only)")
            laps = pd.DataFrame()

        # ── Weather summary ──
        weather_summary = self._summarize_weather(session)

        # ── Circuit features ──
        circuit_features = self._extract_circuit_features(session, laps)

        return results_df, laps, weather_summary, circuit_features

    def _clear_session_cache(self, year: int, round_number: int, session_type: str):
        """
        Clear FastF1's cached data for a specific session that failed to load.
        This ensures the next collection attempt re-fetches from the API
        instead of serving the cached 'not available' error.
        """
        import sqlite3

        cache_db = self.cache_dir / "fastf1_http_cache.sqlite"
        if not cache_db.exists():
            return

        # Targets the legacy FastF1 'cache' table keyed by url. Best-effort:
        # if the schema differs, this no-ops rather than failing collection.
        conn = None
        try:
            conn = sqlite3.connect(str(cache_db))
            cursor = conn.cursor()
            cursor.execute("SELECT url FROM cache")
            urls = cursor.fetchall()

            session_names = {
                "R": "Race", "Q": "Qualifying",
                "FP1": "Practice%201", "FP2": "Practice%202", "FP3": "Practice%203",
                "S": "Sprint", "SQ": "Sprint%20Qualifying",
            }
            session_name = session_names.get(session_type, session_type)

            deleted = 0
            for (url,) in urls:
                if str(year) in url and session_name in url:
                    cursor.execute("DELETE FROM cache WHERE url = ?", (url,))
                    deleted += 1

            if deleted > 0:
                conn.commit()
                logger.info(f"    Cleared {deleted} cached entries for {year} R{round_number} {session_type}")
        except Exception as e:
            logger.debug(f"    Could not clear cache: {e}")
        finally:
            if conn is not None:
                conn.close()

    # ─────────────────────────────────────────
    # Feature Extraction Helpers
    # ─────────────────────────────────────────

    def _compute_driver_lap_stats(self, laps: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-driver lap statistics from race laps.

        Returns DataFrame with columns:
        - AvgLapTime_s: Mean clean lap time
        - LapTimeStd_s: Standard deviation (consistency)
        - LapTimeConsistency: Coefficient of variation (lower = more consistent)
        - FastestLap_s: Personal best in the race
        - NumLaps: Total laps completed
        - NumPitStops: Number of pit stops
        - TotalPitTime_s: Total time spent in pits
        - FirstLapPositionsGained: Positions gained on lap 1 (grid → end of lap 1)
        - OvertakesAfterLap1: Net positions gained from lap 2 onward
        - PaceToMedianRatio: Driver avg pace relative to field median (< 1.0 = faster)
        - TyreDegRadPerLap: Approximate tyre degradation (lap time increase per lap on a stint)
        - NumStints: Number of tyre stints
        """
        if laps.empty:
            return pd.DataFrame()

        # Clean the laps (remove pit laps, outliers, etc.)
        clean = clean_lap_data(laps)
        if clean.empty:
            return pd.DataFrame()

        # Compute field median lap time for relative pace
        field_median_pace = clean["LapTime_s"].median()

        stats = []
        for driver_num, driver_laps in clean.groupby("DriverNumber"):
            driver_num_str = str(driver_num)
            all_driver_laps = laps[laps["DriverNumber"] == driver_num].sort_values("LapNumber")

            # Pit stops: count laps where PitInTime is not NaT
            pit_in_laps = all_driver_laps[all_driver_laps["PitInTime"].notna()]
            num_pit_stops = len(pit_in_laps)

            # Pit time: sum of (PitOutTime - PitInTime) across stints
            total_pit_time = 0.0
            pit_out_laps = all_driver_laps[all_driver_laps["PitOutTime"].notna()]
            for _, lap in pit_out_laps.iterrows():
                if pd.notna(lap.get("PitOutTime")) and pd.notna(lap.get("PitInTime")):
                    pit_duration = (lap["PitOutTime"] - lap["PitInTime"])
                    if isinstance(pit_duration, pd.Timedelta):
                        total_pit_time += pit_duration.total_seconds()

            lap_times_s = driver_laps["LapTime_s"]

            # ── First-lap positions gained ──
            # Position column tracks race position at end of each lap
            first_lap_gained = np.nan
            overtakes_after_lap1 = np.nan
            if "Position" in all_driver_laps.columns and not all_driver_laps.empty:
                lap1 = all_driver_laps[all_driver_laps["LapNumber"] == 1]
                if not lap1.empty:
                    grid_pos = all_driver_laps.iloc[0].get("GridPosition", np.nan)
                    if pd.isna(grid_pos):
                        # Try to get grid pos from the full laps dataframe
                        grid_data = laps[laps["DriverNumber"] == driver_num]
                        if not grid_data.empty and "GridPosition" in grid_data.columns:
                            grid_pos = grid_data.iloc[0].get("GridPosition", np.nan)

                    pos_after_lap1 = lap1.iloc[0].get("Position", np.nan)
                    if pd.notna(grid_pos) and pd.notna(pos_after_lap1):
                        first_lap_gained = float(grid_pos) - float(pos_after_lap1)

                    # Net overtakes after lap 1
                    last_lap = all_driver_laps.iloc[-1]
                    final_pos = last_lap.get("Position", np.nan)
                    if pd.notna(pos_after_lap1) and pd.notna(final_pos):
                        overtakes_after_lap1 = float(pos_after_lap1) - float(final_pos)

            # ── Pace relative to field median ──
            pace_to_median = lap_times_s.mean() / field_median_pace if field_median_pace > 0 else np.nan

            # ── Tyre degradation estimate ──
            # For each stint, compute slope of lap time vs lap number
            tyre_deg = np.nan
            num_stints = 1 + num_pit_stops
            if "Stint" in all_driver_laps.columns:
                stint_slopes = []
                for stint_num, stint_laps in driver_laps.groupby(
                    all_driver_laps.loc[driver_laps.index, "Stint"]
                ):
                    if len(stint_laps) >= 4:  # Need at least 4 laps for meaningful slope
                        x = np.arange(len(stint_laps))
                        y = stint_laps["LapTime_s"].values
                        mask = ~np.isnan(y)
                        if mask.sum() >= 4:
                            slope = np.polyfit(x[mask], y[mask], 1)[0]
                            stint_slopes.append(slope)
                if stint_slopes:
                    tyre_deg = np.mean(stint_slopes)  # Avg seconds gained per lap (positive = degrading)

            stats.append({
                "DriverNumber": driver_num_str,
                "AvgLapTime_s": lap_times_s.mean(),
                "LapTimeStd_s": lap_times_s.std(),
                "LapTimeConsistency": compute_consistency(lap_times_s),
                "FastestLap_s": lap_times_s.min(),
                "NumLaps": len(all_driver_laps),
                "NumPitStops": num_pit_stops,
                "TotalPitTime_s": total_pit_time,
                "FirstLapPositionsGained": first_lap_gained,
                "OvertakesAfterLap1": overtakes_after_lap1,
                "PaceToMedianRatio": pace_to_median,
                "TyreDegPerLap": tyre_deg,
                "NumStints": num_stints,
            })

        return pd.DataFrame(stats)

    def _summarize_weather(self, session) -> dict:
        """Extract weather summary statistics for the race."""
        empty = {
            "WeatherAvgAirTemp": np.nan,
            "WeatherAvgTrackTemp": np.nan,
            "WeatherAvgHumidity": np.nan,
            "WeatherAvgWindSpeed": np.nan,
            "WeatherHadRain": False,
        }
        try:
            weather = session.weather_data
        except Exception:
            return empty
        if weather is None or weather.empty:
            return empty

        return {
            "WeatherAvgAirTemp": weather["AirTemp"].mean(),
            "WeatherAvgTrackTemp": weather["TrackTemp"].mean(),
            "WeatherAvgHumidity": weather["Humidity"].mean(),
            "WeatherAvgWindSpeed": weather["WindSpeed"].mean(),
            "WeatherHadRain": weather["Rainfall"].any() if "Rainfall" in weather else False,
        }

    def _extract_circuit_features(self, session, laps: pd.DataFrame) -> dict:
        """
        Extract circuit characteristics from session data.

        Uses circuit_info for corner count and position data for straight ratio.
        """
        features = {
            "CircuitCorners": np.nan,
            "CircuitStraightRatio": np.nan,
            "CircuitType": "unknown",
        }

        # Get circuit info
        try:
            circuit_info = session.get_circuit_info()
            if circuit_info is not None and hasattr(circuit_info, "corners"):
                features["CircuitCorners"] = len(circuit_info.corners)
        except Exception:
            pass

        # Compute straight ratio from fastest lap telemetry
        if not laps.empty:
            try:
                fastest = laps.pick_fastest()
                if fastest is not None:
                    telemetry = fastest.get_telemetry()
                    if telemetry is not None and not telemetry.empty:
                        features["CircuitStraightRatio"] = compute_straight_ratio(telemetry)
            except Exception:
                pass

        # Classify circuit type
        if not np.isnan(features["CircuitCorners"]) and not np.isnan(features["CircuitStraightRatio"]):
            features["CircuitType"] = classify_circuit(
                int(features["CircuitCorners"]),
                features["CircuitStraightRatio"],
            )

        return features

    # ─────────────────────────────────────────
    # Telemetry (for visualization, not bulk collection)
    # ─────────────────────────────────────────

    def load_session_for_visualization(
        self,
        year: int,
        race: str | int,
        session_type: str = "R",
    ):
        """
        Load a full session object for visualization purposes.
        Returns the FastF1 Session object directly.

        Args:
            year: Season year.
            race: Race name (e.g., 'Australia') or round number.
            session_type: 'R' for race, 'Q' for qualifying, etc.
        """
        session = fastf1.get_session(year, race, session_type)
        session.load(**config.LOAD_OPTIONS)
        return session

    def get_driver_telemetry(
        self,
        session,
        driver: str,
        lap: str = "fastest",
    ) -> pd.DataFrame:
        """
        Get telemetry data for a specific driver and lap.

        Args:
            session: Loaded FastF1 Session object.
            driver: Driver abbreviation (e.g., 'VER') or number.
            lap: 'fastest' for fastest lap, or a lap number.

        Returns:
            Telemetry DataFrame with Speed, RPM, Throttle, Brake, DRS, X, Y, etc.
        """
        driver_laps = session.laps.pick_drivers(driver)

        if lap == "fastest":
            target_lap = driver_laps.pick_fastest()
        else:
            target_lap = driver_laps[driver_laps["LapNumber"] == int(lap)].iloc[0]

        telemetry = target_lap.get_telemetry()
        pos_data = target_lap.get_pos_data()

        # Merge telemetry with position data
        if pos_data is not None and not pos_data.empty:
            telemetry = telemetry.merge_channels(pos_data)

        return telemetry

    def get_circuit_info(self, session):
        """Get circuit info (corners, marshal sectors, rotation angle)."""
        return session.get_circuit_info()


# ─────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect F1 data via FastF1")
    parser.add_argument("--year", type=int, help="Single season to collect")
    parser.add_argument("--round", type=int, help="Single round to collect (requires --year)")
    parser.add_argument("--all", action="store_true", help="Collect all configured seasons")
    args = parser.parse_args()

    pipeline = F1DataPipeline()

    if args.all:
        pipeline.collect_all_seasons()
    elif args.year and args.round:
        df = pipeline.collect_round(args.year, args.round, f"Round {args.round}")
        print(f"\nCollected {len(df)} driver entries")
        print(df[["Abbreviation", "GridPosition", "FinishPosition", "AvgLapTime_s"]].to_string())
    elif args.year:
        pipeline.collect_season(args.year)
    else:
        print("Collecting all configured seasons...")
        pipeline.collect_all_seasons()
