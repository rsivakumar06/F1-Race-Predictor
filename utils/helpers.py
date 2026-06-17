"""
Utility functions for data cleaning, time conversion, and common operations.
"""
import pandas as pd
import numpy as np
from typing import Optional


def timedelta_to_seconds(td: pd.Timedelta) -> Optional[float]:
    """Convert a pandas Timedelta to float seconds. Returns None for NaT."""
    if pd.isna(td):
        return None
    return td.total_seconds()


def safe_timedelta_col_to_seconds(series: pd.Series) -> pd.Series:
    """Convert an entire Series of Timedeltas to float seconds."""
    return series.apply(
        lambda x: x.total_seconds() if pd.notna(x) and isinstance(x, pd.Timedelta) else np.nan
    )


def classify_circuit(corners: int, straight_ratio: float) -> str:
    """
    Classify a circuit type based on its characteristics.

    Args:
        corners: Number of corners on the circuit.
        straight_ratio: Proportion of lap distance that is straights (0-1).

    Returns:
        One of: 'street', 'high_speed', 'technical', 'balanced'
    """
    if corners >= 18:
        return "street"
    elif straight_ratio > 0.45:
        return "high_speed"
    elif corners >= 14:
        return "technical"
    else:
        return "balanced"


def compute_straight_ratio(pos_data: pd.DataFrame, speed_threshold: float = 250.0) -> float:
    """
    Estimate the proportion of a lap spent on straights using telemetry.

    Args:
        pos_data: Telemetry DataFrame with 'Speed' column.
        speed_threshold: Speed (km/h) above which we consider the car on a straight.

    Returns:
        Float ratio (0-1) of samples above the speed threshold.
    """
    if pos_data is None or pos_data.empty or "Speed" not in pos_data.columns:
        return 0.0
    return (pos_data["Speed"] > speed_threshold).mean()


def clean_lap_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Clean lap data by removing outliers and invalid entries.

    Removes:
    - In-laps and out-laps (pit entry/exit laps with artificially slow times)
    - Laps under safety car or red flag
    - Laps with no valid lap time
    - Statistical outliers (>1.5x the median lap time)
    """
    if laps is None or laps.empty:
        return pd.DataFrame()

    cleaned = laps.copy()

    # Remove pit in/out laps
    cleaned = cleaned[
        cleaned["PitInTime"].isna() & cleaned["PitOutTime"].isna()
    ]

    # Remove laps without valid lap time
    cleaned = cleaned.dropna(subset=["LapTime"])

    # Convert lap time to seconds for filtering
    cleaned["LapTime_s"] = safe_timedelta_col_to_seconds(cleaned["LapTime"])

    # Remove extreme outliers (>1.5x median — likely SC laps or slow zones)
    if not cleaned.empty:
        median_time = cleaned["LapTime_s"].median()
        cleaned = cleaned[cleaned["LapTime_s"] < median_time * 1.5]

    return cleaned


def normalize_driver_identifier(driver: str) -> str:
    """Normalize driver identifiers (handle abbreviations vs full names)."""
    return driver.strip().upper()


def get_season_progress(round_number: int, total_rounds: int) -> float:
    """Return season progress as a float 0-1."""
    return round_number / max(total_rounds, 1)


def compute_consistency(lap_times_seconds: pd.Series) -> float:
    """
    Compute driver consistency as coefficient of variation of lap times.
    Lower = more consistent.
    """
    if lap_times_seconds.empty or lap_times_seconds.std() == 0:
        return 0.0
    return lap_times_seconds.std() / lap_times_seconds.mean()
