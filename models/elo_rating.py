"""
Elo Rating System for F1 Drivers.

A Bayesian rating system that:
- Maintains per-driver ratings updated after each race
- Partially resets ratings on regulation era changes (2026)
- Converts ratings to win/podium probability estimates
- Adapts rapidly — K-factor is higher early in a new era

The key insight: Elo captures *current form* without needing features.
It complements the tree model which captures structural patterns.

Usage:
    from models.elo_rating import F1EloRating
    elo = F1EloRating()
    elo.process_historical(features_df)
    probabilities = elo.predict_probabilities(["VER", "NOR", "LEC", ...])
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.constants import REGULATION_ERAS


class F1EloRating:
    """
    Elo-based rating system for F1 drivers.

    Each driver has a rating (default 1500). After each race, ratings
    are updated based on head-to-head results against every other driver
    in the race. Beating a higher-rated driver gains more points.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or config.ELO_PARAMS.copy()
        self.ratings = {}           # driver abbreviation -> current rating
        self.rating_history = []    # list of (year, round, driver, rating) tuples
        self._current_era = None
        self._races_in_era = 0
        # Seeded RNG so win/podium probabilities are reproducible run-to-run
        self._rng = np.random.default_rng(getattr(config, "RANDOM_SEED", 42))

    @property
    def initial_rating(self):
        return self.params["initial_rating"]

    @property
    def base_k(self):
        return self.params["k_factor"]

    @property
    def decay_factor(self):
        return self.params["decay_factor"]

    @property
    def reset_factor(self):
        return self.params["regulation_reset_factor"]

    def get_rating(self, driver: str) -> float:
        """Get current rating for a driver (initializes if new)."""
        if driver not in self.ratings:
            self.ratings[driver] = self.initial_rating
        return self.ratings[driver]

    def get_k_factor(self) -> float:
        """
        Dynamic K-factor that's higher early in a new regulation era.
        This lets ratings adapt quickly when the pecking order is uncertain.

        First 3 races of new era:  K * 2.0 (very fast adaptation)
        Races 4-8:                 K * 1.5 (moderately fast)
        Races 9+:                  K * 1.0 (standard)
        """
        if self._races_in_era <= 3:
            return self.base_k * 2.0
        elif self._races_in_era <= 8:
            return self.base_k * 1.5
        else:
            return self.base_k

    def process_historical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all historical races and build up driver ratings.

        Args:
            df: Features DataFrame with Year, RoundNumber, Abbreviation,
                FinishPosition columns.

        Returns:
            DataFrame with Elo ratings for each driver at each race
            (BEFORE the race — i.e., the prediction-ready rating).
        """
        df = df.sort_values(["Year", "RoundNumber"]).copy()

        pre_race_ratings = []

        for (year, rnd), race_group in df.groupby(["Year", "RoundNumber"], sort=True):
            # Check for regulation era change
            era = REGULATION_ERAS.get(year, "unknown")
            if era != self._current_era:
                if self._current_era is not None:
                    self._apply_era_reset()
                    print(f"  Elo: Regulation change detected at {year}. "
                          f"Ratings partially reset (factor={self.reset_factor})")
                self._current_era = era
                self._races_in_era = 0

            self._races_in_era += 1

            # Record PRE-RACE ratings (these are what we'd use for prediction)
            for _, row in race_group.iterrows():
                driver = row["Abbreviation"]
                pre_race_ratings.append({
                    "Year": year,
                    "RoundNumber": rnd,
                    "Abbreviation": driver,
                    "EloRating": self.get_rating(driver),
                })

            # Update ratings based on race result
            results = (
                race_group[["Abbreviation", "FinishPosition"]]
                .dropna(subset=["FinishPosition"])
                .sort_values("FinishPosition")
            )
            self._update_after_race(results)

            # Store history
            for driver, rating in self.ratings.items():
                self.rating_history.append((year, rnd, driver, rating))

        return pd.DataFrame(pre_race_ratings)

    def _update_after_race(self, results: pd.DataFrame):
        """
        Update Elo ratings using all pairwise comparisons in a race.

        For N drivers, there are N*(N-1)/2 pairwise matchups.
        A driver who finishes P1 "beat" all other drivers.
        A driver who finishes P10 "beat" everyone below and "lost" to everyone above.
        """
        drivers = results["Abbreviation"].tolist()
        positions = results["FinishPosition"].tolist()
        n = len(drivers)
        k = self.get_k_factor()

        # Scale K by number of comparisons so ratings don't inflate
        # Each driver is in (n-1) matchups, so we divide K accordingly
        k_scaled = k / max(n - 1, 1)

        rating_changes = {d: 0.0 for d in drivers}

        for i in range(n):
            for j in range(i + 1, n):
                driver_a = drivers[i]
                driver_b = drivers[j]
                rating_a = self.get_rating(driver_a)
                rating_b = self.get_rating(driver_b)

                # Expected score (probability A beats B)
                expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
                expected_b = 1.0 - expected_a

                # Actual score: whoever finished higher wins the matchup
                # positions[i] < positions[j] since results are sorted
                actual_a = 1.0  # driver_a finished ahead
                actual_b = 0.0

                # Accumulate rating changes
                rating_changes[driver_a] += k_scaled * (actual_a - expected_a)
                rating_changes[driver_b] += k_scaled * (actual_b - expected_b)

        # Apply changes
        for driver, change in rating_changes.items():
            self.ratings[driver] = self.get_rating(driver) + change

    def _apply_era_reset(self):
        """
        Partially reset ratings toward the mean when regulations change.
        This reflects the uncertainty — a team that was dominant might
        not be under new rules. Factor of 0.5 means ratings move
        halfway back to the mean.
        """
        if not self.ratings:
            return

        mean_rating = np.mean(list(self.ratings.values()))

        for driver in self.ratings:
            current = self.ratings[driver]
            self.ratings[driver] = current + self.reset_factor * (mean_rating - current)

    def predict_probabilities(self, drivers: list[str]) -> pd.DataFrame:
        """
        Predict win/podium probabilities and expected finishing position.

        Monte Carlo: each simulated race adds Gaussian noise (sigma from
        mc_score_noise_sigma) to every driver's rating, then ranks by the
        noisy score. Aggregating over many sims gives win%, podium%, and a
        mean position. Higher rating -> better expected finish.

        Args:
            drivers: Driver abbreviations expected in the race.

        Returns:
            DataFrame: Abbreviation, EloRating, WinProb, PodiumProb, ExpectedPosition
        """
        n = len(drivers)
        if n == 0:
            return pd.DataFrame(
                columns=["Abbreviation", "EloRating", "WinProb",
                         "PodiumProb", "ExpectedPosition"]
            )

        ratings_arr = np.array([self.get_rating(d) for d in drivers], dtype=float)
        n_sims = int(self.params.get("mc_simulations", 10000))
        sigma = float(self.params.get("mc_score_noise_sigma", 200.0))

        # Vectorized: noise (n_sims, n), rank each sim, tally in bulk.
        noise = self._rng.normal(0.0, sigma, size=(n_sims, n))
        scores = ratings_arr[None, :] + noise
        order = np.argsort(-scores, axis=1)           # driver indices by finish

        # finishing position (1..n) of each driver in each sim
        positions = np.empty_like(order)
        ranks = np.broadcast_to(np.arange(1, n + 1), (n_sims, n))
        np.put_along_axis(positions, order, ranks, axis=1)

        win_counts = np.bincount(order[:, 0], minlength=n)
        podium_counts = np.bincount(order[:, :3].ravel(), minlength=n)
        position_sums = positions.sum(axis=0)

        result = pd.DataFrame({
            "Abbreviation": drivers,
            "EloRating": ratings_arr,
            "WinProb": win_counts / n_sims,
            "PodiumProb": podium_counts / n_sims,
            "ExpectedPosition": position_sums / n_sims,
        }).sort_values("ExpectedPosition").reset_index(drop=True)

        return result

    def get_current_rankings(self) -> pd.DataFrame:
        """Return all drivers sorted by current Elo rating."""
        if not self.ratings:
            return pd.DataFrame()

        return pd.DataFrame([
            {"Abbreviation": d, "EloRating": r}
            for d, r in self.ratings.items()
        ]).sort_values("EloRating", ascending=False).reset_index(drop=True)

    def save(self, path: Path | None = None):
        """Save Elo state."""
        path = path or config.MODEL_DIR / "elo_ratings.joblib"
        joblib.dump({
            "ratings": self.ratings,
            "rating_history": self.rating_history,
            "params": self.params,
            "current_era": self._current_era,
            "races_in_era": self._races_in_era,
        }, path)
        print(f"Elo ratings saved: {path}")

    def load(self, path: Path | None = None):
        """Load Elo state."""
        path = path or config.MODEL_DIR / "elo_ratings.joblib"
        data = joblib.load(path)
        self.ratings = data["ratings"]
        self.rating_history = data["rating_history"]
        self.params = data["params"]
        self._current_era = data["current_era"]
        self._races_in_era = data["races_in_era"]
        print(f"Elo ratings loaded: {path}")


if __name__ == "__main__":
    # Process all historical data and show current rankings
    df = pd.read_parquet(config.DATA_DIR / "features.parquet")

    elo = F1EloRating()
    pre_race = elo.process_historical(df)

    print(f"\n{'='*60}")
    print("CURRENT ELO RANKINGS")
    print(f"{'='*60}")
    rankings = elo.get_current_rankings()
    for i, row in rankings.iterrows():
        bar = "█" * int((row["EloRating"] - 1300) / 5)
        print(f"  {i+1:2d}. {row['Abbreviation']:4s}  {row['EloRating']:7.1f}  {bar}")

    # Show predictions for next race
    active_drivers = df[df["Year"] == df["Year"].max()]["Abbreviation"].unique().tolist()
    print(f"\n{'='*60}")
    print("WIN PROBABILITIES (next race)")
    print(f"{'='*60}")
    probs = elo.predict_probabilities(active_drivers)
    for _, row in probs.head(10).iterrows():
        bar = "█" * int(row["WinProb"] * 100)
        print(f"  {row['Abbreviation']:4s}  {row['WinProb']*100:5.1f}%  {bar}")

    elo.save()
