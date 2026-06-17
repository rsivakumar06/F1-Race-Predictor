"""
Walk-forward backtest harness.

Why this exists: the in-pipeline "temporal split" tests on a single season
(currently 5 races of 2026), so winner/podium accuracy there is statistical
noise. This evaluates honestly instead: walking chronologically through the
calendar, for each race it trains ONLY on races strictly before it, predicts
that race, and records the result. Aggregated over a whole window that's
hundreds of driver-rows, so MAE/ordering are stable.

It also answers two standing questions:
  1. Does the model actually beat the dumb baseline (grid order for the race
     model)? That's the honest skill test before betting odds enter.
  2. Tree vs Elo vs the dynamic blend — measured on the same races, so we can
     see whether the early-Elo weighting helps or hurts with a real sample.

No leakage: for each evaluated race the model never sees that race or any
later one. The engineered rolling features were already shifted at build time,
so per-row features for race R use only data through R-1.

Run from the project root:
    python evaluation.py                  # default: walk 2024 onward
    python evaluation.py --start-year 2023 --stride 1
    python evaluation.py --skip-quali     # race model only (faster)
"""
import sys
import io
import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from data.feature_engineering import FeatureEngineer
from models.gradient_boost import F1GradientBoostModel
from models.elo_rating import F1EloRating


# ─────────────────────────────────────────────
# Blend weights — mirrors F1EnsembleModel._get_blend_weights
# ─────────────────────────────────────────────
def blend_weights(races_in_era: int) -> tuple[float, float]:
    p = config.ENSEMBLE_PARAMS
    min_races, ramp, max_tree = (
        p["min_races_for_tree_model"], p["tree_weight_ramp_races"], p["max_tree_weight"]
    )
    if races_in_era < min_races:
        tw = 0.15
    elif races_in_era < ramp:
        progress = (races_in_era - min_races) / (ramp - min_races)
        tw = 0.15 + progress * (max_tree - 0.15)
    else:
        tw = max_tree
    return tw, 1.0 - tw


# ─────────────────────────────────────────────
# Metrics (scores are positions: lower = better)
# ─────────────────────────────────────────────
def _spearman(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 3:
        return np.nan
    ar = pd.Series(actual).rank().values
    pr = pd.Series(pred).rank().values
    if np.std(ar) == 0 or np.std(pr) == 0:
        return np.nan
    return float(np.corrcoef(ar, pr)[0, 1])


def variant_metrics(records: pd.DataFrame, score_col: str, actual_col: str) -> dict:
    """Pooled MAE/RMSE over all rows; ordering metrics averaged per race."""
    d = records.dropna(subset=[score_col, actual_col])
    if d.empty:
        return {}
    err = d[score_col].values - d[actual_col].values
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    winner_hits, podium_hits, top3_sum, spear_sum, n_races = 0, 0, 0.0, 0.0, 0
    for _, race in d.groupby(["Year", "RoundNumber"]):
        if len(race) < 2:
            continue
        actual = race[actual_col].values
        pred = race[score_col].values
        abbr = race["Abbreviation"].values

        actual_best = abbr[np.argmin(actual)]
        pred_best = abbr[np.argmin(pred)]
        winner_hits += int(actual_best == pred_best)

        k = min(3, len(race))
        actual_top = set(abbr[np.argsort(actual)[:k]])
        pred_top = set(abbr[np.argsort(pred)[:k]])
        overlap = len(actual_top & pred_top)
        top3_sum += overlap / k
        podium_hits += int(overlap > 0)

        s = _spearman(actual, pred)
        if not np.isnan(s):
            spear_sum += s
        n_races += 1

    nr = max(n_races, 1)
    return {
        "MAE": mae, "RMSE": rmse,
        "Spearman": spear_sum / nr,
        "Top3": top3_sum / nr,
        "Winner": winner_hits / nr,
        "Podium": podium_hits / nr,
        "races": n_races,
    }


def _print_table(title: str, rows: dict, variants: list):
    print(f"\n{title}")
    print(f"  {'variant':12s} {'MAE':>7s} {'RMSE':>7s} {'Spearman':>9s} "
          f"{'Top3':>7s} {'Winner':>7s} {'Podium':>7s}")
    for v in variants:
        m = rows.get(v)
        if not m:
            continue
        print(f"  {v:12s} {m['MAE']:7.3f} {m['RMSE']:7.3f} {m['Spearman']:9.3f} "
              f"{m['Top3']:7.3f} {m['Winner']*100:6.0f}% {m['Podium']*100:6.0f}%")


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────
class WalkForwardEvaluator:
    def __init__(self, features_path: Path | None = None,
                 start_year: int = 2024, stride: int = 1):
        self.engineer = FeatureEngineer()
        path = features_path or (config.DATA_DIR / "features.parquet")
        self.df = pd.read_parquet(path).sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
        self.start_year = start_year
        self.stride = stride

    def _eval_races(self) -> list:
        keys = (
            self.df[self.df["Year"] >= self.start_year][["Year", "RoundNumber"]]
            .drop_duplicates()
            .sort_values(["Year", "RoundNumber"])
            .itertuples(index=False, name=None)
        )
        keys = list(keys)
        return keys[:: self.stride]

    @staticmethod
    def _fit_tree(train_df, feature_cols, target_col, target_mode="position"):
        m = F1GradientBoostModel(target_mode=target_mode)
        m.feature_columns = feature_cols
        clean = train_df.dropna(subset=[target_col])
        if target_mode == "delta":
            clean = clean.dropna(subset=[m.grid_col])
        if clean.empty:
            return None
        X = clean[feature_cols].copy()
        m._feature_medians = X.median()
        X = X.fillna(m._feature_medians).fillna(0)
        m._fit_model(X, clean[target_col].values)
        return m

    def run_race(self) -> pd.DataFrame:
        """One row per (race, driver) with each variant's predicted position."""
        recs = []
        eval_keys = self._eval_races()
        print(f"  Race model: walking {len(eval_keys)} races "
              f"({self.start_year} onward, stride {self.stride})...")

        for i, (yr, rnd) in enumerate(eval_keys, 1):
            before = (self.df["Year"] < yr) | ((self.df["Year"] == yr) & (self.df["RoundNumber"] < rnd))
            prefix = self.df[before]
            test = self.df[(self.df["Year"] == yr) & (self.df["RoundNumber"] == rnd)]
            test = test.dropna(subset=["Target_FinishPosition"])
            if prefix.empty or test.empty:
                continue

            feature_cols = self.engineer.get_race_feature_columns(prefix)
            tree = self._fit_tree(prefix, feature_cols, "Target_RaceDelta", target_mode="delta")
            if tree is None:
                continue
            tree_pred = tree.predict(test)[["Abbreviation", "PredictedPosition"]]

            # Learning-to-rank variant (optimizes within-race order directly)
            rank_model = F1GradientBoostModel(target_mode="rank").fit_quiet(prefix, feature_cols)
            rank_pred = rank_model.predict(test)[["Abbreviation", "PredictedPosition"]].rename(
                columns={"PredictedPosition": "RankPos"}
            )

            # Elo on the prefix only (suppress its regulation-change prints)
            with contextlib.redirect_stdout(io.StringIO()):
                elo = F1EloRating()
                elo.process_historical(prefix)
                elo_pred = elo.predict_probabilities(test["Abbreviation"].tolist())
            tw, ew = blend_weights(elo._races_in_era)

            merged = (
                test[["Year", "RoundNumber", "Abbreviation", "FinishPosition", "GridPosition"]]
                .merge(tree_pred, on="Abbreviation", how="left")
                .merge(elo_pred[["Abbreviation", "ExpectedPosition"]], on="Abbreviation", how="left")
                .merge(rank_pred, on="Abbreviation", how="left")
            )
            merged["actual"] = merged["FinishPosition"]
            merged["grid"] = merged["GridPosition"]
            merged["tree"] = merged["PredictedPosition"]
            merged["elo"] = merged["ExpectedPosition"]
            merged["rank"] = merged["RankPos"]
            merged["blend"] = tw * merged["tree"] + ew * merged["elo"]
            recs.append(merged[["Year", "RoundNumber", "Abbreviation",
                                "actual", "grid", "tree", "elo", "blend", "rank"]])

            if i % 10 == 0:
                print(f"    ...{i}/{len(eval_keys)} races")

        return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

    def run_quali(self) -> pd.DataFrame:
        """Quali model walk-forward vs a driver-form baseline (no Elo here)."""
        recs = []
        eval_keys = self._eval_races()
        print(f"  Quali model: walking {len(eval_keys)} races...")

        for i, (yr, rnd) in enumerate(eval_keys, 1):
            before = (self.df["Year"] < yr) | ((self.df["Year"] == yr) & (self.df["RoundNumber"] < rnd))
            prefix = self.df[before]
            test = self.df[(self.df["Year"] == yr) & (self.df["RoundNumber"] == rnd)]
            test = test.dropna(subset=["Target_QualifyingPosition"])
            if prefix.empty or test.empty:
                continue

            feature_cols = self.engineer.get_quali_feature_columns(prefix)
            tree = self._fit_tree(prefix, feature_cols, "Target_QualifyingPosition")
            if tree is None:
                continue
            tree_pred = tree.predict(test)[["Abbreviation", "PredictedPosition"]]

            # Baseline: each driver's mean qualifying position so far
            form = prefix.groupby("Abbreviation")["QualifyingPosition"].mean()

            merged = test[["Year", "RoundNumber", "Abbreviation", "QualifyingPosition"]].merge(
                tree_pred, on="Abbreviation", how="left"
            )
            merged["actual"] = merged["QualifyingPosition"]
            merged["tree"] = merged["PredictedPosition"]
            merged["form"] = merged["Abbreviation"].map(form)
            recs.append(merged[["Year", "RoundNumber", "Abbreviation", "actual", "tree", "form"]])

            if i % 10 == 0:
                print(f"    ...{i}/{len(eval_keys)} races")

        return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def _report(records: pd.DataFrame, variants: list, label: str):
    if records.empty:
        print(f"\n{label}: no evaluated races.")
        return
    yrs = sorted(records["Year"].unique())
    n_rows = len(records)
    n_races = records.groupby(["Year", "RoundNumber"]).ngroups
    print(f"\n{'='*64}\n  {label}  ({n_races} races, {n_rows} driver-rows)\n{'='*64}")

    overall = {v: variant_metrics(records, v, "actual") for v in variants}
    _print_table("OVERALL (all evaluated races pooled):", overall, variants)

    print("\nPer season (MAE | Winner% — watch sample size in the newest year):")
    for yr in yrs:
        sub = records[records["Year"] == yr]
        nr = sub.groupby(["Year", "RoundNumber"]).ngroups
        parts = []
        for v in variants:
            m = variant_metrics(sub, v, "actual")
            if m:
                parts.append(f"{v} {m['MAE']:.2f}/{m['Winner']*100:.0f}%")
        print(f"  {yr} (n={nr:2d} races): " + "   ".join(parts))


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest")
    ap.add_argument("--start-year", type=int, default=2024,
                    help="Evaluate races from this year onward (earlier years = warmup)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Evaluate every Nth race (>1 = faster, coarser)")
    ap.add_argument("--skip-quali", action="store_true", help="Race model only")
    args = ap.parse_args()

    print("=" * 64)
    print("  WALK-FORWARD BACKTEST")
    print("=" * 64)
    ev = WalkForwardEvaluator(start_year=args.start_year, stride=args.stride)

    race_records = ev.run_race()
    _report(race_records, ["grid", "elo", "tree", "blend", "rank"],
            "RACE MODEL — predicting finishing position")
    print("\n  Read: 'grid' is the baseline (predict finish = grid order). 'tree' is "
          "\n  the delta model (production). 'rank' is the learning-to-rank variant — "
          "\n  watch whether it beats 'tree' on Winner/Top3 specifically.")

    if not args.skip_quali:
        quali_records = ev.run_quali()
        _report(quali_records, ["form", "tree"],
                "QUALI MODEL — predicting qualifying position")
        print("\n  Read: 'form' is the baseline (driver's average quali so far). "
              "\n  'tree' should beat it to justify the model.")


if __name__ == "__main__":
    main()
