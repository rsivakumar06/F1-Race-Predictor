"""
Probability-calibration backtest.

The positional harness (evaluation.py) scores ORDER. This one scores the
win/podium PROBABILITIES the system outputs — a different and so-far unmeasured
thing. Production currently derives those from Elo's Monte Carlo, which we've
shown is the weakest positional signal, so they're probably miscalibrated.

Walk-forward (train only on prior races), it produces per-driver win and podium
probabilities from three sources and scores them:

  - grid   : empirical P(win|grid slot) / P(podium|grid slot) learned from history
  - elo    : Elo Monte Carlo (current production source)
  - tree   : XGBoost classifiers on Target_IsWinner / Target_IsPodium

Metrics (lower is better for all three):
  - Brier    : mean (prob - outcome)^2  — accuracy + calibration in one number
  - LogLoss  : punishes confident-and-wrong harder
  - ECE      : Expected Calibration Error — when it says 30%, does 30% happen?

Run from the project root:
    python calibration.py [--start-year 2024] [--stride 1]
"""
import sys
import io
import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Weight on the (well-calibrated) grid prior when blending it with the
# (discriminating) tree. blend = BLEND_GRID_WEIGHT*grid + (1-...)*tree.
BLEND_GRID_WEIGHT = 0.5

sys.path.insert(0, str(Path(__file__).parent))
import config
from data.feature_engineering import FeatureEngineer
from models.elo_rating import F1EloRating


# ─────────────────────────────────────────────
# Calibration metrics (binary outcome y in {0,1}, probability p in [0,1])
# ─────────────────────────────────────────────
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    """Expected Calibration Error: weighted gap between predicted and actual per bin."""
    edges = np.linspace(0, 1, bins + 1)
    n = len(p)
    err = 0.0
    for i in range(bins):
        if i < bins - 1:
            m = (p >= edges[i]) & (p < edges[i + 1])
        else:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        err += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(err)


def score_source(df, prob_col, outcome_col):
    d = df.dropna(subset=[prob_col, outcome_col])
    p = d[prob_col].to_numpy(dtype=float)
    y = d[outcome_col].to_numpy(dtype=float)
    return {
        "Brier": brier(p, y),
        "LogLoss": logloss(p, y),
        "ECE": ece(p, y),
        "MeanProb": float(p.mean()),
        "BaseRate": float(y.mean()),
    }


def _clf():
    params = config.XGBOOST_PARAMS.copy()
    params.pop("eval_metric", None)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    return xgb.XGBClassifier(**params)


class CalibrationEvaluator:
    def __init__(self, start_year=2024, stride=1):
        self.engineer = FeatureEngineer()
        self.df = (
            pd.read_parquet(config.DATA_DIR / "features.parquet")
            .sort_values(["Year", "RoundNumber"])
            .reset_index(drop=True)
        )
        self.start_year = start_year
        self.stride = stride

    def _eval_races(self):
        keys = list(
            self.df[self.df["Year"] >= self.start_year][["Year", "RoundNumber"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        return keys[:: self.stride]

    def run(self):
        recs = []
        keys = self._eval_races()
        print(f"  Walking {len(keys)} races ({self.start_year} onward)...")

        for i, (yr, rnd) in enumerate(keys, 1):
            before = (self.df["Year"] < yr) | (
                (self.df["Year"] == yr) & (self.df["RoundNumber"] < rnd)
            )
            prefix = self.df[before]
            test = self.df[(self.df["Year"] == yr) & (self.df["RoundNumber"] == rnd)].copy()
            test = test.dropna(subset=["Target_IsWinner", "Target_IsPodium"])
            if prefix.empty or test.empty:
                continue

            # actual outcomes
            test["win"] = test["Target_IsWinner"].astype(float)
            test["pod"] = test["Target_IsPodium"].astype(float)

            # ── Source 1: grid prior (empirical rate by grid slot) ──
            gp = prefix[prefix["GridPosition"] > 0]
            win_by_grid = gp.groupby("GridPosition")["Target_IsWinner"].mean()
            pod_by_grid = gp.groupby("GridPosition")["Target_IsPodium"].mean()
            gw, gp_ = gp["Target_IsWinner"].mean(), gp["Target_IsPodium"].mean()
            test["grid_win"] = test["GridPosition"].map(win_by_grid).fillna(gw)
            test["grid_pod"] = test["GridPosition"].map(pod_by_grid).fillna(gp_)

            # ── Source 2: Elo Monte Carlo ──
            with contextlib.redirect_stdout(io.StringIO()):
                elo = F1EloRating()
                elo.process_historical(prefix)
                ep = elo.predict_probabilities(test["Abbreviation"].tolist())
            ep = ep.set_index("Abbreviation")
            test["elo_win"] = test["Abbreviation"].map(ep["WinProb"])
            test["elo_pod"] = test["Abbreviation"].map(ep["PodiumProb"])

            # ── Source 3: XGBoost classifiers on win / podium ──
            feats = self.engineer.get_race_feature_columns(prefix)
            med = prefix[feats].median()
            Xtr = prefix[feats].fillna(med).fillna(0)
            Xte = test[feats].fillna(med).fillna(0)
            cw, cp = _clf(), _clf()
            cw.fit(Xtr, prefix["Target_IsWinner"].astype(int))
            cp.fit(Xtr, prefix["Target_IsPodium"].astype(int))
            test["tree_win"] = cw.predict_proba(Xte)[:, 1]
            test["tree_pod"] = cp.predict_proba(Xte)[:, 1]

            # ── Source 4: grid+tree blend (calibration from grid, edge from tree) ──
            w = BLEND_GRID_WEIGHT
            test["blend_win"] = w * test["grid_win"] + (1 - w) * test["tree_win"]
            test["blend_pod"] = w * test["grid_pod"] + (1 - w) * test["tree_pod"]

            recs.append(test[[
                "Year", "RoundNumber", "Abbreviation", "win", "pod",
                "grid_win", "grid_pod", "elo_win", "elo_pod",
                "tree_win", "tree_pod", "blend_win", "blend_pod",
            ]])
            if i % 10 == 0:
                print(f"    ...{i}/{len(keys)} races")

        return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def _print_table(title, records, sources, kind):
    print(f"\n{title}")
    print(f"  {'source':8s} {'Brier':>8s} {'LogLoss':>8s} {'ECE':>7s} "
          f"{'MeanProb':>9s} {'BaseRate':>9s}")
    for name, col in sources:
        m = score_source(records, col, kind)
        print(f"  {name:8s} {m['Brier']:8.4f} {m['LogLoss']:8.4f} {m['ECE']:7.4f} "
              f"{m['MeanProb']:9.4f} {m['BaseRate']:9.4f}")


def main():
    ap = argparse.ArgumentParser(description="Probability calibration backtest")
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    print("=" * 64)
    print("  PROBABILITY CALIBRATION BACKTEST")
    print("=" * 64)
    ev = CalibrationEvaluator(args.start_year, args.stride)
    rec = ev.run()
    if rec.empty:
        print("No evaluated races.")
        return

    n_races = rec.groupby(["Year", "RoundNumber"]).ngroup().nunique()
    print(f"\n  {n_races} races, {len(rec)} driver-rows\n{'='*64}")

    _print_table("WIN probability:", rec,
                 [("grid", "grid_win"), ("elo", "elo_win"),
                  ("tree", "tree_win"), ("blend", "blend_win")], "win")
    _print_table("PODIUM probability:", rec,
                 [("grid", "grid_pod"), ("elo", "elo_pod"),
                  ("tree", "tree_pod"), ("blend", "blend_pod")], "pod")

    print(f"\n  Read: lower Brier/LogLoss/ECE = better. 'elo' is current production. "
          f"\n  'blend' = {BLEND_GRID_WEIGHT:.0%} grid + {1-BLEND_GRID_WEIGHT:.0%} tree — "
          f"\n  hoping for grid-level calibration with tree-level discrimination. "
          f"\n  'MeanProb' far from 'BaseRate' signals systematic bias.")


if __name__ == "__main__":
    main()
