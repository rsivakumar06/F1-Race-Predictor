"""
Win / podium probability model.

Replaces Elo's Monte Carlo as the source of win/podium probabilities. The
calibration backtest (calibration.py) showed a 50/50 blend of two complementary
sources beats Elo, grid, and a raw classifier on Brier + LogLoss + ECE:

  - grid prior : empirical P(win|grid slot) / P(podium|grid slot) — well CALIBRATED
  - tree       : XGBoost classifiers on Target_IsWinner / Target_IsPodium — DISCRIMINATING

Blending gives grid-level calibration with better-than-tree discrimination
(mixing in the calibrated grid removes the tree's overconfidence penalty).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

import config


class F1ProbabilityModel:
    def __init__(self, grid_weight: float | None = None, grid_col: str = "GridPosition"):
        self.grid_weight = (
            grid_weight if grid_weight is not None
            else getattr(config, "PROB_GRID_WEIGHT", 0.5)
        )
        self.grid_col = grid_col
        self.feature_columns: list = []
        self._medians = None
        self.win_clf = None
        self.pod_clf = None
        self.win_by_grid = None   # Series: grid slot -> historical win rate
        self.pod_by_grid = None
        self.global_win = 0.05
        self.global_pod = 0.15

    @staticmethod
    def _clf():
        params = config.XGBOOST_PARAMS.copy()
        params.pop("eval_metric", None)
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        return xgb.XGBClassifier(**params)

    def fit(self, df: pd.DataFrame, feature_cols: list):
        self.feature_columns = feature_cols
        clean = df.dropna(subset=["Target_IsWinner", "Target_IsPodium"]).copy()

        # Grid prior — empirical rates by starting slot
        gp = clean[clean[self.grid_col] > 0]
        self.win_by_grid = gp.groupby(self.grid_col)["Target_IsWinner"].mean()
        self.pod_by_grid = gp.groupby(self.grid_col)["Target_IsPodium"].mean()
        self.global_win = float(clean["Target_IsWinner"].mean())
        self.global_pod = float(clean["Target_IsPodium"].mean())

        # Tree classifiers
        X = clean[feature_cols].copy()
        self._medians = X.median()
        X = X.fillna(self._medians).fillna(0)
        self.win_clf = self._clf()
        self.win_clf.fit(X, clean["Target_IsWinner"].astype(int))
        self.pod_clf = self._clf()
        self.pod_clf.fit(X, clean["Target_IsPodium"].astype(int))
        return self

    @property
    def is_trained(self) -> bool:
        return self.win_clf is not None and self.pod_clf is not None

    def predict_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-driver WinProb / PodiumProb (grid+tree blend)."""
        X = df.reindex(columns=self.feature_columns).fillna(self._medians).fillna(0)
        tree_win = self.win_clf.predict_proba(X)[:, 1]
        tree_pod = self.pod_clf.predict_proba(X)[:, 1]

        grid = df[self.grid_col] if self.grid_col in df.columns else pd.Series(np.nan, index=df.index)
        grid_win = grid.map(self.win_by_grid).fillna(self.global_win).to_numpy()
        grid_pod = grid.map(self.pod_by_grid).fillna(self.global_pod).to_numpy()

        w = self.grid_weight
        out = df[["Abbreviation"]].copy()
        out["WinProb"] = w * grid_win + (1 - w) * tree_win
        out["PodiumProb"] = w * grid_pod + (1 - w) * tree_pod
        return out.reset_index(drop=True)

    def save(self, path: Path | None = None):
        path = path or (config.MODEL_DIR / "probability_model.joblib")
        joblib.dump({
            "grid_weight": self.grid_weight,
            "grid_col": self.grid_col,
            "feature_columns": self.feature_columns,
            "medians": self._medians,
            "win_clf": self.win_clf,
            "pod_clf": self.pod_clf,
            "win_by_grid": self.win_by_grid,
            "pod_by_grid": self.pod_by_grid,
            "global_win": self.global_win,
            "global_pod": self.global_pod,
        }, path)

    def load(self, path: Path | None = None):
        path = path or (config.MODEL_DIR / "probability_model.joblib")
        data = joblib.load(path)
        self.grid_weight = data["grid_weight"]
        self.grid_col = data["grid_col"]
        self.feature_columns = data["feature_columns"]
        self._medians = data["medians"]
        self.win_clf = data["win_clf"]
        self.pod_clf = data["pod_clf"]
        self.win_by_grid = data["win_by_grid"]
        self.pod_by_grid = data["pod_by_grid"]
        self.global_win = data["global_win"]
        self.global_pod = data["global_pod"]
        return self
