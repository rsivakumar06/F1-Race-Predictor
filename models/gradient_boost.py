"""
Gradient Boosted Trees model for F1 race prediction.

Uses XGBoost to predict finishing positions from engineered features.
Includes cross-validation, hyperparameter tuning, feature importance,
and proper handling of the 2026 regulation break.

Usage:
    from models.gradient_boost import F1GradientBoostModel
    model = F1GradientBoostModel()
    model.train(features_df, feature_cols)
    predictions = model.predict(new_race_features)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class F1GradientBoostModel:
    """
    XGBoost/LightGBM model for predicting F1 race finishing positions.

    The model predicts a continuous finishing position (1.0-22.0).
    Lower predicted values = better expected finish.
    Rankings are derived by sorting predictions per race.
    """

    def __init__(self, params: dict | None = None, use_lightgbm: bool = False,
                 target_mode: str = "position", grid_col: str = "GridPosition"):
        self.params = params or config.XGBOOST_PARAMS.copy()
        self.use_lightgbm = use_lightgbm and HAS_LGB
        self.model = None
        self.feature_columns = []
        self.training_metrics = {}
        self._feature_medians = None
        # target_mode: "position" predicts FinishPosition directly;
        # "delta" predicts (Finish - Grid) and reconstructs Finish = Grid + delta.
        self.target_mode = target_mode
        self.grid_col = grid_col

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str | None = None,
        eval_split: str = "temporal",
    ) -> dict:
        """
        Train the model with proper evaluation.

        Args:
            df: Features DataFrame (output of FeatureEngineer).
            feature_cols: List of feature column names to use.
            target_col: Target column. Defaults to Target_RaceDelta in delta
                mode, else Target_FinishPosition.
            eval_split: 'temporal' (train past, test latest season) or 'cv'.

        Returns:
            Dictionary of evaluation metrics (reported on finishing positions
            even when the model trains on deltas).
        """
        self.feature_columns = feature_cols

        # Ranking mode has its own fit/eval path (XGBRanker + per-race groups)
        if self.target_mode == "rank":
            return self._train_ranker(df, eval_split)

        if target_col is None:
            target_col = (
                "Target_RaceDelta" if self.target_mode == "delta"
                else "Target_FinishPosition"
            )

        # Drop rows with missing target
        clean = df.dropna(subset=[target_col]).copy()
        # Delta mode needs a valid grid to reconstruct against
        if self.target_mode == "delta":
            clean = clean.dropna(subset=[self.grid_col])

        # Fill NaN features with column medians (tree models handle this, but XGBoost wants no NaN)
        X = clean[feature_cols].copy()
        self._feature_medians = X.median()
        X = X.fillna(self._feature_medians)
        y = clean[target_col].values

        if eval_split == "temporal":
            metrics = self._train_temporal_split(X, y, clean)
        else:
            metrics = self._train_cross_validated(X, y, clean)

        # Final model: train on ALL data
        print("\n  Training final model on all data...")
        self._fit_model(X, y)

        self.training_metrics = metrics
        return metrics

    def _to_positions(self, raw: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Convert raw model output to finishing positions (grid + delta in delta mode)."""
        if self.target_mode != "delta":
            return raw
        grid = df[self.grid_col].to_numpy(dtype=float)
        pos = grid + raw
        return np.where(np.isnan(grid), raw, pos)  # fallback if grid missing

    # ─────────────────────────────────────────────
    # Ranking mode (learning-to-rank within each race)
    # ─────────────────────────────────────────────
    # Instead of regressing a position/delta, XGBRanker optimizes the ORDER of
    # drivers within a race directly. Relevance is derived from finishing
    # position (P1 = highest relevance). Predicted "position" is the within-race
    # rank of the model's score. Groups (qid) = race weekends; XGBoost requires
    # rows of the same group to be contiguous, so we sort by (Year, RoundNumber).

    def _rank_relevance(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-race relevance: max_finish - finish, so P1 gets the highest value."""
        g = frame.groupby(["Year", "RoundNumber"])["Target_FinishPosition"]
        rel = g.transform("max") - frame["Target_FinishPosition"]
        return rel.to_numpy(dtype=float)

    def _fit_ranker_frame(self, frame: pd.DataFrame):
        """Sort by race, build qid + relevance, fit XGBRanker."""
        frame = frame.dropna(subset=["Target_FinishPosition"]).sort_values(
            ["Year", "RoundNumber"]
        )
        X = frame[self.feature_columns].fillna(self._feature_medians).fillna(0)
        relevance = self._rank_relevance(frame)
        # ngroup on a frame already sorted by the group keys yields contiguous,
        # non-decreasing group ids — exactly what XGBoost's qid requires.
        qid = frame.groupby(["Year", "RoundNumber"]).ngroup().to_numpy()

        params = self.params.copy()
        params.pop("eval_metric", None)  # "mae" is invalid for ranking objectives
        params["objective"] = getattr(config, "XGBOOST_RANK_OBJECTIVE", "rank:pairwise")
        self.model = xgb.XGBRanker(**params)
        self.model.fit(X, relevance, qid=qid)

    def _ranker_positions(self, df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        """Within each race, higher score = better finish -> rank 1..N."""
        tmp = pd.DataFrame({"_score": scores})
        if {"Year", "RoundNumber"}.issubset(df.columns):
            tmp["Year"] = df["Year"].to_numpy()
            tmp["RoundNumber"] = df["RoundNumber"].to_numpy()
            return (
                tmp.groupby(["Year", "RoundNumber"])["_score"]
                .rank(ascending=False, method="first")
                .to_numpy()
            )
        return tmp["_score"].rank(ascending=False, method="first").to_numpy()

    def _train_ranker(self, df: pd.DataFrame, eval_split: str) -> dict:
        clean = df.dropna(subset=["Target_FinishPosition"]).copy()
        self._feature_medians = clean[self.feature_columns].median()

        metrics = {}
        if eval_split == "temporal":
            latest = clean["Year"].max()
            train = clean[clean["Year"] < latest]
            test = clean[clean["Year"] == latest].copy()
            if len(train) and len(test):
                print(f"  Temporal split: train on <{latest} ({len(train)} rows), "
                      f"test on {latest} ({len(test)} rows)")
                self._fit_ranker_frame(train)
                X_test = test[self.feature_columns].fillna(self._feature_medians).fillna(0)
                scores = self.model.predict(X_test)
                y_pred_pos = self._ranker_positions(test, scores)
                metrics = self._evaluate(
                    test["Target_FinishPosition"].to_numpy(), y_pred_pos, test
                )
                print(f"\n  ── Test Results ({latest} season) ──")
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}")

        print("\n  Training final model on all data...")
        self._fit_ranker_frame(clean)
        self.training_metrics = metrics
        return metrics

    def fit_quiet(self, df: pd.DataFrame, feature_cols: list):
        """Fit the final model on df with no eval/printing. Works for all modes.
        Used by the walk-forward harness, which does its own evaluation."""
        self.feature_columns = feature_cols
        if self.target_mode == "rank":
            clean = df.dropna(subset=["Target_FinishPosition"]).copy()
            self._feature_medians = clean[feature_cols].median()
            self._fit_ranker_frame(clean)
            return self
        target_col = "Target_RaceDelta" if self.target_mode == "delta" else "Target_FinishPosition"
        clean = df.dropna(subset=[target_col]).copy()
        if self.target_mode == "delta":
            clean = clean.dropna(subset=[self.grid_col])
        self._feature_medians = clean[feature_cols].median()
        X = clean[feature_cols].fillna(self._feature_medians).fillna(0)
        self._fit_model(X, clean[target_col].to_numpy())
        return self

    def _train_temporal_split(self, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame) -> dict:
        """
        Temporal split: train on all but the most recent season, test on the latest.
        This simulates real-world usage — predicting future races from past data.
        """
        latest_year = df["Year"].max()

        train_mask = df["Year"] < latest_year
        test_mask = df["Year"] == latest_year

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_test) == 0:
            print(f"  Warning: No test data for year {latest_year}, using CV instead")
            return self._train_cross_validated(X, y, df)

        print(f"  Temporal split: train on <{latest_year} ({len(X_train)} rows), "
              f"test on {latest_year} ({len(X_test)} rows)")

        self._fit_model(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Evaluate on finishing positions (reconstruct from delta if needed)
        df_test = df[test_mask]
        y_pred_pos = self._to_positions(y_pred, df_test)
        y_true_pos = self._to_positions(y_test, df_test)
        metrics = self._evaluate(y_true_pos, y_pred_pos, df_test)

        print(f"\n  ── Test Results ({latest_year} season) ──")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        return metrics

    def _train_cross_validated(self, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame) -> dict:
        """
        Grouped cross-validation where each fold is a set of race weekends.
        Groups ensure all drivers from the same race stay together.
        """
        groups = df["Year"].astype(str) + "_" + df["RoundNumber"].astype(str)

        gkf = GroupKFold(n_splits=5)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self._fit_model(X_train, y_train)
            y_pred = self.model.predict(X_test)

            df_test = df.iloc[test_idx]
            fold_mae = mean_absolute_error(
                self._to_positions(y_test, df_test),
                self._to_positions(y_pred, df_test),
            )
            fold_metrics.append(fold_mae)
            print(f"    Fold {fold+1}: MAE = {fold_mae:.3f}")

        metrics = {
            "cv_mae_mean": np.mean(fold_metrics),
            "cv_mae_std": np.std(fold_metrics),
        }
        print(f"\n  CV MAE: {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}")

        return metrics

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the underlying model."""
        if self.use_lightgbm:
            params = {
                "n_estimators": self.params.get("n_estimators", 200),
                "max_depth": self.params.get("max_depth", 4),
                "learning_rate": self.params.get("learning_rate", 0.05),
                "subsample": self.params.get("subsample", 0.8),
                "colsample_bytree": self.params.get("colsample_bytree", 0.8),
                "min_child_weight": self.params.get("min_child_weight", 3),
                "reg_alpha": self.params.get("reg_alpha", 0.1),
                "reg_lambda": self.params.get("reg_lambda", 1.0),
                "random_state": self.params.get("random_state", 42),
                "verbose": -1,
            }
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(X, y)
        else:
            params = self.params.copy()
            params.pop("eval_metric", None)
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X, y, verbose=False)

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame) -> dict:
        """
        Evaluate predictions with multiple metrics.

        Beyond MAE/RMSE, we also compute ranking-based metrics
        since we care about getting the ORDER right, not just the
        exact finishing position.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # ── Ranking accuracy per race ──
        # For each race, check if our predicted rankings match actual rankings
        correct_winner = 0
        correct_podium = 0
        total_races = 0
        top3_overlap_sum = 0

        for (year, rnd), race_group in df.groupby(["Year", "RoundNumber"]):
            idx = race_group.index
            actual_order = y_true[df.index.get_indexer(idx)]
            pred_order = y_pred[df.index.get_indexer(idx)]

            # Rank drivers by predicted position (lower = better)
            actual_ranking = np.argsort(actual_order)
            pred_ranking = np.argsort(pred_order)

            # Did we get the winner right?
            if actual_ranking[0] == pred_ranking[0]:
                correct_winner += 1

            # How many of the actual top 3 did we predict in our top 3?
            actual_top3 = set(actual_ranking[:3])
            pred_top3 = set(pred_ranking[:3])
            overlap = len(actual_top3 & pred_top3)
            top3_overlap_sum += overlap / 3.0

            # Did we get at least 1 podium right?
            if overlap > 0:
                correct_podium += 1

            total_races += 1

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "winner_accuracy": correct_winner / max(total_races, 1),
            "podium_accuracy": correct_podium / max(total_races, 1),
            "top3_overlap": top3_overlap_sum / max(total_races, 1),
            "total_test_races": total_races,
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict finishing positions for a set of drivers.

        Args:
            df: DataFrame with feature columns (one row per driver).

        Returns:
            DataFrame with columns: Abbreviation, PredictedPosition, PredictedRank
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Reindex to the exact training schema. One-hot columns (circuit type,
        # grid bucket) can be absent for a single race; reindex adds them back
        # as NaN so column order matches the fitted model, then medians fill in.
        X = df.reindex(columns=self.feature_columns)
        X = X.fillna(self._feature_medians).fillna(0)
        raw_predictions = self.model.predict(X)

        result = df[["Abbreviation", "TeamName"]].copy()
        if self.target_mode == "rank":
            # Score -> within-race rank (1 = predicted winner)
            result["PredictedPosition"] = self._ranker_positions(df, raw_predictions)
        else:
            result["PredictedPosition"] = self._to_positions(raw_predictions, df)
        result["PredictedRank"] = result["PredictedPosition"].rank(method="min").astype(int)
        result = result.sort_values("PredictedRank")

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top N most important features."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": self.feature_columns,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

        return importance_df.head(top_n)

    def save(self, path: Path | None = None):
        """Save trained model and metadata."""
        path = path or config.MODEL_DIR / "xgboost_model.joblib"
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "feature_medians": self._feature_medians,
            "params": self.params,
            "metrics": self.training_metrics,
            "target_mode": self.target_mode,
            "grid_col": self.grid_col,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path: Path | None = None):
        """Load trained model and metadata."""
        path = path or config.MODEL_DIR / "xgboost_model.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self._feature_medians = data["feature_medians"]
        self.params = data["params"]
        self.training_metrics = data["metrics"]
        self.target_mode = data.get("target_mode", "position")
        self.grid_col = data.get("grid_col", "GridPosition")
        print(f"Model loaded: {path}")


if __name__ == "__main__":
    from data.feature_engineering import FeatureEngineer

    # Load features
    engineer = FeatureEngineer()
    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    feature_cols = engineer.get_feature_columns(df)

    print(f"Training on {len(df)} rows with {len(feature_cols)} features")

    # Train and evaluate
    model = F1GradientBoostModel()
    metrics = model.train(df, feature_cols, eval_split="temporal")

    # Feature importance
    print(f"\n{'='*60}")
    print("TOP 20 FEATURES")
    print(f"{'='*60}")
    importance = model.get_feature_importance(20)
    for _, row in importance.iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"  {row['Feature']:45s} {bar} {row['Importance']:.4f}")

    # Save
    model.save()
