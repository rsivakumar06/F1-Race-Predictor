"""
Ensemble Model — Blends XGBoost predictions with Elo ratings.

The weighting shifts dynamically based on how far into a new regulation
era we are. Early in 2026, Elo gets more weight (it adapts faster).
As more races happen, the tree model gets more weight (it has more
structural features to learn from).

Usage:
    from models.ensemble import F1Ensemble
    ensemble = F1Ensemble()
    ensemble.train(features_df, feature_cols)
    predictions = ensemble.predict(race_features_df, race_drivers)
"""
import sys
import io
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.gradient_boost import F1GradientBoostModel
from models.elo_rating import F1EloRating
from models.probability_model import F1ProbabilityModel
from data.feature_engineering import FeatureEngineer


class F1Ensemble:
    """
    Blends XGBoost finishing position predictions with Elo expected positions.

    The blend weight shifts dynamically:
    - Early in a new era (< 5 races):  Elo dominates (70/30)
    - Mid era (5-15 races):            Gradual shift toward XGBoost
    - Mature era (15+ races):          XGBoost dominates (60/40)
    """

    def __init__(self, params: dict | None = None):
        self.params = params or config.ENSEMBLE_PARAMS.copy()
        # Race model predicts deviation from grid (Finish - Grid) and reconstructs
        # finishing position as Grid + delta. The walk-forward backtest showed this
        # framing is the only way the model can add value over the grid baseline.
        self.tree_model = F1GradientBoostModel(target_mode="delta")
        self.quali_model = None
        self.has_quali_model = False
        self.elo_model = F1EloRating()
        # Win/podium probabilities — grid+tree blend, replacing Elo's Monte Carlo
        # (the calibration backtest showed Elo was the worst-calibrated source).
        self.prob_model = F1ProbabilityModel()
        self.is_trained = False

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "Target_FinishPosition",
        engineer=None,
    ) -> dict:
        """
        Train both component models.

        Args:
            df: Features DataFrame.
            feature_cols: Feature columns for the tree model.
            target_col: Target variable.

        Returns:
            Combined metrics dictionary.
        """
        print("=" * 60)
        print("  TRAINING ENSEMBLE MODEL (2-STAGE)")
        print("=" * 60)

        # ── Stage 1: Qualifying Predictor ──
        print("\n── Stage 1: Qualifying Prediction Model ──")
        quali_feature_cols = engineer.get_quali_feature_columns(df) if engineer else feature_cols
        self.quali_model = F1GradientBoostModel()

        # Only train on rows that have qualifying data
        quali_df = df.dropna(subset=["QualifyingPosition"])
        if len(quali_df) > 20:
            quali_metrics = self.quali_model.train(
                quali_df, quali_feature_cols,
                target_col="Target_QualifyingPosition",
                eval_split="temporal",
            )
            self.has_quali_model = True
            print(f"  Quali model trained on {len(quali_df)} rows, {len(quali_feature_cols)} features")
        else:
            quali_metrics = {}
            self.has_quali_model = False
            print("  Not enough quali data to train quali model")

        # ── Stage 2: Race Predictor (uses quali position as a feature) ──
        print("\n── Stage 2: Race Prediction Model ──")
        race_feature_cols = engineer.get_race_feature_columns(df) if engineer else feature_cols
        # Delta target — the model learns Finish - Grid, predict() returns Finish.
        tree_metrics = self.tree_model.train(
            df, race_feature_cols, target_col="Target_RaceDelta", eval_split="temporal"
        )

        # ── Process Elo ratings ──
        print("\n── Elo Ratings ──")
        elo_ratings_df = self.elo_model.process_historical(df)

        # ── Win/podium probability model (grid+tree blend) ──
        print("\n── Win/Podium Probability Model ──")
        self.prob_model.fit(df, race_feature_cols)
        print(f"  Probability model: {self.prob_model.grid_weight:.0%} grid prior + "
              f"{1 - self.prob_model.grid_weight:.0%} tree classifiers")

        # ── Evaluate ensemble on the test set (latest season) ──
        latest_year = df["Year"].max()
        test_df = df[df["Year"] == latest_year].copy()

        if len(test_df) > 0:
            print(f"\n── Ensemble Evaluation ({latest_year}, held-out) ──")
            ensemble_metrics = self._evaluate_ensemble(df, race_feature_cols, latest_year)
        else:
            ensemble_metrics = {}

        self.is_trained = True

        # ── Combined metrics ──
        all_metrics = {
            "quali": quali_metrics,
            "tree": tree_metrics,
            "ensemble": ensemble_metrics,
        }

        # Print summary
        print(f"\n{'='*60}")
        print("  ENSEMBLE SUMMARY")
        print(f"{'='*60}")

        elo_rankings = self.elo_model.get_current_rankings()
        print(f"\n  Current Elo Top 10:")
        for i, row in elo_rankings.head(10).iterrows():
            print(f"    {i+1:2d}. {row['Abbreviation']:4s}  (Elo: {row['EloRating']:.0f})")

        print(f"\n  Race XGBoost MAE: {tree_metrics.get('mae', tree_metrics.get('cv_mae_mean', 'N/A'))}")
        if quali_metrics:
            print(f"  Quali XGBoost MAE: {quali_metrics.get('mae', quali_metrics.get('cv_mae_mean', 'N/A'))}")
        if ensemble_metrics:
            print(f"  Ensemble MAE: {ensemble_metrics.get('ensemble_mae', 'N/A'):.3f}")
            print(f"  Winner accuracy: {ensemble_metrics.get('winner_accuracy', 0)*100:.0f}%")
            print(f"  Podium accuracy: {ensemble_metrics.get('podium_accuracy', 0)*100:.0f}%")

        # Feature importance from tree model
        print(f"\n  Top 10 Features:")
        importance = self.tree_model.get_feature_importance(10)
        for _, row in importance.iterrows():
            print(f"    {row['Feature']:42s}  {row['Importance']:.4f}")

        return all_metrics

    def predict_quali(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Stage 1: Predict qualifying grid BEFORE qualifying happens.
        Uses practice data + historical form only.

        Returns:
            DataFrame with predicted qualifying positions.
        """
        if not self.has_quali_model:
            # Fallback: use Elo expected positions as quali prediction
            drivers = df["Abbreviation"].tolist()
            elo_preds = self.elo_model.predict_probabilities(drivers)
            result = df[["Abbreviation", "TeamName"]].copy()
            result = result.merge(
                elo_preds[["Abbreviation", "ExpectedPosition"]],
                on="Abbreviation", how="left",
            )
            result["PredictedQualiPosition"] = result["ExpectedPosition"]
            result["PredictedQualiRank"] = result["PredictedQualiPosition"].rank(method="min").astype(int)
            result = result.sort_values("PredictedQualiRank").reset_index(drop=True)
            result["Source"] = "Elo (no quali model)"
            return result

        if verbose:
            # Show what features the quali model is using
            print(f"\n  Quali model uses {len(self.quali_model.feature_columns)} features:")
            # Check for any suspicious feature names
            suspicious = [f for f in self.quali_model.feature_columns
                         if any(s in f.lower() for s in ["qualifyingposition", "bestqualitime",
                                "qualigaptopole_s", "gridposition"])]
            if suspicious:
                print(f"  WARNING: Potentially leaked features: {suspicious}")
            else:
                print(f"  No qualifying/grid leakage detected")

            # Show top practice features
            practice_feats = [f for f in self.quali_model.feature_columns if "Practice" in f or "practice" in f]
            if practice_feats:
                print(f"  Practice features ({len(practice_feats)}): {practice_feats[:5]}...")

        # Use the trained quali model
        quali_preds = self.quali_model.predict(df)
        result = quali_preds.rename(columns={
            "PredictedPosition": "PredictedQualiPosition",
            "PredictedRank": "PredictedQualiRank",
        })
        result["Source"] = "XGBoost quali model"
        return result

    def predict_race(
        self,
        df: pd.DataFrame,
        actual_quali: pd.DataFrame | None = None,
        drivers: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Stage 2: Predict race finishing positions.

        If actual_quali is provided (post-qualifying), it uses the real grid
        positions to adjust the prediction. Otherwise uses predicted quali.

        Args:
            df: Feature DataFrame for the race (one row per driver).
            actual_quali: DataFrame with 'Abbreviation' and 'QualifyingPosition'.
                          If None, uses whatever grid data is already in df.
            drivers: Optional list of driver abbreviations.

        Returns:
            DataFrame with blended race predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        race_df = df.copy()

        # If actual quali results provided, inject them into the features
        if actual_quali is not None:
            for _, qrow in actual_quali.iterrows():
                mask = race_df["Abbreviation"] == qrow["Abbreviation"]
                if mask.any():
                    race_df.loc[mask, "QualifyingPosition"] = qrow["QualifyingPosition"]
                    race_df.loc[mask, "GridPosition"] = qrow["QualifyingPosition"]
                    if "BestQualiTime_s" in qrow:
                        race_df.loc[mask, "BestQualiTime_s"] = qrow.get("BestQualiTime_s", np.nan)

            # Recompute grid-derived features
            race_df["IsPoleSitter"] = (race_df["GridPosition"] == 1).astype(int)
            race_df["IsFrontRow"] = (race_df["GridPosition"] <= 2).astype(int)
            race_df["IsTopThreeGrid"] = (race_df["GridPosition"] <= 3).astype(int)
            race_df["IsTopTenGrid"] = (race_df["GridPosition"] <= 10).astype(int)
            max_grid = race_df["GridPosition"].max()
            race_df["GridPosition_norm"] = race_df["GridPosition"] / max(max_grid, 1)

        if drivers is None:
            drivers = race_df["Abbreviation"].tolist()

        # ── XGBoost race predictions ──
        tree_preds = self.tree_model.predict(race_df)

        # ── Elo predictions ──
        elo_preds = self.elo_model.predict_probabilities(drivers)

        # ── Blend ──
        # The backtest showed Elo's ExpectedPosition drags the tree down in every
        # regime, so by default the finishing-ORDER comes from the tree alone.
        # Elo is still merged in for its WinProb/PodiumProb, which are a separate
        # (and useful) output. Set BLEND_ELO_INTO_POSITION=True to restore the old
        # dynamic positional blend.
        blend_elo = getattr(config, "BLEND_ELO_INTO_POSITION", False)
        tree_weight, elo_weight = self._get_blend_weights() if blend_elo else (1.0, 0.0)

        merged = tree_preds.merge(
            elo_preds[["Abbreviation", "EloRating", "WinProb", "PodiumProb", "ExpectedPosition"]],
            on="Abbreviation",
            how="left",
        )

        if blend_elo:
            merged["BlendedPosition"] = (
                tree_weight * merged["PredictedPosition"]
                + elo_weight * merged["ExpectedPosition"]
            )
        else:
            merged["BlendedPosition"] = merged["PredictedPosition"]
        merged["TreeWeight"] = tree_weight
        merged["EloWeight"] = elo_weight

        # ── Win/podium probabilities from the grid+tree blend (not Elo) ──
        if getattr(self, "prob_model", None) is not None and self.prob_model.is_trained:
            probs = self.prob_model.predict_probabilities(race_df)
            merged = merged.drop(columns=["WinProb", "PodiumProb"]).merge(
                probs, on="Abbreviation", how="left"
            )

        # ── Finishing order ──
        # Reconciliation: the calibrated win/podium probabilities set the TOP of
        # the order (where the backtest shows they're most reliable — they own the
        # winner/podium call), and the tree sets everything below. This recovers
        # grid-level winner accuracy and makes the headline P1 == the win favourite
        # (100% coherence) at negligible MAE cost. Toggle via config.
        reconcile = getattr(config, "RECONCILE_ORDER_WITH_PROBS", True)
        has_probs = "WinProb" in merged.columns and merged["WinProb"].notna().any()
        if reconcile and has_probs and len(merged) >= 3:
            abbr = merged["Abbreviation"].to_numpy()
            win = merged["WinProb"].to_numpy(dtype=float)
            pod = merged["PodiumProb"].to_numpy(dtype=float)
            tpos = merged["BlendedPosition"].to_numpy(dtype=float)
            p1 = int(np.argmax(win))
            pod_rest = [i for i in np.argsort(-pod) if i != p1][:2]
            top = [p1] + pod_rest
            rest = [i for i in np.argsort(tpos) if i not in top]
            recon_idx = top + rest
            final_rank = np.empty(len(abbr), dtype=int)
            final_rank[recon_idx] = np.arange(1, len(abbr) + 1)
            merged["FinalRank"] = final_rank
        else:
            merged["FinalRank"] = merged["BlendedPosition"].rank(method="min").astype(int)
        merged = merged.sort_values("FinalRank").reset_index(drop=True)

        # Add grid position for context
        grid_map = {}
        if "GridPosition" in race_df.columns:
            for _, row in race_df.iterrows():
                drv = row.get("Abbreviation")
                gp = row.get("GridPosition")
                if drv and pd.notna(gp):
                    grid_map[drv] = gp
        # Fallback to QualifyingPosition if GridPosition is missing
        if not grid_map and "QualifyingPosition" in race_df.columns:
            for _, row in race_df.iterrows():
                drv = row.get("Abbreviation")
                qp = row.get("QualifyingPosition")
                if drv and pd.notna(qp):
                    grid_map[drv] = qp

        if grid_map:
            merged["GridPosition"] = merged["Abbreviation"].map(grid_map)

        return merged

    def predict(self, df, drivers=None):
        """Backward compatible — calls predict_race."""
        return self.predict_race(df, drivers=drivers)

    def _get_blend_weights(self) -> tuple[float, float]:
        """
        Compute dynamic blend weights based on how many races
        into the current regulation era we are.

        With only 2-3 races in 2026, the tree model trained on 2022-2025
        data doesn't know the new pecking order. Elo adapts instantly.
        """
        races_in_era = self.elo_model._races_in_era
        min_races = self.params["min_races_for_tree_model"]
        ramp_races = self.params["tree_weight_ramp_races"]
        max_tree = self.params["max_tree_weight"]

        if races_in_era < min_races:
            # Very early in era — Elo dominates heavily
            tree_weight = 0.15
            elo_weight = 0.85
        elif races_in_era < ramp_races:
            # Ramping up tree weight linearly
            progress = (races_in_era - min_races) / (ramp_races - min_races)
            tree_weight = 0.15 + progress * (max_tree - 0.15)
            elo_weight = 1.0 - tree_weight
        else:
            # Mature era — tree model dominates
            tree_weight = max_tree
            elo_weight = 1.0 - max_tree

        return tree_weight, elo_weight

    def _evaluate_ensemble(
        self,
        full_df: pd.DataFrame,
        feature_cols: list,
        test_year: int,
    ) -> dict:
        """
        Honest held-out evaluation: fit fresh tree + Elo on data BEFORE the test
        season, then score the test season. This avoids the trap of grading the
        final all-data model on rows it trained on (which inflates to ~100%).
        For a fully out-of-sample, race-by-race estimate use evaluation.py.
        """
        train = full_df[full_df["Year"] < test_year]
        test_df = full_df[full_df["Year"] == test_year]
        if train.empty or test_df.empty:
            print("  (skipping — no pre-test data to hold out)")
            return {}

        # Held-out tree — same target_mode as production, fit quietly on pre-test data
        ho_tree = F1GradientBoostModel(target_mode=self.tree_model.target_mode)
        ho_tree.fit_quiet(train, feature_cols)

        # Held-out Elo (suppress its regulation-change prints)
        blend_elo = getattr(config, "BLEND_ELO_INTO_POSITION", False)
        ho_elo = None
        if blend_elo:
            with contextlib.redirect_stdout(io.StringIO()):
                ho_elo = F1EloRating()
                ho_elo.process_historical(train)
        tree_weight, elo_weight = (
            self._get_blend_weights() if blend_elo else (1.0, 0.0)
        )
        print(f"  Blend weights: Tree={tree_weight:.2f}, Elo={elo_weight:.2f}"
              f"{'' if blend_elo else '  (Elo used for win/podium probs only)'}")

        correct_winner = correct_podium = total_races = 0
        all_mae = []

        for (year, rnd), race_group in test_df.groupby(["Year", "RoundNumber"]):
            rg = race_group.dropna(subset=["Target_FinishPosition"])
            if rg.empty:
                continue
            actual = rg["Target_FinishPosition"].values
            tree_pos = ho_tree.predict(rg).set_index("Abbreviation")["PredictedPosition"]
            tree_pos = rg["Abbreviation"].map(tree_pos).values

            if blend_elo and ho_elo is not None:
                elo_preds = ho_elo.predict_probabilities(rg["Abbreviation"].tolist())
                elo_map = elo_preds.set_index("Abbreviation")["ExpectedPosition"]
                elo_pos = rg["Abbreviation"].map(elo_map).fillna(11.0).values
                blended = tree_weight * tree_pos + elo_weight * elo_pos
            else:
                blended = tree_pos

            all_mae.append(np.mean(np.abs(actual - blended)))
            actual_order = np.argsort(actual)
            pred_order = np.argsort(blended)
            correct_winner += int(actual_order[0] == pred_order[0])
            correct_podium += int(len(set(actual_order[:3]) & set(pred_order[:3])) > 0)
            total_races += 1

        return {
            "ensemble_mae": np.mean(all_mae) if all_mae else 0,
            "winner_accuracy": correct_winner / max(total_races, 1),
            "podium_accuracy": correct_podium / max(total_races, 1),
            "total_races": total_races,
            "tree_weight": tree_weight,
            "elo_weight": elo_weight,
        }

    def save(self, path: Path | None = None):
        """Save all models."""
        self.tree_model.save()
        self.elo_model.save()
        if self.has_quali_model and self.quali_model is not None:
            self.quali_model.save(config.MODEL_DIR / "xgboost_quali_model.joblib")
        if self.prob_model.is_trained:
            self.prob_model.save()
        # Save ensemble state
        joblib.dump({
            "has_quali_model": self.has_quali_model,
            "params": self.params,
        }, config.MODEL_DIR / "ensemble_state.joblib")
        print("Ensemble saved (all models).")

    def load(self, path: Path | None = None):
        """Load all models."""
        self.tree_model.load()
        self.elo_model.load()
        # Load ensemble state
        state_path = config.MODEL_DIR / "ensemble_state.joblib"
        if state_path.exists():
            state = joblib.load(state_path)
            self.has_quali_model = state.get("has_quali_model", False)
            self.params = state.get("params", self.params)
        if self.has_quali_model:
            quali_path = config.MODEL_DIR / "xgboost_quali_model.joblib"
            if quali_path.exists():
                self.quali_model = F1GradientBoostModel()
                self.quali_model.load(quali_path)
        prob_path = config.MODEL_DIR / "probability_model.joblib"
        if prob_path.exists():
            self.prob_model.load(prob_path)
        self.is_trained = True
        print("Ensemble loaded (all models).")


# ─────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load features
    df = pd.read_parquet(config.DATA_DIR / "features.parquet")
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(df)

    # Train ensemble
    ensemble = F1Ensemble()
    metrics = ensemble.train(df, feature_cols)

    # Predict next race for active 2026 drivers
    latest = df[df["Year"] == df["Year"].max()]
    if not latest.empty:
        last_round = latest["RoundNumber"].max()
        last_race = latest[latest["RoundNumber"] == last_round]

        print(f"\n{'='*60}")
        print(f"  PREDICTION (based on latest data)")
        print(f"{'='*60}")
        predictions = ensemble.predict(last_race)
        for _, row in predictions.iterrows():
            prob_bar = "█" * int(row.get("WinProb", 0) * 100)
            print(
                f"  P{row['FinalRank']:2d}  {row['Abbreviation']:4s}  "
                f"{row['TeamName']:20s}  "
                f"Elo:{row['EloRating']:5.0f}  "
                f"Win:{row.get('WinProb', 0)*100:4.1f}%  {prob_bar}"
            )

    # Save
    ensemble.save()
