"""
Central configuration for the F1 Race Predictor.
Edit these values to control data collection, feature engineering, and model behavior.
"""
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
PLOT_DIR = PROJECT_ROOT / "visualizations" / "output"

# Create directories on import
for d in [CACHE_DIR, DATA_DIR, MODEL_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Data Collection
# ─────────────────────────────────────────────
# Seasons to collect data from (2022-2025 = previous regs, 2026 = new regs)
HISTORICAL_SEASONS = [2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026

ALL_SEASONS = HISTORICAL_SEASONS + [CURRENT_SEASON]

# Session types to load
SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "R"]

# What data to load per session
LOAD_OPTIONS = {
    "laps": True,
    "telemetry": True,
    "weather": True,
    "messages": True,
}

# Practice session loading (lighter — no telemetry needed for bulk collection)
PRACTICE_LOAD_OPTIONS = {
    "laps": True,
    "telemetry": False,
    "weather": True,
    "messages": False,
}

# Collection resilience. Retries with exponential backoff guard against
# transient API/network failures; the inter-round sleep keeps long bulk
# runs from hammering the FastF1 endpoints. Raise COLLECT_SLEEP_S if a
# full-season pull keeps dropping its tail.
COLLECT_RETRIES = 3
COLLECT_RETRY_BACKOFF_S = 3.0
COLLECT_SLEEP_S = 1.0

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
# Rolling window sizes for driver/team statistics
ROLLING_WINDOWS = {
    "short": 3,     # Last 3 races — recent form
    "medium": 5,    # Last 5 races — medium-term trend
    "long": 10,     # Last 10 races — season form
}

# Features that carry across regulation eras (used for historical training)
CROSS_ERA_FEATURES = [
    "driver_quali_avg_gap_to_pole",
    "driver_quali_consistency",
    "driver_race_completion_rate",
    "driver_wet_performance_delta",
    "driver_overtaking_rate",
    "driver_first_lap_positions_gained",
    "team_pit_stop_avg_duration",
    "team_pit_stop_consistency",
    "team_reliability_rate",
    "team_points_trajectory",
    "circuit_type",              # street / permanent / hybrid
    "circuit_corners_count",
    "circuit_straight_ratio",    # proportion of track that is straights
]

# 2026-specific features (only available once 2026 data exists)
NEW_ERA_FEATURES = [
    "driver_energy_management_efficiency",
    "team_active_aero_straight_speed_gain",
    "driver_boost_usage_effectiveness",
    "circuit_braking_to_straight_ratio",  # proxy for energy harvest potential
]

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────
# XGBoost parameters (tuned for small F1 datasets — avoid overfitting)
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",  # Predict finishing position
    "eval_metric": "mae",
    "random_state": 42,
}

# Objective used when a model runs in target_mode="rank" (learning-to-rank).
# "rank:pairwise" is the most robust across XGBoost versions. "rank:ndcg" is
# more top-of-order focused (weights getting the winner/podium right) and is
# worth trying once pairwise is confirmed working.
XGBOOST_RANK_OBJECTIVE = "rank:pairwise"

# Elo rating system parameters
ELO_PARAMS = {
    "initial_rating": 1500,
    "k_factor": 32,               # How quickly ratings change per race
    "decay_factor": 0.95,         # Between-season rating decay toward mean
    "home_advantage": 0,          # No home advantage in F1
    "regulation_reset_factor": 0.5,  # How much to reset ratings on reg change
    # Monte Carlo for win/podium/expected-position from ratings.
    # score_noise_sigma is the per-race Elo variance; ~200 means a 200-pt gap
    # is roughly a 1-sigma edge. Tune down to make favourites win more often.
    "mc_simulations": 10000,
    "mc_score_noise_sigma": 200.0,
}

# Global RNG seed — makes Monte Carlo predictions reproducible run-to-run.
RANDOM_SEED = 42

# Ensemble blending (shifts weight toward Elo early in a new regulation era)
ENSEMBLE_PARAMS = {
    "min_races_for_tree_model": 5,   # Races before tree model gets meaningful weight
    "tree_weight_ramp_races": 12,    # Races to reach full tree weight
    "max_tree_weight": 0.6,          # Max weight for tree model at maturity
    "max_elo_weight": 0.4,           # Max weight for Elo at maturity
}

# Whether Elo's expected position is blended into the finishing-ORDER prediction.
# The walk-forward backtest showed it consistently hurt (Elo is the weakest
# positional signal in every season), so default off — the order comes from the
# tree alone. Elo's win/podium probabilities are still produced regardless.
# Set True to restore the old dynamic positional blend (uses ENSEMBLE_PARAMS).
BLEND_ELO_INTO_POSITION = False

# Win/podium probabilities come from a blend of the grid prior (calibration) and
# XGBoost classifiers (discrimination). This is grid's weight; the calibration
# backtest found 50/50 best. Elo's Monte Carlo is no longer used for probabilities.
PROB_GRID_WEIGHT = 0.5

# Reconcile the displayed finishing order with the win/podium probabilities:
# probabilities set the top (P1 = win favourite, P2-P3 = podium favourites), the
# tree sets the rest. The backtest showed this recovers grid-level winner accuracy
# (45% -> 57%) and makes the headline coherent with the odds, at negligible MAE cost.
RECONCILE_ORDER_WITH_PROBS = True

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
PLOT_STYLE = "fastf1"  # Use FastF1's dark theme
FIGURE_DPI = 150
ANIMATION_FPS = 30
ANIMATION_INTERVAL_MS = 50  # ms between animation frames
