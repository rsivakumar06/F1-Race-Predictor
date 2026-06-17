# F1 Race Winner Predictor

A machine learning project that uses [FastF1](https://docs.fastf1.dev/) timing and
telemetry data to predict Formula 1 race results. Built to handle the 2026
regulation era with a hybrid historical + adaptive approach.

## Project Structure

```
f1_predictor/
├── config.py                  # Central configuration (seasons, features, model params)
├── requirements.txt           # Python dependencies
├── main.py                    # Entry point — run the full pipeline or individual steps
├── data/
│   ├── pipeline.py            # Data ingestion from FastF1
│   ├── feature_engineering.py # Feature extraction and transformation
│   └── processed/             # Cached processed DataFrames (auto-created, not tracked)
├── models/
│   ├── gradient_boost.py      # Gradient-boosted tree model (XGBoost; optional LightGBM backend)
│   ├── elo_rating.py          # Bayesian Elo rating system
│   ├── ensemble.py            # Two-stage ensemble (qualifying + race)
│   ├── probability_model.py   # Win/podium probability estimation
│   └── saved/                 # Serialized trained models (auto-created, not tracked)
├── visualizations/
│   ├── race_plots.py          # Lap times, tyre strategy, position charts
│   ├── telemetry_plots.py     # Speed/throttle/brake trace comparisons
│   ├── track_viz.py           # Circuit rendering + speed heatmaps
│   └── track_animation.py     # Animated driver positions around the track
├── utils/
│   ├── helpers.py             # Time conversion, data cleaning utilities
│   └── constants.py           # Driver/team mappings, compound colors
└── cache/                     # FastF1 cache directory (auto-created, not tracked)
```

> **Note on cloning:** `data/processed/`, `models/saved/`, and `cache/` are
> generated artifacts and are not committed to the repository. A fresh clone
> contains source code only — you regenerate the data and trained models by
> running the pipeline (see Quick Start). The initial data collection downloads
> several seasons of session data from FastF1 and is rate-limited, so the first
> full run can take a while.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and process data
```bash
# Download all sessions for configured seasons (default: 2022–2026)
python main.py --step collect

# Or collect a single season (preferred for incremental updates)
python main.py --step collect --year 2026

# Run feature engineering on collected data
python main.py --step features
```

### 3. Train the model
```bash
python main.py --step train
```

### 4. Predict a race
```bash
python main.py --step predict --race "Australia" --year 2026
```

### 5. Visualize
```bash
# Generate race analysis plots
python main.py --step visualize --race "Australia" --year 2026

# Animate driver positions on track
python main.py --step animate --race "Australia" --year 2026 --driver "VER"
```

### 6. Run the full pipeline
```bash
python main.py --step all
```

Run commands from the repository root — `main.py` adds the project root to the
Python path so the `data.`, `models.`, and `visualizations.` imports resolve.

## Architecture

The system uses a **two-stage pipeline**:

1. **Stage 1 — Qualifying grid.** Predicts the qualifying order from pre-qualifying
   features (practice pace, historical form). This produces the grid that Stage 2
   consumes.
2. **Stage 2 — Race result.** Predicts finishing order from the qualifying grid plus
   race-specific features.

The race model is built around a **gradient-boosted tree** (XGBoost, with an optional
LightGBM backend). It predicts positions gained or lost relative to the grid rather
than absolute finishing position, which keeps the grid's strong prior information in
the model. A **Bayesian Elo rating system** contributes win and podium probability
estimates.

### Handling the 2026 regulation break

The 2026 season introduced a major regulation change:

- New power units (roughly 50/50 ICE/electric split, no MGU-H)
- Active aerodynamics (front + rear wing adjustment)
- Smaller, lighter, narrower cars
- Sustainable fuels

Because a regulation break resets a lot of competitive order, the ensemble includes
dynamic weighting machinery that can lean on the faster-adapting rating component
early in a new era and shift toward the tree model as more races accumulate and the
structural features become informative. Component weighting is controlled in
`config.py`.

## Data Sources

All data comes from [FastF1](https://docs.fastf1.dev/), which provides:

- Lap timing (sector times, speed traps, pit stops)
- Car telemetry (speed, RPM, gear, throttle, brake, DRS)
- Car position (GPS coordinates for track mapping)
- Tyre data (compound, tyre life, stint info)
- Weather (air/track temperature, humidity, wind, rainfall)
- Session results (grid, finish, points, status)
- Circuit info (corner locations, marshal sectors)

FastF1 requires no API key. Data is available from the 2018 season onward.

## License

This project is released under the MIT License. See `LICENSE` for details.
