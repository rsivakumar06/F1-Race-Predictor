# F1 Race Winner Predictor

A machine learning project that uses FastF1 telemetry data to predict Formula 1 race winners.
Built to handle the 2026 regulation era with a hybrid historical + adaptive approach.

## Project Structure

```
f1_predictor/
├── config.py                 # Central configuration (seasons, features, model params)
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point — run full pipeline or individual steps
├── data/
│   ├── pipeline.py           # Data ingestion from FastF1 API
│   ├── feature_engineering.py # Feature extraction and transformation
│   └── processed/            # Cached processed DataFrames (auto-created)
├── models/
│   ├── gradient_boost.py     # XGBoost / LightGBM race predictor
│   ├── elo_rating.py         # Bayesian Elo/Glicko rating system
│   ├── ensemble.py           # Blend tree model + Elo predictions
│   └── saved/                # Serialized trained models (auto-created)
├── visualizations/
│   ├── race_plots.py         # Lap times, tyre strategy, position charts
│   ├── telemetry_plots.py    # Speed/throttle/brake trace comparisons
│   ├── track_viz.py          # Circuit rendering + speed heatmaps
│   └── track_animation.py    # Animated driver positions around track
├── utils/
│   ├── helpers.py            # Time conversion, data cleaning utilities
│   └── constants.py          # Driver/team mappings, compound colors
└── cache/                    # FastF1 API cache directory
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and process data
```bash
# Download all sessions for configured seasons (default: 2022-2026)
python main.py --step collect

# Run feature engineering on collected data
python main.py --step features
```

### 3. Train the model
```bash
python main.py --step train
```

### 4. Predict next race
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

### 6. Run everything
```bash
python main.py --step all
```

## Handling the 2026 Regulation Break

The 2026 season introduced the biggest regulation change in F1 history:
- New power units (50/50 ICE/electric split, no MGU-H)
- Active aerodynamics (front + rear wing adjustment)
- Smaller, lighter, narrower cars
- Sustainable fuels

The model handles this with a **two-layer approach**:
1. **Gradient Boosted Trees** trained on 2022-2025 data for regulation-agnostic
   features (driver consistency, qualifying conversion, team reliability)
2. **Bayesian Elo System** that starts fresh for 2026 and rapidly adapts

The ensemble blends both, weighting the Elo component more heavily early in 2026
and gradually increasing the tree model's weight as more 2026 data accumulates.

## Data Sources

All data comes from FastF1 (https://docs.fastf1.dev/), which provides:
- Lap timing (sector times, speed traps, pit stops)
- Car telemetry (speed, RPM, gear, throttle, brake, DRS at ~4Hz)
- Car position (X/Y GPS coordinates for track mapping)
- Tyre data (compound, tyre life, stint info)
- Weather (air/track temp, humidity, wind, rainfall)
- Session results (grid, finish, points, status)
- Circuit info (corner locations, marshal sectors, track rotation)

Data is available from the 2018 season onwards.
