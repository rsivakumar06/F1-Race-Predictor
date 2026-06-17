"""
Constants for F1 data — compound colors, team mappings, etc.
These are supplementary to FastF1's built-in mappings.
"""

# Tyre compound colors for visualization
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC300",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": "#888888",
}

# Compound performance ordering (fastest to slowest in dry conditions)
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]

# 2026 regulation era identifier
REGULATION_ERAS = {
    2022: "ground_effect_2022",
    2023: "ground_effect_2022",
    2024: "ground_effect_2022",
    2025: "ground_effect_2022",
    2026: "active_aero_2026",
}

# Session type display names
SESSION_NAMES = {
    "R": "Race",
    "Q": "Qualifying",
    "S": "Sprint",
    "SQ": "Sprint Qualifying",
    "FP1": "Free Practice 1",
    "FP2": "Free Practice 2",
    "FP3": "Free Practice 3",
}

# DNF status codes that indicate mechanical failure (vs crash or other)
MECHANICAL_DNF_STATUSES = [
    "Engine", "Gearbox", "Hydraulics", "Brakes", "Suspension",
    "Electronics", "Power Unit", "ERS", "Turbo", "Fuel system",
    "Overheating", "Oil leak", "Water leak", "Electrical",
    "Driveshaft", "Exhaust", "Clutch", "Throttle",
]

CRASH_DNF_STATUSES = [
    "Accident", "Collision", "Spun off", "Collision damage",
]
