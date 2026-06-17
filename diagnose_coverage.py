"""
Data coverage diagnostic.

Compares the rounds actually in all_races_raw.parquet against the official
FastF1 schedule (same filtering the pipeline uses: drop testing + round 0),
so you can see exactly which Grands Prix never made it into the dataset.

Run from the project root:
    python diagnose_coverage.py
"""
import pandas as pd
import config

try:
    import fastf1
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    HAVE_FF1 = True
except Exception:
    HAVE_FF1 = False

df = pd.read_parquet(config.DATA_DIR / "all_races_raw.parquet")

print("Rows per season:")
print(df.groupby("Year").size().to_string())

print("\nRounds present per season:")
print(df.groupby("Year")["RoundNumber"].nunique().to_string())

# how many rows actually have a finishing result (what the race model trains on)
if "FinishPosition" in df.columns:
    with_result = df.dropna(subset=["FinishPosition"])
    print("\nRows WITH a finish result per season (race-model training rows):")
    print(with_result.groupby("Year").size().to_string())

print("\nMissing rounds per season (vs official schedule):")
for year in sorted(df["Year"].unique()):
    year = int(year)
    present = {int(r) for r in df[df["Year"] == year]["RoundNumber"].unique()}

    if not HAVE_FF1:
        print(f"  {year}: present={sorted(present)} (fastf1 unavailable; cannot compare)")
        continue

    try:
        sched = fastf1.get_event_schedule(year)
        sched = sched[(sched["EventFormat"] != "testing") & (sched["RoundNumber"] > 0)]
        expected = {int(r) for r in sched["RoundNumber"]}
    except Exception as e:
        print(f"  {year}: schedule lookup failed ({e}); present={sorted(present)}")
        continue

    missing = sorted(expected - present)
    # map missing round numbers to event names for readability
    names = {int(r["RoundNumber"]): r["EventName"] for _, r in sched.iterrows()}
    missing_named = [f"R{m} {names.get(m, '?')}" for m in missing]
    print(f"  {year}: {len(present)}/{len(expected)} present"
          + (f"  | missing: {missing_named}" if missing else "  | complete"))
