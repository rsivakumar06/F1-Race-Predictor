"""
Race Analysis Visualizations.

Lap time distributions, position charts, tyre strategy, and race pace comparisons.
Uses FastF1's built-in plotting utilities and Matplotlib.

Usage:
    from visualizations.race_plots import RacePlots
    plots = RacePlots()
    session = pipeline.load_session_for_visualization(2026, 'Australia', 'R')
    plots.lap_time_distribution(session)
    plots.position_chart(session)
    plots.tyre_strategy(session)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Apply FastF1 plotting setup
try:
    import fastf1
    import fastf1.plotting
    fastf1.plotting.setup_mpl(color_scheme='fastf1')
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
except Exception:
    pass

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC300",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": "#888888",
}


class RacePlots:
    """Generate race analysis visualizations."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or config.PLOT_DIR

    def lap_time_distribution(self, session, drivers: list | None = None, save: bool = True):
        """
        Violin/box plot of lap time distributions per driver.
        Shows consistency and pace — tight distributions = consistent driver.
        """
        laps = session.laps.pick_quicklaps()
        if drivers:
            laps = laps[laps["Driver"].isin(drivers)]

        # Convert lap times to seconds
        laps = laps.copy()
        laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

        # Get finishing order for sorting
        results = session.results
        driver_order = results.sort_values("Position")["Abbreviation"].tolist()
        drivers_in_data = [d for d in driver_order if d in laps["Driver"].unique()]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Collect data for box plot
        data = []
        colors = []
        for driver in drivers_in_data:
            driver_laps = laps[laps["Driver"] == driver]["LapTime_s"].dropna()
            data.append(driver_laps.values)
            try:
                color = fastf1.plotting.get_driver_color(driver, session)
                colors.append(color)
            except Exception:
                colors.append("#AAAAAA")

        bp = ax.boxplot(data, labels=drivers_in_data, patch_artist=True,
                        showfliers=True, widths=0.6)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Lap Time (seconds)")
        ax.set_title(f"{session.event['EventName']} {session.event.year} — Lap Time Distribution")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save:
            path = self.output_dir / f"laptimes_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig

    def position_chart(self, session, save: bool = True):
        """
        Lap-by-lap position chart for all drivers.
        Shows overtakes, safety car impacts, and race evolution.
        """
        laps = session.laps

        fig, ax = plt.subplots(figsize=(16, 10))

        results = session.results
        driver_order = results.sort_values("Position")["Abbreviation"].tolist()

        for driver in driver_order:
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
            if driver_laps.empty:
                continue

            try:
                color = fastf1.plotting.get_driver_color(driver, session)
            except Exception:
                color = "#AAAAAA"

            ax.plot(
                driver_laps["LapNumber"],
                driver_laps["Position"],
                label=driver,
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )

        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Position")
        ax.set_title(f"{session.event['EventName']} {session.event.year} — Position Chart")
        ax.set_ylim(0.5, len(driver_order) + 0.5)
        ax.invert_yaxis()
        ax.set_yticks(range(1, len(driver_order) + 1))
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            path = self.output_dir / f"positions_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig

    def tyre_strategy(self, session, save: bool = True):
        """
        Horizontal bar chart showing each driver's tyre strategy.
        Each stint is a colored bar (compound color) showing laps on that tyre.
        """
        laps = session.laps
        results = session.results
        driver_order = results.sort_values("Position")["Abbreviation"].tolist()

        fig, ax = plt.subplots(figsize=(16, 10))

        for i, driver in enumerate(driver_order):
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
            if driver_laps.empty:
                continue

            stints = driver_laps.groupby("Stint").agg(
                start=("LapNumber", "min"),
                end=("LapNumber", "max"),
                compound=("Compound", "first"),
            )

            for _, stint in stints.iterrows():
                compound = stint["compound"]
                color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])
                width = stint["end"] - stint["start"] + 1
                ax.barh(
                    i, width, left=stint["start"],
                    color=color, edgecolor="black", linewidth=0.5, height=0.7,
                )

        ax.set_yticks(range(len(driver_order)))
        ax.set_yticklabels(driver_order)
        ax.set_xlabel("Lap Number")
        ax.set_title(f"{session.event['EventName']} {session.event.year} — Tyre Strategy")
        ax.invert_yaxis()

        # Legend
        legend_patches = [
            mpatches.Patch(color=c, label=name)
            for name, c in COMPOUND_COLORS.items()
            if name in laps["Compound"].unique()
        ]
        ax.legend(handles=legend_patches, loc='lower right')
        ax.grid(axis='x', alpha=0.2)

        plt.tight_layout()
        if save:
            path = self.output_dir / f"tyrestrategy_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig

    def race_pace_comparison(self, session, driver1: str, driver2: str, save: bool = True):
        """
        Compare race pace between two drivers lap by lap.
        Shows who was faster where and the gap evolution.
        """
        laps = session.laps

        d1_laps = laps[laps["Driver"] == driver1].sort_values("LapNumber").copy()
        d2_laps = laps[laps["Driver"] == driver2].sort_values("LapNumber").copy()

        d1_laps["LapTime_s"] = d1_laps["LapTime"].dt.total_seconds()
        d2_laps["LapTime_s"] = d2_laps["LapTime"].dt.total_seconds()

        try:
            color1 = fastf1.plotting.get_driver_color(driver1, session)
            color2 = fastf1.plotting.get_driver_color(driver2, session)
        except Exception:
            color1, color2 = "#FF6B6B", "#4ECDC4"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Lap times
        ax1.plot(d1_laps["LapNumber"], d1_laps["LapTime_s"], label=driver1,
                 color=color1, linewidth=1.5, alpha=0.85)
        ax1.plot(d2_laps["LapNumber"], d2_laps["LapTime_s"], label=driver2,
                 color=color2, linewidth=1.5, alpha=0.85)
        ax1.set_ylabel("Lap Time (s)")
        ax1.set_title(f"{session.event['EventName']} {session.event.year} — {driver1} vs {driver2}")
        ax1.legend()
        ax1.grid(alpha=0.2)

        # Gap (cumulative time difference)
        merged = d1_laps[["LapNumber", "LapTime_s"]].merge(
            d2_laps[["LapNumber", "LapTime_s"]],
            on="LapNumber", suffixes=(f"_{driver1}", f"_{driver2}"),
        )
        merged["Gap"] = (
            merged[f"LapTime_s_{driver1}"].cumsum() - merged[f"LapTime_s_{driver2}"].cumsum()
        )

        ax2.fill_between(merged["LapNumber"], merged["Gap"], 0,
                         where=merged["Gap"] > 0, color=color2, alpha=0.4,
                         label=f"{driver2} ahead")
        ax2.fill_between(merged["LapNumber"], merged["Gap"], 0,
                         where=merged["Gap"] <= 0, color=color1, alpha=0.4,
                         label=f"{driver1} ahead")
        ax2.axhline(0, color="white", linewidth=0.5, alpha=0.5)
        ax2.set_xlabel("Lap Number")
        ax2.set_ylabel("Cumulative Gap (s)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)

        plt.tight_layout()
        if save:
            path = self.output_dir / f"pace_{driver1}v{driver2}_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig


if __name__ == "__main__":
    session = fastf1.get_session(2025, 1, "R")
    session.load(telemetry=False, weather=False, messages=False)

    plots = RacePlots()
    plots.lap_time_distribution(session)
    plots.position_chart(session)
    plots.tyre_strategy(session)
    plots.race_pace_comparison(session, "NOR", "VER")
