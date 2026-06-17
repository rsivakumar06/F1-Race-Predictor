"""
Telemetry Comparison Visualizations.

Overlay speed, throttle, and brake traces for two drivers on their
fastest laps. Shows exactly where time is gained/lost.

Usage:
    from visualizations.telemetry_plots import TelemetryPlots
    plots = TelemetryPlots()
    session = pipeline.load_session_for_visualization(2026, 'Australia', 'Q')
    plots.speed_trace_comparison(session, 'RUS', 'ANT')
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    import fastf1
    import fastf1.plotting
    fastf1.plotting.setup_mpl(color_scheme='fastf1')
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
except Exception:
    pass


class TelemetryPlots:
    """Compare car telemetry between drivers."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or config.PLOT_DIR

    def speed_trace_comparison(self, session, driver1: str, driver2: str,
                                lap: str = "fastest", save: bool = True):
        """
        Overlay speed traces for two drivers.
        Shows braking points, top speed differences, and corner speeds.
        """
        tel1, tel2, color1, color2 = self._get_telemetry_pair(session, driver1, driver2, lap)

        fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                                  gridspec_kw={"height_ratios": [3, 1, 1, 1]},
                                  sharex=True)

        event_name = session.event['EventName']
        year = session.event.year

        # Speed
        axes[0].plot(tel1["Distance"], tel1["Speed"], label=driver1,
                     color=color1, linewidth=1.2)
        axes[0].plot(tel2["Distance"], tel2["Speed"], label=driver2,
                     color=color2, linewidth=1.2)
        axes[0].set_ylabel("Speed (km/h)")
        axes[0].set_title(f"{event_name} {year} — Telemetry: {driver1} vs {driver2}")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.2)

        # Throttle
        axes[1].plot(tel1["Distance"], tel1["Throttle"], color=color1, linewidth=1)
        axes[1].plot(tel2["Distance"], tel2["Throttle"], color=color2, linewidth=1)
        axes[1].set_ylabel("Throttle %")
        axes[1].set_ylim(-5, 105)
        axes[1].grid(alpha=0.2)

        # Brake
        axes[2].plot(tel1["Distance"], tel1["Brake"].astype(int), color=color1, linewidth=1)
        axes[2].plot(tel2["Distance"], tel2["Brake"].astype(int), color=color2, linewidth=1)
        axes[2].set_ylabel("Brake")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(alpha=0.2)

        # Gear
        axes[3].plot(tel1["Distance"], tel1["nGear"], color=color1, linewidth=1)
        axes[3].plot(tel2["Distance"], tel2["nGear"], color=color2, linewidth=1)
        axes[3].set_ylabel("Gear")
        axes[3].set_xlabel("Distance (m)")
        axes[3].grid(alpha=0.2)

        plt.tight_layout()
        if save:
            path = self.output_dir / f"telemetry_{driver1}v{driver2}_{year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig

    def delta_time_on_track(self, session, driver1: str, driver2: str,
                             lap: str = "fastest", save: bool = True):
        """
        Show the time delta between two drivers overlaid on the track map.
        Positive = driver1 is ahead, negative = driver2 is ahead.
        """
        from matplotlib.collections import LineCollection
        from matplotlib import colormaps

        tel1, tel2, color1, color2 = self._get_telemetry_pair(session, driver1, driver2, lap)

        # Compute delta: cumulative time difference at each distance point
        # Merge on distance (nearest match)
        ref = tel1[["Distance", "Time"]].copy()
        ref["Time_s_1"] = ref["Time"].dt.total_seconds()
        comp = tel2[["Distance", "Time"]].copy()
        comp["Time_s_2"] = comp["Time"].dt.total_seconds()

        merged = pd.merge_asof(
            ref.sort_values("Distance"),
            comp.sort_values("Distance"),
            on="Distance",
            direction="nearest",
        )
        merged["Delta"] = merged["Time_s_1"] - merged["Time_s_2"]

        # Get track coordinates from first driver
        x = tel1["X"].values
        y = tel1["Y"].values
        delta = np.interp(
            np.linspace(0, 1, len(x)),
            np.linspace(0, 1, len(merged)),
            merged["Delta"].values,
        )

        # Rotate track to match official orientation
        try:
            circuit_info = session.get_circuit_info()
            angle = circuit_info.rotation / 180 * np.pi
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            x, y = x_rot, y_rot
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(12, 10))

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        max_delta = max(abs(delta.min()), abs(delta.max()), 0.5)
        norm = plt.Normalize(-max_delta, max_delta)
        lc = LineCollection(segments, cmap=colormaps["RdBu_r"], norm=norm, linewidth=4)
        lc.set_array(delta[:-1])
        ax.add_collection(lc)

        # Background track
        ax.plot(x, y, color='gray', linewidth=8, alpha=0.3, zorder=0)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{session.event['EventName']} {session.event.year}\n"
            f"Delta: {driver1} vs {driver2} "
            f"(Red = {driver2} faster, Blue = {driver1} faster)"
        )

        cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
        cbar.set_label("Time Delta (s)")

        plt.tight_layout()
        if save:
            path = self.output_dir / f"delta_{driver1}v{driver2}_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig

    def _get_telemetry_pair(self, session, driver1, driver2, lap="fastest"):
        """Helper to load telemetry for two drivers."""
        d1_laps = session.laps.pick_drivers(driver1)
        d2_laps = session.laps.pick_drivers(driver2)

        if lap == "fastest":
            l1 = d1_laps.pick_fastest()
            l2 = d2_laps.pick_fastest()
        else:
            l1 = d1_laps[d1_laps["LapNumber"] == int(lap)].iloc[0]
            l2 = d2_laps[d2_laps["LapNumber"] == int(lap)].iloc[0]

        tel1 = l1.get_telemetry().add_distance()
        tel2 = l2.get_telemetry().add_distance()

        try:
            color1 = fastf1.plotting.get_driver_color(driver1, session)
            color2 = fastf1.plotting.get_driver_color(driver2, session)
        except Exception:
            color1, color2 = "#FF6B6B", "#4ECDC4"

        return tel1, tel2, color1, color2


if __name__ == "__main__":
    session = fastf1.get_session(2025, 1, "Q")
    session.load()

    plots = TelemetryPlots()
    plots.speed_trace_comparison(session, "NOR", "VER")
    plots.delta_time_on_track(session, "NOR", "VER")
