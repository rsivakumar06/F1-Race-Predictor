"""
Track Map Visualization.

Renders circuit maps from X/Y position data with telemetry overlays.
Speed heatmaps, gear maps, and corner annotations.

Usage:
    from visualizations.track_viz import TrackViz
    viz = TrackViz()
    session = pipeline.load_session_for_visualization(2026, 'Australia', 'R')
    viz.speed_heatmap(session, 'RUS')
    viz.draw_track_map(session, annotate_corners=True)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    import fastf1
    import fastf1.plotting
    fastf1.plotting.setup_mpl(color_scheme='fastf1')
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
except Exception:
    pass


def _rotate(xy, angle):
    """Rotate 2D points by angle (radians) around the origin."""
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot)


class TrackViz:
    """Render circuit maps with telemetry overlays."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or config.PLOT_DIR

    def draw_track_map(self, session, annotate_corners: bool = True, save: bool = True):
        """
        Draw the circuit from X/Y position data with corner annotations.
        """
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()

        try:
            circuit_info = session.get_circuit_info()
            angle = circuit_info.rotation / 180 * np.pi
        except Exception:
            circuit_info = None
            angle = 0

        x = pos["X"].values
        y = pos["Y"].values
        coords = _rotate(np.column_stack([x, y]), angle=angle)
        x, y = coords[:, 0], coords[:, 1]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(x, y, color="white", linewidth=4, alpha=0.9)
        ax.plot(x, y, color="#333333", linewidth=8, alpha=0.5, zorder=0)

        # Start/finish line
        ax.scatter(x[0], y[0], color="#FF3333", s=100, zorder=5, marker="o", label="Start/Finish")

        # Corner annotations
        if annotate_corners and circuit_info is not None:
            for _, corner in circuit_info.corners.iterrows():
                txt = f"{corner['Number']}{corner.get('Letter', '')}"
                offset_angle = corner["Angle"] / 180 * np.pi
                offset = _rotate(np.array([[500, 0]]), angle=offset_angle)[0]

                cx, cy = _rotate(
                    np.array([[corner["X"], corner["Y"]]]), angle=angle
                )[0]

                tx = cx + offset[0]
                ty = cy + offset[1]

                ax.scatter(cx, cy, color="#FFD700", s=30, zorder=4)
                ax.annotate(txt, (tx, ty), fontsize=8, color="#FFD700",
                          ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333",
                                   edgecolor="#FFD700", alpha=0.8))

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{session.event['EventName']} {session.event.year} — Circuit Map")

        plt.tight_layout()
        if save:
            path = self.output_dir / f"trackmap_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig

    def speed_heatmap(self, session, driver: str, lap: str = "fastest",
                      save: bool = True):
        """
        Track map colored by speed using LineCollection + colormap.
        Shows acceleration zones, braking points, and top speed sections.
        """
        return self._track_heatmap(session, driver, "Speed", "Speed (km/h)",
                                    "plasma", lap, save)

    def gear_map(self, session, driver: str, lap: str = "fastest",
                 save: bool = True):
        """
        Track map colored by gear selection.
        """
        return self._track_heatmap(session, driver, "nGear", "Gear",
                                    "tab10", lap, save)

    def throttle_map(self, session, driver: str, lap: str = "fastest",
                     save: bool = True):
        """
        Track map colored by throttle application.
        """
        return self._track_heatmap(session, driver, "Throttle", "Throttle %",
                                    "YlOrRd", lap, save)

    def _track_heatmap(self, session, driver, channel, label, cmap_name,
                       lap="fastest", save=True):
        """Generic track heatmap for any telemetry channel."""
        driver_laps = session.laps.pick_drivers(driver)
        if lap == "fastest":
            target_lap = driver_laps.pick_fastest()
        else:
            target_lap = driver_laps[driver_laps["LapNumber"] == int(lap)].iloc[0]

        telemetry = target_lap.get_telemetry()

        x = telemetry["X"].values
        y = telemetry["Y"].values
        color_data = telemetry[channel].values

        # Rotate track
        try:
            circuit_info = session.get_circuit_info()
            angle = circuit_info.rotation / 180 * np.pi
        except Exception:
            angle = 0

        coords = _rotate(np.column_stack([x, y]), angle=angle)
        x, y = coords[:, 0], coords[:, 1]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Background track
        ax.plot(x, y, color="black", linewidth=12, zorder=0)

        # Colored segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(color_data.min(), color_data.max())
        colormap = mpl.colormaps.get_cmap(cmap_name)
        lc = LineCollection(segments, cmap=colormap, norm=norm, linewidth=5)
        lc.set_array(color_data)
        ax.add_collection(lc)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{session.event['EventName']} {session.event.year} — "
            f"{driver} {label}"
        )

        cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
        cbar.set_label(label)

        plt.tight_layout()
        if save:
            ch = channel.lower()
            path = self.output_dir / f"track_{ch}_{driver}_{session.event.year}_{session.event['RoundNumber']}.png"
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()
        return fig


if __name__ == "__main__":
    session = fastf1.get_session(2025, 1, "R")
    session.load()

    viz = TrackViz()
    viz.draw_track_map(session)
    viz.speed_heatmap(session, "NOR")
    viz.gear_map(session, "NOR")
