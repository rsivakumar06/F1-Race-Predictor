"""
Track Animation — Pygame-based full race replay.

Hardware-accelerated rendering for buttery smooth 60fps.
Dark F1-style UI with live leaderboard, lap counter, driver markers.

Usage:
    from visualizations.track_animation import TrackAnimation
    anim = TrackAnimation()
    session = pipeline.load_session_for_visualization(2025, 'Australia', 'R')
    anim.animate_race(session)                   # 20x speed
    anim.animate_race(session, speed=40)         # Faster
    anim.animate_race(session, speed=5)          # Slower, more detail

Controls:
    SPACE  — Pause / Resume
    UP     — Increase speed
    DOWN   — Decrease speed
    Q/ESC  — Quit
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    import fastf1
    import fastf1.plotting
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
except Exception:
    pass

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def _rotate(xy, angle):
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot)


def _hex_to_rgb(hex_color):
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _darken(rgb, factor=0.4):
    return tuple(int(c * factor) for c in rgb)


def _lighten(rgb, factor=1.4):
    return tuple(min(255, int(c * factor)) for c in rgb)


# ─────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────
BG = (0, 0, 0)
TRACK_FILL = (40, 40, 40)
TRACK_EDGE = (255, 255, 255)
TEXT_WHITE = (224, 224, 224)
TEXT_DIM = (119, 119, 119)
TEXT_DARK = (80, 80, 80)
TEXT_OUT_OF_POINTS = (60, 60, 60)  # Dimmed for non-points positions
F1_RED = (255, 24, 1)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
BRONZE = (205, 127, 50)
POINTS_ZONE = (180, 180, 180)  # Brighter white for points positions
BOARD_BG = (10, 10, 10)
BOARD_LINE = (35, 35, 35)
GREEN_FLAG = (0, 200, 80)


class TrackAnimation:
    """Pygame-based race animation — true 60fps."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or config.PLOT_DIR

    def animate_race(self, session, speed: float = 20.0, fps: int = 60):
        """
        Full race replay with all drivers.

        Args:
            session: Loaded FastF1 race session.
            speed: Playback speed multiplier.
            fps: Target framerate.
        """
        if not HAS_PYGAME:
            print("  pygame not installed. Run: pip install pygame")
            return

        laps = session.laps
        results = session.results
        drivers = results.sort_values("Position")["Abbreviation"].tolist()
        total_laps = int(laps["LapNumber"].max())
        event_name = session.event["EventName"]
        year = session.event.year

        # Detect if this is a sprint race (top 8 score points instead of top 10)
        session_name = session.name if hasattr(session, "name") else ""
        is_sprint = "sprint" in session_name.lower() if session_name else total_laps < 25
        points_cutoff = 8 if is_sprint else 10
        session_label = "SPRINT" if is_sprint else "RACE"

        print(f"  {event_name} {year} ({session_label}) — {len(drivers)} drivers, "
              f"{total_laps} laps, top {points_cutoff} score points")

        # ── Track rotation ──
        try:
            angle = session.get_circuit_info().rotation / 180 * np.pi
        except Exception:
            angle = 0

        # ── Track outline ──
        fastest = laps.pick_fastest()
        track_pos = fastest.get_pos_data()
        track_xy = _rotate(
            np.column_stack([track_pos["X"].values, track_pos["Y"].values]),
            angle=angle,
        )

        # ── Load driver data ──
        print("  Loading driver positions...")
        driver_data = {}
        driver_colors = {}
        driver_teams = {}

        for driver in drivers:
            d_laps = laps[laps["Driver"] == driver]
            if d_laps.empty:
                continue
            try:
                pos = d_laps.get_pos_data()
                if pos is None or pos.empty:
                    continue
                coords = _rotate(
                    np.column_stack([pos["X"].values, pos["Y"].values]),
                    angle=angle,
                )
                time_vals = pos["SessionTime"].dt.total_seconds().values \
                    if "SessionTime" in pos.columns else np.arange(len(coords))

                driver_data[driver] = {"x": coords[:, 0], "y": coords[:, 1], "time": time_vals}
            except Exception:
                continue

            try:
                hex_c = fastf1.plotting.get_driver_color(driver, session)
                driver_colors[driver] = _hex_to_rgb(hex_c)
            except Exception:
                driver_colors[driver] = (170, 170, 170)

            tm = results[results["Abbreviation"] == driver]
            driver_teams[driver] = tm.iloc[0]["TeamName"] if not tm.empty else ""

        active_drivers = list(driver_data.keys())
        n_drivers = len(active_drivers)
        print(f"  {n_drivers} drivers loaded")

        # ── Build actual DNF lookup from race results ──
        # This distinguishes real DNFs from drivers whose data simply ends at race finish
        actual_dnf_drivers = set()
        driver_final_status = {}
        for _, row in results.iterrows():
            drv = row["Abbreviation"]
            status = str(row.get("Status", ""))
            driver_final_status[drv] = status
            if status != "Finished" and "Lap" not in status:
                # "Finished" = completed race, "+1 Lap" etc = finished but lapped
                # Anything else (Engine, Accident, etc) = DNF
                actual_dnf_drivers.add(drv)

        # Track which drivers have crossed the finish line on the final lap
        # We'll populate this during the animation
        finished_drivers = set()  # drivers who have completed the race

        # ── Time base ──
        all_times = np.concatenate([d["time"] for d in driver_data.values()])
        t_min, t_max = all_times.min(), all_times.max()
        race_duration = t_max - t_min

        # ── Pre-interpolate positions at fine time resolution ──
        # Pre-compute at 10Hz (more than enough for smooth animation)
        sample_rate = 10
        n_samples = int(race_duration * sample_rate)
        sample_times = np.linspace(t_min, t_max, n_samples)

        print("  Interpolating positions...")
        for driver in driver_data:
            d = driver_data[driver]
            d_tmin, d_tmax = d["time"].min(), d["time"].max()
            d["x_s"] = np.interp(sample_times, d["time"], d["x"])
            d["y_s"] = np.interp(sample_times, d["time"], d["y"])
            d["active_s"] = (sample_times >= d_tmin) & (sample_times <= d_tmax)

        # ── Race position lookup ──
        print("  Building position lookup...")
        driver_position_samples = {}
        for driver in active_drivers:
            d_laps_sorted = laps[laps["Driver"] == driver].sort_values("LapNumber")
            time_pos = []
            for _, lr in d_laps_sorted.iterrows():
                pv = lr.get("Position")
                ls = lr.get("LapStartTime")
                lt = lr.get("LapTime")
                if pd.notna(pv) and pd.notna(ls):
                    t = (ls + lt).total_seconds() if pd.notna(lt) else ls.total_seconds()
                    time_pos.append((t, int(pv)))

            if not time_pos:
                grid = results[results["Abbreviation"] == driver]
                gp = int(grid.iloc[0]["GridPosition"]) if not grid.empty else 20
                driver_position_samples[driver] = np.full(n_samples, gp)
                continue

            time_pos.sort()
            t_vals = np.array([tp[0] for tp in time_pos])
            p_vals = np.array([tp[1] for tp in time_pos], dtype=float)

            pos_interp = np.full(n_samples, p_vals[0])
            for i, t in enumerate(sample_times):
                mask = t_vals <= t
                if mask.any():
                    pos_interp[i] = p_vals[mask][-1]
            driver_position_samples[driver] = pos_interp

        # ── Leader lap lookup ──
        leader_laps = []
        for driver in active_drivers:
            for _, lr in laps[laps["Driver"] == driver].sort_values("LapNumber").iterrows():
                if pd.notna(lr.get("LapStartTime")):
                    leader_laps.append((lr["LapStartTime"].total_seconds(), int(lr["LapNumber"])))
        leader_laps.sort()

        def get_leader_lap(t):
            best = 1
            for lt, ln in leader_laps:
                if lt <= t:
                    best = max(best, ln)
            return best

        # ──────────────────────────────────
        # PYGAME SETUP — Render at 2x for anti-aliasing
        # ──────────────────────────────────
        DISPLAY_W, DISPLAY_H = 1600, 900
        SSAA = 2  # Supersampling factor
        WIDTH, HEIGHT = DISPLAY_W * SSAA, DISPLAY_H * SSAA
        BOARD_W = 420 * SSAA
        TRACK_W = WIDTH - BOARD_W

        pygame.init()
        display = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.HWSURFACE | pygame.DOUBLEBUF)
        screen = pygame.Surface((WIDTH, HEIGHT))  # Internal hi-res surface
        pygame.display.set_caption(f"F1 Race Replay — {event_name} {year}")
        clock = pygame.time.Clock()

        # Fonts (scaled up for hi-res surface)
        font_big = pygame.font.SysFont("consolas", 32 * SSAA, bold=True)
        font_med = pygame.font.SysFont("consolas", 18 * SSAA, bold=True)
        font_sm = pygame.font.SysFont("consolas", 14 * SSAA)
        font_xs = pygame.font.SysFont("consolas", 12 * SSAA)
        font_label = pygame.font.SysFont("consolas", 11 * SSAA, bold=True)

        # ── Normalize track coordinates to fit screen ──
        tx, ty = track_xy[:, 0], track_xy[:, 1]
        margin = 100  # Extra margin so labels don't clip
        header_h = 90  # Space reserved for title + lap counter
        x_range = tx.max() - tx.min()
        y_range = ty.max() - ty.min()
        avail_w = TRACK_W - 2 * margin
        avail_h = HEIGHT - header_h - 2 * margin
        scale = min(avail_w / x_range, avail_h / y_range) * 0.9  # 90% fill for breathing room
        cx = margin + avail_w / 2
        cy = header_h + margin + avail_h / 2

        def world_to_screen(wx, wy):
            sx = cx + (wx - (tx.min() + tx.max()) / 2) * scale
            sy = cy - (wy - (ty.min() + ty.max()) / 2) * scale  # Flip Y
            return int(sx), int(sy)

        # Pre-compute track screen points
        track_screen = [world_to_screen(tx[i], ty[i]) for i in range(len(tx))]

        # Pre-compute driver screen positions for all samples
        print("  Pre-computing screen positions...")
        for driver in driver_data:
            d = driver_data[driver]
            sx = cx + (d["x_s"] - (tx.min() + tx.max()) / 2) * scale
            sy = cy - (d["y_s"] - (ty.min() + ty.max()) / 2) * scale
            d["sx"] = sx.astype(int)
            d["sy"] = sy.astype(int)

        # ──────────────────────────────────
        # MAIN LOOP
        # ──────────────────────────────────
        print(f"\n  ▶ Playing at {speed}x speed ({fps}fps)")
        print("  SPACE=Pause  UP/DOWN=Speed  Q=Quit\n")

        current_time = t_min
        paused = False
        running = True

        while running:
            dt = clock.tick(fps) / 1000.0  # Seconds since last frame

            # ── Events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        speed = min(speed * 1.5, 200)
                        print(f"  Speed: {speed:.0f}x")
                    elif event.key == pygame.K_DOWN:
                        speed = max(speed / 1.5, 1)
                        print(f"  Speed: {speed:.0f}x")

            # ── Advance time ──
            if not paused:
                current_time += dt * speed
                if current_time >= t_max:
                    current_time = t_max
                    paused = True  # Auto-pause at end

            # Find nearest sample index
            sample_idx = int((current_time - t_min) / race_duration * (n_samples - 1))
            sample_idx = max(0, min(sample_idx, n_samples - 1))

            elapsed = current_time - t_min
            current_lap = min(get_leader_lap(current_time), total_laps)

            # ── Get sorted positions ──
            # Determine race-end state: once leader hits final lap, race is ending
            race_finished = current_lap >= total_laps and current_time >= t_max - 5

            pos_list = []
            for driver in active_drivers:
                d = driver_data[driver]
                is_dnf = driver in actual_dnf_drivers

                if race_finished and not is_dnf:
                    # Race is over — show all non-DNF drivers as finished
                    # Use their final race position from results
                    finished_drivers.add(driver)
                    final_pos = driver_position_samples[driver][-1]
                    pos_list.append((driver, int(final_pos), "finished"))
                elif is_dnf and not d["active_s"][sample_idx]:
                    # Actual DNF — driver retired
                    pos_list.append((driver, 999, "dnf"))
                elif d["active_s"][sample_idx]:
                    # Still racing
                    pos = driver_position_samples[driver][sample_idx]
                    # Check if this driver just finished (on the last lap)
                    if current_lap >= total_laps and driver not in actual_dnf_drivers:
                        # Check if their data is about to end (within last few samples)
                        d_tmax = d["time"].max()
                        if current_time >= d_tmax - 2:
                            finished_drivers.add(driver)
                            pos_list.append((driver, int(pos), "finished"))
                        else:
                            pos_list.append((driver, int(pos), "racing"))
                    else:
                        pos_list.append((driver, int(pos), "racing"))
                elif driver in finished_drivers:
                    # Already crossed the line earlier
                    final_pos = driver_position_samples[driver][-1]
                    pos_list.append((driver, int(final_pos), "finished"))
                else:
                    # Data ended but not a known DNF — treat as finished
                    if driver not in actual_dnf_drivers:
                        finished_drivers.add(driver)
                        final_pos = driver_position_samples[driver][-1]
                        pos_list.append((driver, int(final_pos), "finished"))
                    else:
                        pos_list.append((driver, 999, "dnf"))

            # Sort: racing/finished by position, DNFs at bottom
            active_list = sorted([p for p in pos_list if p[2] != "dnf"], key=lambda x: x[1])
            dnf_list = [p for p in pos_list if p[2] == "dnf"]
            sorted_positions = active_list + dnf_list

            # ══════════════════════════════
            # DRAW
            # ══════════════════════════════
            screen.fill(BG)

            # ── Track ──
            if len(track_screen) > 2:
                # White border (outer edge)
                pygame.draw.lines(screen, TRACK_EDGE, True, track_screen, 14 * SSAA)
                # Dark track surface (inner)
                pygame.draw.lines(screen, TRACK_FILL, True, track_screen, 8 * SSAA)

            # Start/finish
            if track_screen:
                pygame.draw.circle(screen, F1_RED, track_screen[0], 6 * SSAA)

            # ── Drivers on track ──
            for driver in active_drivers:
                d = driver_data[driver]
                is_dnf = driver in actual_dnf_drivers

                # Show driver if they're actively racing OR have finished (not DNF)
                if d["active_s"][sample_idx]:
                    sx = int(d["sx"][sample_idx])
                    sy = int(d["sy"][sample_idx])
                elif driver in finished_drivers and not is_dnf:
                    # Show at their last known position
                    last_valid = np.where(d["active_s"])[0]
                    if len(last_valid) > 0:
                        li = last_valid[-1]
                        sx = int(d["sx"][li])
                        sy = int(d["sy"][li])
                    else:
                        continue
                else:
                    continue  # DNF — hide from track
                color = driver_colors.get(driver, (170, 170, 170))

                # Glow
                glow_r = 15 * SSAA
                glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color, 60), (glow_r, glow_r), glow_r)
                screen.blit(glow_surf, (sx - glow_r, sy - glow_r))

                # Dot
                pygame.draw.circle(screen, color, (sx, sy), 6 * SSAA)
                pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 6 * SSAA, max(1, SSAA))

                # Label
                label = font_label.render(driver, True, (255, 255, 255))
                lw, lh = label.get_size()
                # Background pill
                pad = 4 * SSAA
                pill = pygame.Surface((lw + 2 * pad, lh + pad), pygame.SRCALPHA)
                pygame.draw.rect(pill, (*color, 200), (0, 0, lw + 2 * pad, lh + pad),
                                border_radius=4 * SSAA)
                screen.blit(pill, (sx - lw // 2 - pad, sy - lh - 14 * SSAA))
                screen.blit(label, (sx - lw // 2, sy - lh - 12 * SSAA))

            # ── Header ──
            # Event name
            title_surf = font_med.render(f"{event_name} {year}", True, TEXT_WHITE)
            screen.blit(title_surf, (TRACK_W // 2 - title_surf.get_width() // 2, 12 * SSAA))

            # Lap counter
            if race_finished:
                lap_str = "CHEQUERED FLAG"
                lap_surf = font_big.render(lap_str, True, TEXT_WHITE)
            else:
                lap_str = f"LAP {current_lap}/{total_laps}"
                lap_surf = font_big.render(lap_str, True, F1_RED)
            screen.blit(lap_surf, (TRACK_W // 2 - lap_surf.get_width() // 2, 40 * SSAA))

            # Timer + speed
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            timer_str = f"{mins:02d}:{secs:02d}  {speed:.0f}x"
            timer_surf = font_sm.render(timer_str, True, TEXT_DIM)
            screen.blit(timer_surf, (TRACK_W - timer_surf.get_width() - 20 * SSAA, 16 * SSAA))

            # Pause indicator
            if paused:
                pause_surf = font_med.render("PAUSED", True, GOLD)
                screen.blit(pause_surf, (TRACK_W - pause_surf.get_width() - 20 * SSAA, 44 * SSAA))

            # ── Leaderboard panel ──
            board_rect = pygame.Rect(TRACK_W, 0, BOARD_W, HEIGHT)
            pygame.draw.rect(screen, BOARD_BG, board_rect)

            # Board title
            board_title = font_med.render("LIVE STANDINGS", True, F1_RED)
            screen.blit(board_title, (TRACK_W + BOARD_W // 2 - board_title.get_width() // 2, 16 * SSAA))

            # Column headers
            hdr_y = 50 * SSAA
            for text, x_off in [("POS", 15), ("", 55), ("DRIVER", 65), ("TEAM", 180)]:
                h = font_xs.render(text, True, TEXT_DARK)
                screen.blit(h, (TRACK_W + x_off * SSAA, hdr_y))

            # Separator
            pygame.draw.line(screen, BOARD_LINE,
                           (TRACK_W + 10 * SSAA, hdr_y + 20 * SSAA),
                           (WIDTH - 10 * SSAA, hdr_y + 20 * SSAA), SSAA)

            # Driver rows
            row_h = 34 * SSAA
            start_y = hdr_y + 28 * SSAA
            for i, (driver, race_pos, status) in enumerate(sorted_positions):
                ry = start_y + i * row_h
                if ry > HEIGHT - 20 * SSAA:
                    break

                color = driver_colors.get(driver, (170, 170, 170))
                team = driver_teams.get(driver, "")
                is_active = status != "dnf"
                has_finished = status == "finished"

                # Alternate row background (brighter in points zone)
                if i % 2 == 0:
                    bg_alpha = 80 if is_active and i < points_cutoff else 40
                    row_bg = pygame.Surface((BOARD_W - 20 * SSAA, row_h - 2 * SSAA), pygame.SRCALPHA)
                    row_bg.fill((*BOARD_LINE, bg_alpha))
                    screen.blit(row_bg, (TRACK_W + 10 * SSAA, ry))

                # Color bar (dimmed outside points zone)
                bar_color = color if is_active and i < points_cutoff else _darken(color, 0.4)
                pygame.draw.rect(screen, bar_color,
                               (TRACK_W + 12 * SSAA, ry + 4 * SSAA, 4 * SSAA, row_h - 10 * SSAA),
                               border_radius=2 * SSAA)

                if status == "dnf":
                    # ── DNF: red X, dimmed name, red DNF tag ──
                    x_surf = font_med.render(" X", True, F1_RED)
                    screen.blit(x_surf, (TRACK_W + 22 * SSAA, ry + 6 * SSAA))
                    name_surf = font_med.render(driver, True, (65, 65, 65))
                    screen.blit(name_surf, (TRACK_W + 65 * SSAA, ry + 6 * SSAA))
                    dnf_surf = font_sm.render("DNF", True, F1_RED)
                    screen.blit(dnf_surf, (TRACK_W + 180 * SSAA, ry + 8 * SSAA))

                else:
                    board_position = i + 1

                    # ── Position number color: gold/silver/bronze/points/out ──
                    if board_position == 1:
                        pos_color = GOLD
                    elif board_position == 2:
                        pos_color = SILVER
                    elif board_position == 3:
                        pos_color = BRONZE
                    elif board_position <= points_cutoff:
                        pos_color = POINTS_ZONE
                    else:
                        pos_color = TEXT_OUT_OF_POINTS

                    pos_surf = font_med.render(f"{board_position:2d}", True, pos_color)
                    screen.blit(pos_surf, (TRACK_W + 22 * SSAA, ry + 6 * SSAA))

                    # ── Driver name: bright in points, dimmed outside ──
                    if board_position <= points_cutoff:
                        name_color = color
                    else:
                        name_color = _darken(color, 0.55)
                    name_surf = font_med.render(driver, True, name_color)
                    screen.blit(name_surf, (TRACK_W + 65 * SSAA, ry + 6 * SSAA))

                    # ── Checkered flag for finished drivers ──
                    if has_finished:
                        flag_surf = font_med.render(" F", True, TEXT_WHITE)
                        name_w = name_surf.get_width()
                        # Draw a tiny checkered pattern instead of just "F"
                        fx = TRACK_W + 65 * SSAA + name_w + 6 * SSAA
                        fy = ry + 6 * SSAA
                        sq = 4 * SSAA  # Square size
                        for row_idx in range(3):
                            for col_idx in range(3):
                                sq_color = TEXT_WHITE if (row_idx + col_idx) % 2 == 0 else BG
                                pygame.draw.rect(screen, sq_color,
                                               (fx + col_idx * sq, fy + row_idx * sq, sq, sq))

                    # ── Team name: dimmed more for non-points ──
                    team_short = team[:16] if len(team) > 16 else team
                    team_color = TEXT_DIM if board_position <= points_cutoff else TEXT_DARK
                    team_surf = font_xs.render(team_short, True, team_color)
                    screen.blit(team_surf, (TRACK_W + 180 * SSAA, ry + 10 * SSAA))

                    # ── Points zone separator line ──
                    if board_position == points_cutoff:
                        sep_y = ry + row_h - 1 * SSAA
                        pygame.draw.line(screen, F1_RED,
                                       (TRACK_W + 10 * SSAA, sep_y),
                                       (WIDTH - 10 * SSAA, sep_y), SSAA)

            # ── Controls hint ──
            hint = font_xs.render("SPACE=Pause  ↑↓=Speed  Q=Quit", True, TEXT_DARK)
            screen.blit(hint, (TRACK_W + BOARD_W // 2 - hint.get_width() // 2, HEIGHT - 24 * SSAA))

            # ── Downscale and flip ──
            pygame.transform.smoothscale(screen, (DISPLAY_W, DISPLAY_H), display)
            pygame.display.flip()

        pygame.quit()
        print("  Animation closed.")


if __name__ == "__main__":
    session = fastf1.get_session(2025, 1, "R")
    session.load()
    TrackAnimation().animate_race(session, speed=20)
