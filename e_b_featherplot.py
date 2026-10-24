#!/usr/bin/env python3
"""Plot geographic GNEISS trajectories with east/north ExB drift vectors.

Vectors are linearly sampled at shared TG times, without smoothing or
extrapolation. The default window is 100..500 s after each nominal launch:
10:20:40..10:27:20 UTC for 397 and 10:21:10..10:27:50 UTC for 398.
The default vector interval is 10 s.
GPS positions use receiver UTC; ExB times use the T0 in each CSV header.
The trajectories are direct GPS positions at rocket altitude.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
import numpy as np

from core.missions import ROCKET_TIME_WINDOWS
from core.paths import (
    DATA_ROOT,
    GNEISS_LEFT_TRAJECTORY_PATH,
    GNEISS_RIGHT_TRAJECTORY_PATH,
    OUTPUT_ROOT,
)
from core.time_utils import hhmmss_fractional_to_seconds
from core.traj_utils import load_traj_records


TG_UTC = dt.datetime(2026, 2, 10, 10, 19, tzinfo=dt.timezone.utc)
TG_SECONDS_OF_DAY = 10 * 3600 + 19 * 60
ROCKET_COLORS = {"397": "tab:blue", "398": "tab:orange"}
EXB_DIR = DATA_ROOT / "raw" / "rocket" / "gneiss-stackplot"
EXB_PATHS = {
    "397": EXB_DIR / "exb_july30_397.csv",
    "398": EXB_DIR / "exb_oct6_398.csv",
}
TRAJECTORY_PATHS = {
    "397": GNEISS_LEFT_TRAJECTORY_PATH,
    "398": GNEISS_RIGHT_TRAJECTORY_PATH,
}
DEFAULT_OUTPUT = OUTPUT_ROOT / "asi-mapping" / "e_b_featherplot_20260210.png"
# GPS normally samples every 0.05 s, with observed gaps up to 2.1 s.
# ExB normally samples every 0.0004 s; skip arrows across its acquisition gaps.
MAX_GPS_GAP_S = 5.0
MAX_EXB_GAP_S = 0.1


def load_exb(path, rocket):
    """Return TG seconds and east/north m/s using the CSV's own UTC T0."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        header = stream.readline()
    match = re.search(
        r"\b(397|398)\s+t0\s*\(s\):\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
        header,
    )
    if match is None or match.group(1) != rocket:
        raise ValueError(f"Expected rocket {rocket} and UTC T0 in the header of {path}")
    columns = [part.strip() for part in header.split(",")]
    if columns[1:3] != ["ExB_east (m/s)", "ExB_north (m/s)"]:
        raise ValueError(f"Expected east and north ExB columns in m/s in {path}")
    time_text = match.group(2)
    t0 = dt.datetime.strptime(time_text, "%H:%M:%S.%f" if "." in time_text else "%H:%M:%S")
    offset_s = t0.hour * 3600 + t0.minute * 60 + t0.second + t0.microsecond / 1e6 - TG_SECONDS_OF_DAY
    data = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1, 2), ndmin=2)
    time_s = data[:, 0] + offset_s
    if time_s.size < 2 or not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError(f"Expected at least two strictly increasing finite timestamps in {path}")
    return time_s, data[:, 1], data[:, 2]


def load_trajectory(path, rocket):
    """Use geographic GPS positions and UTC, independently of Official T0."""
    utc, _flight, latitude, longitude, altitude = load_traj_records(path)
    time_s = utc - TG_SECONDS_OF_DAY
    start, end = (
        hhmmss_fractional_to_seconds(token) - TG_SECONDS_OF_DAY
        for token in ROCKET_TIME_WINDOWS[rocket]
    )
    in_flight = np.isfinite(time_s) & (time_s >= start) & (time_s <= end)
    time_s = time_s[in_flight]
    if time_s.size < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError(f"Expected increasing GPS UTC times covering rocket {rocket}'s flight in {path}")
    return {
        "time_s": time_s,
        "latitude_deg": latitude[in_flight],
        "longitude_deg": longitude[in_flight],
        "altitude_km": altitude[in_flight],
    }


def interpolate_samples(source_time, values, targets, max_gap_s):
    """Interpolate within valid brackets; retain gaps and reject extrapolation."""
    source_time = np.asarray(source_time, dtype=float)
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if source_time.ndim != 1 or source_time.size < 2 or values.shape != source_time.shape:
        raise ValueError("Interpolation requires matching one-dimensional time/value arrays")
    result = np.full(targets.shape, np.nan)
    right = np.searchsorted(source_time, targets, side="left")
    right_clipped = np.clip(right, 0, source_time.size - 1)
    exact = (right < source_time.size) & (source_time[right_clipped] == targets)
    result[exact] = values[right_clipped[exact]]
    inside = ~exact & (right > 0) & (right < source_time.size)
    indices = np.flatnonzero(inside)
    hi = right[indices]
    lo = hi - 1
    gaps = source_time[hi] - source_time[lo]
    valid = (gaps > 0) & (gaps <= max_gap_s) & np.isfinite(values[lo]) & np.isfinite(values[hi])
    indices, lo, hi, gaps = indices[valid], lo[valid], hi[valid], gaps[valid]
    fraction = (targets[indices] - source_time[lo]) / gaps
    result[indices] = values[lo] + fraction * (values[hi] - values[lo])
    return result


def velocity_to_degree_offsets(
    latitude_deg, east_m_s, north_m_s, scale_seconds, altitude_km=0.0
):
    """Convert horizontal velocity into a WGS84 map displacement for display.

    ``scale_seconds`` is only an arrow-length scale: 20 means 1000 m/s is
    drawn as 20 km. These arrows do not predict the rocket's displacement.
    """
    latitude = np.deg2rad(latitude_deg)
    semimajor_m = 6378137.0
    eccentricity_squared = 6.6943799901413165e-3
    denominator = 1.0 - eccentricity_squared * np.sin(latitude) ** 2
    altitude_m = np.asarray(altitude_km, dtype=float) * 1000.0
    prime_vertical_m = semimajor_m / np.sqrt(denominator) + altitude_m
    meridional_m = (
        semimajor_m * (1.0 - eccentricity_squared) / denominator**1.5
        + altitude_m
    )
    dlon = np.rad2deg(np.asarray(east_m_s) * scale_seconds / (prime_vertical_m * np.cos(latitude)))
    dlat = np.rad2deg(np.asarray(north_m_s) * scale_seconds / meridional_m)
    return dlon, dlat


def sample_vectors(trajectory, flow, targets):
    samples = {"time_s": targets}
    for name in ("latitude_deg", "longitude_deg", "altitude_km"):
        values = trajectory[name]
        if name == "longitude_deg":
            # Mission longitudes do not cross the seam, but keep interpolation circular.
            finite = np.isfinite(values)
            if np.all(finite):
                values = np.rad2deg(np.unwrap(np.deg2rad(values)))
        samples[name] = interpolate_samples(trajectory["time_s"], values, targets, MAX_GPS_GAP_S)
    flow_time, east, north = flow
    samples["east_m_s"] = interpolate_samples(flow_time, east, targets, MAX_EXB_GAP_S)
    samples["north_m_s"] = interpolate_samples(flow_time, north, targets, MAX_EXB_GAP_S)
    valid = np.logical_and.reduce([np.isfinite(value) for value in samples.values()])
    return {name: values[valid] for name, values in samples.items()}


def vector_time_window(rocket, start_s, end_s):
    """Convert a nominal-launch-relative window to the shared TG clock.

    Window references are 10:19:00 UTC for 397 and 10:19:30 UTC for 398.
    ExB samples retain the separate, precise T0 parsed from their CSV headers.
    """
    launch_offset = hhmmss_fractional_to_seconds(ROCKET_TIME_WINDOWS[rocket][0]) - TG_SECONDS_OF_DAY
    return start_s + launch_offset, end_s + launch_offset


def plot_feather(trajectories, samples, args):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 10), layout="compressed")
    windows = {
        rocket: vector_time_window(rocket, args.start, args.end)
        for rocket in trajectories
    }
    all_latitudes = np.concatenate([traj["latitude_deg"] for traj in trajectories.values()])
    reference_latitude = float(np.nanmean(all_latitudes))
    quiver = None
    for rocket, trajectory in trajectories.items():
        color = ROCKET_COLORS[rocket]
        ax.plot(
            trajectory["longitude_deg"], trajectory["latitude_deg"],
            color=color, linewidth=1.4, label=f"{rocket}", zorder=2,
        )
        vector = samples[rocket]
        dlon, dlat = velocity_to_degree_offsets(
            vector["latitude_deg"], vector["east_m_s"], vector["north_m_s"], args.vector_scale_seconds,
        )
        quiver = ax.quiver(
            vector["longitude_deg"], vector["latitude_deg"], dlon, dlat,
            angles="xy", scale_units="xy", scale=1, pivot="tail", color=color,
            width=0.120, headwidth=3.5, headlength=4.5, zorder=3,
        )
        ax.scatter(
            vector["longitude_deg"], vector["latitude_deg"], color=color,
            s=10, edgecolors=color, linewidths=0.3, zorder=4,
        )
        # Quiver autoscaling normally considers only its anchors, not arrow tips.
        ax.update_datalim(np.column_stack((vector["longitude_deg"] + dlon, vector["latitude_deg"] + dlat)))

    ax.autoscale_view()
    ax.margins(x=0.20, y=0.12)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(reference_latitude)))
    ax.set_xlabel("Geodetic longitude (degrees east)")
    ax.set_ylabel("Geodetic latitude (degrees north)")
    window_labels = [
        f"{rocket}: " + "–".join(
            (TG_UTC + dt.timedelta(seconds=float(time_s))).strftime("%H:%M:%S")
            for time_s in window
        ) + " UTC"
        for rocket, window in windows.items()
    ]
    ax.set_title(
        "GNEISS E × B drift along rocket trajectories\n"
        f"2026-02-10 · vectors every {args.step:g} s\n"
        + "\n".join(window_labels)
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    key_dlon, _ = velocity_to_degree_offsets(reference_latitude, 1000.0, 0.0, args.vector_scale_seconds)
    ax.quiverkey(quiver, 0.78, 0.06, float(key_dlon), "1000 m/s", labelpos="N", color="black")
    fig.supxlabel(
        "Direct GPS trajectories; dots and arrows match each rocket's trajectory color.\n"
        f"Arrow length: 1000 m/s = {args.vector_scale_seconds:g} km on the map.\n"
        f"397: {args.exb_397.name}  |  398: {args.exb_398.name}",
        fontsize=9,
    )
    return fig, ax


def write_samples(path, samples):
    columns = ["rocket", "utc", "time_since_tg_s", "latitude_deg", "longitude_deg", "altitude_km", "east_m_s", "north_m_s", "horizontal_speed_m_s"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for rocket, vector in samples.items():
            for index, time_s in enumerate(vector["time_s"]):
                east, north = vector["east_m_s"][index], vector["north_m_s"][index]
                writer.writerow([
                    rocket, (TG_UTC + dt.timedelta(seconds=float(time_s))).isoformat(), time_s,
                    vector["latitude_deg"][index], vector["longitude_deg"][index],
                    vector["altitude_km"][index], east, north, np.hypot(east, north),
                ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=100.0, help="First vector time in s after each nominal launch: 397 10:19:00, 398 10:19:30 UTC (default: 100)")
    parser.add_argument("--end", type=float, default=500.0, help="Last vector time in s after each nominal launch (default: 500)")
    parser.add_argument("--step", type=float, default=10.0, help="Vector interval in s, anchored to shared TG (default: 10)")
    parser.add_argument("--vector-scale-seconds", type=float, default=80.0, help="Arrow display scale: 1000 m/s spans this many km (default: 80)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Figure path; sampled vectors also saved as a CSV beside it")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively")
    for rocket in ROCKET_COLORS:
        parser.add_argument(f"--exb-{rocket}", type=Path, default=EXB_PATHS[rocket])
        parser.add_argument(f"--trajectory-{rocket}", type=Path, default=TRAJECTORY_PATHS[rocket])
    args = parser.parse_args(argv)
    if not all(np.isfinite(value) for value in (args.start, args.end, args.step, args.vector_scale_seconds)):
        parser.error("Time limits, interval, and vector scale must be finite")
    if args.start >= args.end or args.step <= 0 or args.vector_scale_seconds <= 0:
        parser.error("Require start < end, a positive interval, and a positive vector scale")
    if args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("Output must have a .png, .pdf, or .svg extension")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    trajectories, samples = {}, {}
    for rocket in ROCKET_COLORS:
        start_s, end_s = vector_time_window(rocket, args.start, args.end)
        targets = np.arange(np.ceil(start_s / args.step), np.floor(end_s / args.step) + 1) * args.step
        if targets.size == 0:
            raise ValueError(f"No vector times fall inside rocket {rocket}'s requested window")
        print(f"Loading rocket {rocket} GPS and ExB...", flush=True)
        trajectories[rocket] = load_trajectory(getattr(args, f"trajectory_{rocket}"), rocket)
        flow = load_exb(getattr(args, f"exb_{rocket}"), rocket)
        samples[rocket] = sample_vectors(trajectories[rocket], flow, targets)
        count = len(samples[rocket]["time_s"])
        if count == 0:
            raise ValueError(f"No valid GPS/ExB vector samples for rocket {rocket} in the requested window")
        maximum = np.max(np.hypot(samples[rocket]["east_m_s"], samples[rocket]["north_m_s"]))
        print(f"  TG+{start_s:g}–{end_s:g} s: {count}/{targets.size} vectors; "
              f"maximum sampled horizontal speed {maximum:.1f} m/s", flush=True)
    fig, _ax = plot_feather(trajectories, samples, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    csv_path = args.output.with_suffix(".csv")
    write_samples(csv_path, samples)
    print(f"Saved {args.output}\nSaved {csv_path}")
    import matplotlib.pyplot as plt
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
