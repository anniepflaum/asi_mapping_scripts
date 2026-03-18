#!/usr/bin/env python3
"""Shared NSROC trajectory CSV helpers."""

import csv
import re

import numpy as np
from apexpy import Apex

from asi_time_utils import hhmmss_fractional_to_seconds


apex = Apex()
TRAJ_T0_RE = re.compile(r"^T0:\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s+UTC$")


def get_launch_start_from_traj_csv(filename):
    """Read the T0 launch time from an NSROC attitude-solution CSV as HHMMSS(.fraction)."""
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Trajectory input must be an NSROC attitude-solution CSV: {filename}")
    with open(filename, "r", encoding="utf-8") as fd:
        for line in fd:
            match = TRAJ_T0_RE.match(line.strip())
            if match:
                launch_start = f"{match.group(4)}{match.group(5)}{match.group(6)}"
                if match.group(7):
                    launch_start = f"{launch_start}.{match.group(7)}"
                return launch_start
    return None


def format_time_since_launch(map_time, launch_start):
    """Format T+ seconds relative to launch for a requested map time."""
    if map_time is None or launch_start is None:
        return None
    rel_sec = hhmmss_fractional_to_seconds(map_time) - hhmmss_fractional_to_seconds(launch_start)
    return f"T{rel_sec:+.1f} s"


def load_traj(filename, map_time=None):
    """
    Load rocket trajectory from an NSROC attitude-solution CSV.
    Map lat/lon to 110 km altitude and optionally return the nearest map-time point.
    """
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Trajectory input must be an NSROC attitude-solution CSV: {filename}")
    launch_start = get_launch_start_from_traj_csv(filename)
    with open(filename, "r", encoding="utf-8") as fd:
        lines = fd.readlines()

    header_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Time,") and "Latgd" in stripped and "Long" in stripped and "Alt" in stripped:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not find trajectory header in {filename}")

    reader = csv.reader(lines[header_idx:])
    header = next(reader, None)
    next(reader, None)
    if header is None:
        raise ValueError(f"Could not read trajectory header in {filename}")
    columns = [column.strip() for column in header]

    rows = []
    for values in reader:
        if not values or not any(value.strip() for value in values):
            continue
        row = {column: value.strip() for column, value in zip(columns, values)}
        if not row.get("Time"):
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No trajectory samples found in {filename}")

    times = np.array([float(row["Time"]) for row in rows], dtype=float)
    lats = np.array([float(row["Latgd"]) for row in rows], dtype=float)
    lons = np.array([float(row["Long"]) for row in rows], dtype=float)
    alts = np.array([float(row["Alt"]) for row in rows], dtype=float) / 1000.0

    lats, lons, _ = apex.map_to_height(lats, lons, alts, 110.0)
    idx = np.argwhere(np.isclose(times % 60, 0.0, atol=0.05))
    latsm = lats[idx].squeeze()
    lonsm = lons[idx].squeeze()
    aidx = np.argmax(alts)
    lata = lats[aidx]
    lona = lons[aidx]

    traj_lat_at_map = None
    traj_lon_at_map = None
    if map_time is not None and launch_start is not None:
        map_sec = hhmmss_fractional_to_seconds(map_time)
        launch_sec = hhmmss_fractional_to_seconds(launch_start)
        rel_sec = map_sec - launch_sec
        if rel_sec >= 0 and rel_sec <= times[-1]:
            traj_time_idx = np.argmin(np.abs(times - rel_sec))
            traj_lat_at_map = lats[traj_time_idx]
            traj_lon_at_map = lons[traj_time_idx]

    return lats, lons, latsm, lonsm, lata, lona, traj_lat_at_map, traj_lon_at_map


def load_traj_times(filename):
    """Read the raw trajectory time column from an NSROC attitude-solution CSV."""
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Trajectory input must be an NSROC attitude-solution CSV: {filename}")
    with open(filename, "r", encoding="utf-8") as fd:
        lines = fd.readlines()

    header_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Time,") and "Latgd" in stripped and "Long" in stripped and "Alt" in stripped:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not find trajectory header in {filename}")

    reader = csv.reader(lines[header_idx:])
    next(reader, None)
    next(reader, None)
    rows = []
    for values in reader:
        if not values or not any(value.strip() for value in values):
            continue
        rows.append(values)
    return np.array([float(row[0].strip()) for row in rows], dtype=float)


def build_traj_lookup(traj_path):
    """Load and cache one trajectory for repeated nearest-time lookup."""
    lats, lons, _latm, _lonm, _lata, _lona, _lat_map, _lon_map = load_traj(traj_path)
    launch_start = get_launch_start_from_traj_csv(traj_path)
    return {
        "times": load_traj_times(traj_path),
        "lats": np.asarray(lats, dtype=float),
        "lons": np.asarray(lons, dtype=float),
        "launch_sec": hhmmss_fractional_to_seconds(launch_start) if launch_start else None,
        "path": traj_path,
    }


def lookup_traj_position(traj_lookup, map_time):
    """Return the mapped lat/lon trajectory point nearest a requested map time."""
    launch_sec = traj_lookup["launch_sec"]
    if map_time is None or launch_sec is None:
        return None, None
    rel_sec = hhmmss_fractional_to_seconds(map_time) - launch_sec
    times = traj_lookup["times"]
    if times.size == 0 or rel_sec < 0 or rel_sec > times[-1]:
        return None, None
    idx = int(np.argmin(np.abs(times - rel_sec)))
    return float(traj_lookup["lats"][idx]), float(traj_lookup["lons"][idx])


def resample_traj_by_distance(traj_lookup, n_samples):
    """Resample a mapped trajectory to equal cumulative-distance intervals."""
    lats = np.asarray(traj_lookup["lats"], dtype=float)
    lons = np.asarray(traj_lookup["lons"], dtype=float)
    if lats.size == 0 or lons.size == 0 or lats.size != lons.size:
        raise ValueError("trajectory lookup has invalid coordinates")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    mean_lat = np.deg2rad(np.nanmean(lats))
    x = lons * np.cos(mean_lat)
    y = lats
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    if s[-1] <= 0:
        lat_resampled = np.full(n_samples, lats[0], dtype=float)
        lon_resampled = np.full(n_samples, lons[0], dtype=float)
        return lat_resampled, lon_resampled, np.linspace(0.0, 1.0, n_samples)

    target_s = np.linspace(0.0, s[-1], n_samples)
    lat_resampled = np.interp(target_s, s, lats)
    lon_resampled = np.interp(target_s, s, lons)
    return lat_resampled, lon_resampled, target_s / s[-1]
