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
