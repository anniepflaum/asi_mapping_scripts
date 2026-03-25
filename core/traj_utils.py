#!/usr/bin/env python3
"""Shared GPS-export trajectory CSV helpers."""

import csv
import re

import numpy as np
from apexpy import Apex

from core.time_utils import hhmmss_fractional_to_seconds


apex = Apex()
GPS_TIME_RE = re.compile(r"^\d{3}\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$")
UTC_TIME_COLUMN = "Time (UTC / From GPS Receiver)"
FLIGHT_TIME_COLUMN = "Flight Time (Official T0)"
LAT_COLUMN = "Latitude"
LON_COLUMN = "Longitude"
ALT_COLUMN = "Altitude (km)"


def parse_gps_utc_time(value):
    """Parse GPS-export UTC time token 'DDD HH:MM:SS.sss' into seconds since midnight."""
    m = GPS_TIME_RE.fullmatch(str(value).strip())
    if not m:
        raise ValueError(f"Invalid GPS UTC time: {value}")
    hour = int(m.group(1))
    minute = int(m.group(2))
    second = int(m.group(3))
    frac = m.group(4) or ""
    microsecond = int(frac.ljust(6, "0")) if frac else 0
    return hour * 3600.0 + minute * 60.0 + second + microsecond / 1e6


def format_seconds_of_day(seconds):
    """Format seconds since midnight as HHMMSS(.fraction)."""
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    whole_seconds = int(seconds)
    frac = seconds - whole_seconds
    frac_str = f"{frac:.6f}".split(".")[1].rstrip("0")
    base = f"{hours:02d}{minutes:02d}{whole_seconds:02d}"
    return f"{base}.{frac_str}" if frac_str else base


def load_traj_records(filename):
    """Load a GPS trajectory export and return UTC time, flight time, and position arrays."""
    with open(filename, "r", encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd)
        rows = [row for row in reader if row.get(UTC_TIME_COLUMN)]
    if not rows:
        raise ValueError(f"No trajectory samples found in {filename}")

    utc_times = np.array([parse_gps_utc_time(row[UTC_TIME_COLUMN]) for row in rows], dtype=float)
    flight_times = np.array([float(row[FLIGHT_TIME_COLUMN]) for row in rows], dtype=float)
    lats = np.array([float(row[LAT_COLUMN]) for row in rows], dtype=float)
    lons = np.array([float(row[LON_COLUMN]) for row in rows], dtype=float)
    alts = np.array([float(row[ALT_COLUMN]) for row in rows], dtype=float)
    return utc_times, flight_times, lats, lons, alts


def get_launch_start_from_traj_csv(filename):
    """Estimate T0 launch time from GPS UTC and official flight-time columns."""
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Trajectory input must be a GPS export CSV: {filename}")
    utc_times, flight_times, _lats, _lons, _alts = load_traj_records(filename)
    launch_seconds = np.nanmedian(utc_times - flight_times)
    return format_seconds_of_day(float(launch_seconds))


def format_time_since_launch(map_time, launch_start):
    """Format T+ seconds relative to launch for a requested map time."""
    if map_time is None or launch_start is None:
        return None
    rel_sec = hhmmss_fractional_to_seconds(map_time) - hhmmss_fractional_to_seconds(launch_start)
    return f"T{rel_sec:+.1f} s"


def load_traj(filename, map_time=None):
    """
    Load rocket trajectory from a GPS export CSV.
    Map lat/lon to 110 km altitude and optionally return the nearest map-time point.
    """
    if not filename.lower().endswith(".csv"):
        raise ValueError(f"Trajectory input must be a GPS export CSV: {filename}")
    utc_times, flight_times, lats, lons, alts = load_traj_records(filename)

    lats, lons, _ = apex.map_to_height(lats, lons, alts, 110.0)
    idx = np.argwhere(np.isclose(flight_times % 60, 0.0, atol=0.05))
    latsm = lats[idx].squeeze()
    lonsm = lons[idx].squeeze()
    aidx = np.argmax(alts)
    lata = lats[aidx]
    lona = lons[aidx]

    traj_lat_at_map = None
    traj_lon_at_map = None
    if map_time is not None:
        map_sec = hhmmss_fractional_to_seconds(map_time)
        if map_sec >= utc_times[0] and map_sec <= utc_times[-1]:
            traj_time_idx = np.argmin(np.abs(utc_times - map_sec))
            traj_lat_at_map = lats[traj_time_idx]
            traj_lon_at_map = lons[traj_time_idx]

    return lats, lons, latsm, lonsm, lata, lona, traj_lat_at_map, traj_lon_at_map


def load_traj_times(filename):
    """Read the flight-time column from a GPS export CSV."""
    _utc_times, flight_times, _lats, _lons, _alts = load_traj_records(filename)
    return flight_times


def build_traj_lookup(traj_path):
    """Load and cache one trajectory for repeated nearest-time lookup."""
    lats, lons, _latm, _lonm, _lata, _lona, _lat_map, _lon_map = load_traj(traj_path)
    utc_times, flight_times, _raw_lats, _raw_lons, _raw_alts = load_traj_records(traj_path)
    launch_start = get_launch_start_from_traj_csv(traj_path)
    return {
        "times": flight_times,
        "utc_times": utc_times,
        "lats": np.asarray(lats, dtype=float),
        "lons": np.asarray(lons, dtype=float),
        "launch_sec": hhmmss_fractional_to_seconds(launch_start) if launch_start else None,
        "path": traj_path,
    }


def lookup_traj_position(traj_lookup, map_time):
    """Return the mapped lat/lon trajectory point nearest a requested map time."""
    if map_time is None:
        return None, None
    map_sec = hhmmss_fractional_to_seconds(map_time)
    utc_times = np.asarray(traj_lookup["utc_times"], dtype=float)
    if utc_times.size == 0 or map_sec < utc_times[0] or map_sec > utc_times[-1]:
        return None, None
    idx = int(np.argmin(np.abs(utc_times - map_sec)))
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


def resample_traj_by_time(traj_lookup, n_samples):
    """Resample a mapped trajectory to equal flight-time intervals."""
    times = np.asarray(traj_lookup["times"], dtype=float)
    lats = np.asarray(traj_lookup["lats"], dtype=float)
    lons = np.asarray(traj_lookup["lons"], dtype=float)
    if times.size == 0 or lats.size == 0 or lons.size == 0:
        raise ValueError("trajectory lookup has no samples")
    if not (times.size == lats.size == lons.size):
        raise ValueError("trajectory lookup arrays must be the same length")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    target_t = np.linspace(times[0], times[-1], n_samples)
    lat_resampled = np.interp(target_t, times, lats)
    lon_resampled = np.interp(target_t, times, lons)
    return lat_resampled, lon_resampled, target_t
