#!/usr/bin/env python3
"""Shared GPS-export trajectory helpers for CSV and XLSX mission products."""

import csv
import datetime as dt
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from apexpy import Apex

from core.paths import (
    GIRAFF_LEFT_TRAJECTORY_PATH,
    GIRAFF_RIGHT_TRAJECTORY_PATH,
    GNEISS_LEFT_TRAJECTORY_PATH,
    GNEISS_RIGHT_TRAJECTORY_PATH,
    MISSION_TRAJECTORY_PATHS,
)
from core.time_utils import hhmmss_fractional_to_seconds


apex = Apex()
GPS_TIME_RE = re.compile(r"^\d{3}\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$")
UTC_TIME_COLUMN = "Time (UTC / From GPS Receiver)"
FLIGHT_TIME_COLUMN = "Flight Time (Official T0)"
LAT_COLUMN = "Latitude"
LON_COLUMN = "Longitude"
ALT_COLUMN = "Altitude (km)"
XLSX_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
XLSX_FLIGHT_TIME_COLUMN = "Flight Time"
XLSX_LAT_COLUMN = "Lat"
XLSX_LON_COLUMN = "Long"
XLSX_ALT_COLUMN = "Alt"
XLSX_GPS_MSEC_COLUMN = "GPS Time (mSec of week)"
XLSX_GPS_WEEK_COLUMN = "GPS Week"
GIRAFF_LAUNCH_DATETIME = dt.datetime(2025, 2, 2)


def mapped_apex_height(color="green"):
    """Return the target apex mapping height for a given ASI color."""
    return 200.0 if str(color).lower() == "red" else 110.0


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


def mission_trajectory_paths(mission):
    mission_key = str(mission).upper()
    if mission_key not in MISSION_TRAJECTORY_PATHS:
        raise ValueError(f"Unsupported mission: {mission}")
    return MISSION_TRAJECTORY_PATHS[mission_key]


def trajectory_display_labels(mission):
    mission_key = str(mission).upper()
    if mission_key == "GIRAFF":
        return {"left": "36381 Main", "right": "36381 Sub", "left_tag": "Main", "right_tag": "Sub"}
    return {"left": "36.397", "right": "36.398", "left_tag": "397", "right_tag": "398"}


def load_traj_records(filename):
    """Load a GPS trajectory export and return UTC time, flight time, and position arrays."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        return load_traj_records_xlsx(filename)

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


def xlsx_cell_ref_to_index(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def load_xlsx_shared_strings(xlsx_path):
    try:
        with ZipFile(xlsx_path) as zf, zf.open("xl/sharedStrings.xml") as fd:
            root = ET.parse(fd).getroot()
    except KeyError:
        return []

    strings = []
    for si in root.findall("main:si", XLSX_NS):
        text_parts = [node.text or "" for node in si.findall(".//main:t", XLSX_NS)]
        strings.append("".join(text_parts))
    return strings


def iter_xlsx_sheet_rows(xlsx_path, sheet_name="xl/worksheets/sheet1.xml"):
    shared_strings = load_xlsx_shared_strings(xlsx_path)
    with ZipFile(xlsx_path) as zf, zf.open(sheet_name) as fd:
        root = ET.parse(fd).getroot()

    for row in root.findall(".//main:sheetData/main:row", XLSX_NS):
        values = {}
        for cell in row.findall("main:c", XLSX_NS):
            ref = cell.get("r", "")
            idx = xlsx_cell_ref_to_index(ref)
            cell_type = cell.get("t")
            value_node = cell.find("main:v", XLSX_NS)
            value = value_node.text if value_node is not None else ""
            if cell_type == "s" and value:
                value = shared_strings[int(value)]
            values[idx] = value
        if values:
            width = max(values) + 1
            yield [values.get(i, "") for i in range(width)]


def gps_msec_of_week_to_seconds_of_day(value):
    milliseconds = float(value)
    seconds_of_week = milliseconds / 1000.0
    return float(seconds_of_week % 86400.0)


def parse_giraff_sample_datetime(gps_msec_value):
    """
    Hardwire GIRAFF workbook samples to the Feb. 2, 2025 UTC launch date.
    The workbook's GPS week values are not used for calendar reconstruction.
    """
    return GIRAFF_LAUNCH_DATETIME + dt.timedelta(seconds=gps_msec_of_week_to_seconds_of_day(gps_msec_value))


def load_traj_records_xlsx(filename):
    """Load a mission trajectory workbook and return UTC time, flight time, and position arrays."""
    rows = list(iter_xlsx_sheet_rows(filename))
    if not rows:
        raise ValueError(f"No trajectory samples found in {filename}")

    header = [str(item).strip() for item in rows[0]]
    index_by_name = {name: idx for idx, name in enumerate(header) if name}
    required = {
        XLSX_FLIGHT_TIME_COLUMN,
        XLSX_LAT_COLUMN,
        XLSX_LON_COLUMN,
        XLSX_ALT_COLUMN,
        XLSX_GPS_MSEC_COLUMN,
        XLSX_GPS_WEEK_COLUMN,
    }
    missing = sorted(required - set(index_by_name))
    if missing:
        raise ValueError(f"Missing required XLSX trajectory columns in {filename}: {', '.join(missing)}")

    samples = []
    for row in rows[1:]:
        gps_value = row[index_by_name[XLSX_GPS_MSEC_COLUMN]] if index_by_name[XLSX_GPS_MSEC_COLUMN] < len(row) else ""
        if gps_value in ("", None):
            continue
        samples.append(row)
    if not samples:
        raise ValueError(f"No trajectory samples found in {filename}")

    sample_datetimes = np.array(
        [parse_giraff_sample_datetime(row[index_by_name[XLSX_GPS_MSEC_COLUMN]]) for row in samples],
        dtype=object,
    )
    utc_times = np.array(
        [(sample_dt - GIRAFF_LAUNCH_DATETIME).total_seconds() for sample_dt in sample_datetimes],
        dtype=float,
    )
    flight_times = np.array([float(row[index_by_name[XLSX_FLIGHT_TIME_COLUMN]]) for row in samples], dtype=float)
    lats = np.array([float(row[index_by_name[XLSX_LAT_COLUMN]]) for row in samples], dtype=float)
    lons = np.array([float(row[index_by_name[XLSX_LON_COLUMN]]) for row in samples], dtype=float)
    alts = np.array([float(row[index_by_name[XLSX_ALT_COLUMN]]) for row in samples], dtype=float)
    order = np.lexsort((flight_times, utc_times))
    utc_times = utc_times[order]
    flight_times = flight_times[order]
    lats = lats[order]
    lons = lons[order]
    alts = alts[order]
    return utc_times, flight_times, lats, lons, alts


def get_launch_start_from_traj_csv(filename):
    """Estimate T0 launch time from GPS UTC and official flight-time columns."""
    utc_times, flight_times, _lats, _lons, _alts = load_traj_records(filename)
    launch_seconds = np.nanmedian(utc_times - flight_times)
    return format_seconds_of_day(float(launch_seconds))


def format_time_since_launch(map_time, launch_start):
    """Format T+ seconds relative to launch for a requested map time."""
    if map_time is None or launch_start is None:
        return None
    rel_sec = hhmmss_fractional_to_seconds(map_time) - hhmmss_fractional_to_seconds(launch_start)
    return f"T{rel_sec:+.1f} s"


def fixed_utc_minute_marker_indices(utc_times, second_of_minute=30.0):
    """Return nearest-sample indices for fixed UTC minute markers, e.g. HH:MM:30."""
    if second_of_minute is None:
        return np.array([], dtype=int)
    utc_times = np.asarray(utc_times, dtype=float)
    if utc_times.size == 0:
        return np.array([], dtype=int)

    first_target = np.ceil((utc_times[0] - second_of_minute) / 60.0) * 60.0 + second_of_minute
    last_target = np.floor((utc_times[-1] - second_of_minute) / 60.0) * 60.0 + second_of_minute
    if first_target > last_target:
        return np.array([], dtype=int)

    targets = np.arange(first_target, last_target + 1e-9, 60.0, dtype=float)
    idx = [int(np.argmin(np.abs(utc_times - target))) for target in targets]
    return np.asarray(sorted(set(idx)), dtype=int)


def trajectory_marker_second(filename):
    """Return the fixed UTC second-of-minute used for trajectory marker placement."""
    path_str = str(filename)
    path_name = Path(path_str).name
    if path_name == GNEISS_LEFT_TRAJECTORY_PATH.name or "36397" in path_name:
        return 0.0
    if path_name == GNEISS_RIGHT_TRAJECTORY_PATH.name or "36398" in path_name:
        return 30.0
    if path_name == GIRAFF_LEFT_TRAJECTORY_PATH.name or "MAIN_PAYLOAD" in path_name:
        return None
    if path_name == GIRAFF_RIGHT_TRAJECTORY_PATH.name or "SUB_PAYLOAD" in path_name:
        return None
    return 30.0


def load_traj(filename, map_time=None, color="green"):
    """
    Load rocket trajectory from a GPS export file.
    Map lat/lon to the color-specific altitude and optionally return the nearest map-time point.
    """
    utc_times, flight_times, lats, lons, alts = load_traj_records(filename)

    lats, lons, _ = apex.map_to_height(lats, lons, alts, mapped_apex_height(color))
    idx = fixed_utc_minute_marker_indices(utc_times, second_of_minute=trajectory_marker_second(filename))
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


def build_traj_lookup(traj_path, color="green"):
    """Load and cache one trajectory for repeated nearest-time lookup."""
    lats, lons, _latm, _lonm, _lata, _lona, _lat_map, _lon_map = load_traj(traj_path, color=color)
    utc_times, flight_times, raw_lats, raw_lons, raw_alts = load_traj_records(traj_path)
    launch_start = get_launch_start_from_traj_csv(traj_path)
    return {
        "times": flight_times,
        "utc_times": utc_times,
        "lats": np.asarray(lats, dtype=float),
        "lons": np.asarray(lons, dtype=float),
        "raw_lats": np.asarray(raw_lats, dtype=float),
        "raw_lons": np.asarray(raw_lons, dtype=float),
        "raw_alts_km": np.asarray(raw_alts, dtype=float),
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


def lookup_traj_geodetic_position(traj_lookup, map_time):
    """Return the raw geodetic lat/lon/alt trajectory point nearest a requested map time."""
    if map_time is None:
        return None, None, None
    map_sec = hhmmss_fractional_to_seconds(map_time)
    utc_times = np.asarray(traj_lookup["utc_times"], dtype=float)
    if utc_times.size == 0 or map_sec < utc_times[0] or map_sec > utc_times[-1]:
        return None, None, None
    idx = int(np.argmin(np.abs(utc_times - map_sec)))
    return (
        float(traj_lookup["raw_lats"][idx]),
        float(traj_lookup["raw_lons"][idx]),
        float(traj_lookup["raw_alts_km"][idx]),
    )


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
