"""EZIE MEM ephemeris loading and magnetic-field-line mapping helpers."""

import csv
import datetime as dt
from collections import defaultdict

import numpy as np
from apexpy import Apex

from core.constants import DEFAULT_GREEN_ALT_KM
from core.paths import EZIE_MEM_EPHEMERIS_PATH
from core.time_utils import hhmmss_fractional_to_seconds


EZIE_SPACECRAFT_ORDER = ("EZIE-B", "EZIE-A", "EZIE-C")
EZIE_MEM_INDICES = {"1", "2", "3"}
EZIE_MEM_COLORS = {
    "1": "tab:blue",
    "2": "tab:orange",
    "3": "tab:green",
}
EZIE_SPACECRAFT_LINESTYLES = {
    "EZIE-B": "-",
    "EZIE-A": "--",
    "EZIE-C": ":",
}
EZIE_MEM_MARKERS = {
    "1": "o",
    "2": "s",
    "3": "D",
}

_apex = Apex()


def parse_iso_seconds_of_day(value):
    sample_dt = dt.datetime.fromisoformat(value)
    return (
        sample_dt.hour * 3600.0
        + sample_dt.minute * 60.0
        + sample_dt.second
        + sample_dt.microsecond / 1e6
    )


def ezie_sort_key(key):
    spacecraft, mem_index = key
    try:
        spacecraft_order = EZIE_SPACECRAFT_ORDER.index(spacecraft)
    except ValueError:
        spacecraft_order = len(EZIE_SPACECRAFT_ORDER)
    try:
        mem_order = int(mem_index)
    except ValueError:
        mem_order = 99
    return spacecraft_order, mem_order


def load_ezie_mem_tracks(path=EZIE_MEM_EPHEMERIS_PATH, map_alt_km=DEFAULT_GREEN_ALT_KM):
    grouped = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd)
        required = {
            "spacecraft",
            "mem_index",
            "time_utc",
            "latitude_deg",
            "longitude_deg",
            "altitude_km",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing EZIE ephemeris columns in {path}: {', '.join(sorted(missing))}")
        for row in reader:
            if row["mem_index"] not in EZIE_MEM_INDICES:
                continue
            grouped[(row["spacecraft"], row["mem_index"])].append(row)

    tracks = []
    for spacecraft, mem_index in sorted(grouped, key=ezie_sort_key):
        rows = grouped[(spacecraft, mem_index)]
        times = np.asarray([parse_iso_seconds_of_day(row["time_utc"]) for row in rows], dtype=float)
        lats = np.asarray([float(row["latitude_deg"]) for row in rows], dtype=float)
        lons = np.asarray([float(row["longitude_deg"]) for row in rows], dtype=float)
        alts = np.asarray([float(row["altitude_km"]) for row in rows], dtype=float)
        order = np.argsort(times)
        times = times[order]
        lats = lats[order]
        lons = lons[order]
        alts = alts[order]
        mapped_lats, mapped_lons, _mapped_alts = _apex.map_to_height(lats, lons, alts, map_alt_km)
        mem_label = f"{spacecraft} MEM{mem_index}"
        tracks.append(
            {
                "spacecraft": spacecraft,
                "mem_index": mem_index,
                "label": mem_label,
                "times": times,
                "lat": np.asarray(mapped_lats, dtype=float),
                "lon": np.asarray(mapped_lons, dtype=float),
            }
        )
    return tracks


def ezie_position_at_time(track, map_time):
    if map_time is None:
        return None, None
    map_sec = hhmmss_fractional_to_seconds(map_time)
    times = track["times"]
    if times.size == 0 or map_sec < times[0] or map_sec > times[-1]:
        return None, None
    idx = int(np.argmin(np.abs(times - map_sec)))
    return float(track["lat"][idx]), float(track["lon"][idx])
