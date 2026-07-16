#!/usr/bin/env python3
"""Shared TIFF discovery and nearest-frame loading helpers."""

import datetime as dt
import os
import re
import json
from glob import glob
from pathlib import Path

import numpy as np
import tifffile

from core.paths import WORKSPACE_DIR


GIRAFF_GREEN_CACHE_DIR = WORKSPACE_DIR / "images" / "green" / "VEE" / "GIRAFF"
GIRAFF_RED_CACHE_DIR = WORKSPACE_DIR / "images" / "red" / "6300" / "VEE" / "GIRAFF"
GIRAFF_BLUE_CACHE_DIR = WORKSPACE_DIR / "images" / "blue" / "VEE" / "GIRAFF"

GIRAFF_TIFF_PATHS_BY_COLOR_DATE = {
    "green": {
        "20250202": Path("/Volumes/LynchK/GIRAFF/GIRAFF/SOK_250202_5577/ut06/SOK250202_06595930_16bit.tif"),
        "20250209": Path("/Volumes/LynchK/GIRAFF/GIRAFF/SOK_250209_5577/ut07/SOK250209_07495931_16bit.tif"),
    },
    "red": {
        "20250202": Path("/Volumes/LynchK/GIRAFF/GIRAFF/THA_250202_8446/ut06/THA250202_06595921_16bit.tif"),
        "20250209": Path("/Volumes/LynchK/GIRAFF/GIRAFF/THA_250209_8446/ut07/THA250209_07475922_16bit.tif"),
    },
    "blue": {},
}
GIRAFF_CACHE_PATHS_BY_COLOR_DATE = {
    "green": {
        "20250202": (
            GIRAFF_GREEN_CACHE_DIR / "SOK250202_launch_070618_071637_uint16.npy",
            GIRAFF_GREEN_CACHE_DIR / "SOK250202_launch_070618_071637_metadata.json",
        ),
        "20250209": (
            GIRAFF_GREEN_CACHE_DIR / "SOK250209_083140_084415_uint16.npy",
            GIRAFF_GREEN_CACHE_DIR / "SOK250209_083140_084415_metadata.json",
        ),
    },
    "red": {
        "20250202": (
            GIRAFF_RED_CACHE_DIR / "THA250202_070618_071637_uint16.npy",
            GIRAFF_RED_CACHE_DIR / "THA250202_070618_071637_metadata.json",
        ),
        "20250209": (
            GIRAFF_RED_CACHE_DIR / "THA250209_083140_084415_uint16.npy",
            GIRAFF_RED_CACHE_DIR / "THA250209_083140_084415_metadata.json",
        ),
    },
    "blue": {},
}

# Backward-compatible names for older scripts/imports. Green remains the default.
GIRAFF_CACHE_DIR = GIRAFF_GREEN_CACHE_DIR
GIRAFF_TIFF_PATH = GIRAFF_TIFF_PATHS_BY_COLOR_DATE["green"]["20250202"]
GIRAFF_LOG_PATH = GIRAFF_TIFF_PATH.with_suffix(".log")
GIRAFF_CACHE_NPY_PATH, GIRAFF_CACHE_METADATA_PATH = GIRAFF_CACHE_PATHS_BY_COLOR_DATE["green"]["20250202"]
GIRAFF_TIFF_PATHS_BY_DATE = GIRAFF_TIFF_PATHS_BY_COLOR_DATE["green"]
GIRAFF_LOG_PATHS_BY_DATE = {
    date: path.with_suffix(".log") for date, path in GIRAFF_TIFF_PATHS_BY_DATE.items()
}
GIRAFF_CACHE_PATHS_BY_DATE = GIRAFF_CACHE_PATHS_BY_COLOR_DATE["green"]


def normalize_color(color):
    return str(color or "green").lower()


def iter_giraff_cache_paths():
    for paths_by_date in GIRAFF_CACHE_PATHS_BY_COLOR_DATE.values():
        for cache_path, metadata_path in paths_by_date.values():
            yield cache_path, metadata_path


def parse_tiff_start_datetime(tiff_path):
    """Parse TIFF start datetime from filename pattern *_YYYYMMDD_HHMMSS.tiff."""
    fname = os.path.basename(tiff_path)
    m = re.search(r"_(\d{8})_(\d{6})\.tiff?$", fname)
    if not m:
        raise ValueError(f"Could not parse start time from TIFF filename: {fname}")
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _parse_log_value(value):
    value = value.strip()
    if not value:
        return None
    try:
        if re.search(r"[.eE]", value):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_giraff_log(log_path):
    """Parse a GIRAFF sidecar acquisition log."""
    values = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
            values[key] = _parse_log_value(value)

    date_value = values.get("Date")
    start_value = values.get("TimeStart")
    stop_value = values.get("TimeStop")
    start_dt = dt.datetime.fromisoformat(f"{date_value}T{start_value}") if date_value and start_value else None
    end_dt = dt.datetime.fromisoformat(f"{date_value}T{stop_value}") if date_value and stop_value else None
    return {
        "values": values,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "n_frames": int(values["NumberOfImages"]) if values.get("NumberOfImages") is not None else None,
        "frame_interval": float(values["ImageCadence(s)"]) if values.get("ImageCadence(s)") is not None else None,
    }


def _parse_iso_datetime(value):
    return dt.datetime.fromisoformat(value) if value else None


def get_giraff_cache_paths(date_str=None, color="green"):
    date_key = str(date_str) if date_str is not None else None
    color_key = normalize_color(color)
    paths_by_date = GIRAFF_CACHE_PATHS_BY_COLOR_DATE.get(color_key, {})
    if date_key in paths_by_date:
        return paths_by_date[date_key]
    return (None, None)


def get_giraff_cache_metadata(date_str=None, color="green", cache_path=None, metadata_path=None):
    """Return local GIRAFF launch-window cache metadata if available."""
    if cache_path is None or metadata_path is None:
        cache_path, metadata_path = get_giraff_cache_paths(date_str, color=color)
    if cache_path is None or metadata_path is None or not cache_path.exists() or not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["cache_start_dt"] = _parse_iso_datetime(meta.get("cache_start_time"))
    meta["cache_end_dt"] = _parse_iso_datetime(meta.get("cache_end_time"))
    meta["cache_path"] = str(cache_path)
    meta["metadata_path"] = str(metadata_path)
    return meta


def load_giraff_cache_frame(target_dt, verbose=True, cache_path=None, metadata_path=None):
    """Load the closest GIRAFF frame from the local disk-backed launch cache."""
    meta = get_giraff_cache_metadata(cache_path=cache_path, metadata_path=metadata_path)
    if meta is None:
        return None

    frame_interval = float(meta["frame_interval_seconds"])
    start_dt = meta["cache_start_dt"]
    end_dt = meta["cache_end_dt"]
    n_frames = int(meta["cache_frame_count"])
    if start_dt is None or end_dt is None or n_frames <= 0:
        return None

    range_tolerance = dt.timedelta(seconds=frame_interval / 2)
    if not (start_dt - range_tolerance <= target_dt <= end_dt + range_tolerance):
        return None

    raw_idx = int(round((target_dt - start_dt).total_seconds() / frame_interval))
    idx = min(max(raw_idx, 0), n_frames - 1)
    frame_dt = start_dt + dt.timedelta(seconds=idx * frame_interval)
    cache_path = Path(meta["cache_path"])
    stack = np.load(cache_path, mmap_mode="r")
    im = np.asarray(stack[idx], dtype=np.float32)
    if verbose:
        print(
            f"VEE: {cache_path.name} cached frame "
            f"{idx + 1}/{n_frames} (delta {abs((frame_dt - target_dt).total_seconds()):.2f}s)"
        )
    return im, im.copy().astype(np.float32)


def sidecar_log_path(tiff_path):
    """Return the sidecar .log path for a TIFF if it exists."""
    for date_key, path in GIRAFF_TIFF_PATHS_BY_DATE.items():
        if Path(tiff_path) == path:
            log_path = GIRAFF_LOG_PATHS_BY_DATE[date_key]
            return log_path if log_path.exists() else None
    log_path = Path(tiff_path).with_suffix(".log")
    return log_path if log_path.exists() else None


def tiff_timing_metadata(tiff_path, fallback_frame_interval):
    """Return start/end/count/cadence metadata for a TIFF."""
    tiff_path = Path(tiff_path)
    for cache_path, metadata_path in iter_giraff_cache_paths():
        if tiff_path == cache_path:
            cache_meta = get_giraff_cache_metadata(cache_path=cache_path, metadata_path=metadata_path)
            if cache_meta is None:
                raise FileNotFoundError(f"GIRAFF cache metadata not found: {metadata_path}")
            return {
                "path": str(cache_path),
                "start_dt": cache_meta["cache_start_dt"],
                "end_dt": cache_meta["cache_end_dt"],
                "n_frames": int(cache_meta["cache_frame_count"]),
                "frame_interval": float(cache_meta["frame_interval_seconds"]),
                "source": str(metadata_path),
                "cache_path": str(cache_path),
            }
    log_path = sidecar_log_path(tiff_path)
    if log_path is not None:
        log_meta = parse_giraff_log(log_path)
        start_dt = log_meta["start_dt"]
        n_frames = log_meta["n_frames"]
        frame_interval = log_meta["frame_interval"] or fallback_frame_interval
        end_dt = log_meta["end_dt"]
        if end_dt is None and start_dt is not None and n_frames is not None:
            end_dt = start_dt + dt.timedelta(seconds=(n_frames - 1) * frame_interval)
        return {
            "path": tiff_path,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "n_frames": n_frames,
            "frame_interval": frame_interval,
            "source": str(log_path),
        }

    start_dt = parse_tiff_start_datetime(tiff_path)
    with tifffile.TiffFile(tiff_path) as tif:
        n_frames = len(tif.pages)
    return {
        "path": tiff_path,
        "start_dt": start_dt,
        "end_dt": start_dt + dt.timedelta(seconds=(n_frames - 1) * fallback_frame_interval),
        "n_frames": n_frames,
        "frame_interval": fallback_frame_interval,
        "source": "tiff_pages",
    }


def get_giraff_tiff_candidates(date_str, color="green", override_dirs=None):
    """Discover GIRAFF SOK/VEE TIFFs for a YYYYMMDD date."""
    date_key = str(date_str)
    color_key = normalize_color(color)
    cache_path, metadata_path = get_giraff_cache_paths(date_key, color=color_key)
    if get_giraff_cache_metadata(cache_path=cache_path, metadata_path=metadata_path) is not None:
        return [str(cache_path)]

    tiff_path = GIRAFF_TIFF_PATHS_BY_COLOR_DATE.get(color_key, {}).get(date_key)
    if tiff_path is None:
        supported = sorted(GIRAFF_TIFF_PATHS_BY_COLOR_DATE.get(color_key, {}))
        supported_text = ", ".join(supported) if supported else "none"
        print(f"VEE: supported GIRAFF {color_key} dates are {supported_text}, not {date_str}.")
        return []
    log_path = tiff_path.with_suffix(".log")
    if not tiff_path.exists():
        print(f"VEE: hard-coded GIRAFF {color_key} TIFF not found: {tiff_path}")
        return []
    if not log_path.exists():
        print(f"VEE: hard-coded GIRAFF {color_key} log not found: {log_path}")
        return []
    return [str(tiff_path)]


def red_image_dirs(site, mission, red_wavelength="6300"):
    wavelength = str(red_wavelength or "6300")
    mission_key = str(mission).upper()
    dirs = [f"../images/red/{wavelength}/{site}"]
    if site == "VEE":
        dirs.append(f"../images/red/{wavelength}/VEE/{mission_key}")
    if wavelength == "6300":
        dirs.append(f"../images/red/{site}")
        if site == "VEE":
            dirs.append(f"../images/red/VEE/{mission_key}")
    return dirs


def get_site_tiff_candidates(site, date_str, color, override_dirs=None, mission="GNEISS", red_wavelength="6300"):
    """
    Return candidate TIFF paths for a site.
    Priority:
    1) Explicit override directories if provided.
    2) Auto-discovered TIFFs in ../images/<COLOR>/<SITE>/.
       Red TIFFs default to ../images/red/6300/<SITE>/ and may be selected
       from ../images/red/<WAVELENGTH>/<SITE>/.
       VEE TIFFs may also live in a mission subdirectory.
    """
    if isinstance(override_dirs, str):
        override_dirs = [override_dirs]
    if str(mission).upper() == "GIRAFF" and site == "VEE":
        return get_giraff_tiff_candidates(date_str, color=color, override_dirs=override_dirs)

    site_prefixes = [site]
    color_key = normalize_color(color)
    if override_dirs:
        dirs_to_search = list(override_dirs)
    elif color_key == "red":
        dirs_to_search = red_image_dirs(site, mission, red_wavelength=red_wavelength)
    else:
        dirs_to_search = [f"../images/{color}/{site}"]
    if site == "VEE" and color_key != "red":
        alt_dir = f"../images/{color}/VEE/{str(mission).upper()}"
        if alt_dir not in dirs_to_search:
            dirs_to_search.append(alt_dir)

    matched_paths = []
    searched_patterns = []
    for folder in dirs_to_search:
        for prefix in site_prefixes:
            patterns = [
                os.path.join(folder, f"{prefix}_558_{date_str}_*.tiff"),
                os.path.join(folder, f"*_{date_str}_*.tiff"),
            ]
            for pattern in patterns:
                searched_patterns.append(pattern)
                matched_paths.extend(glob(pattern))

    unique_paths = sorted(set(matched_paths))
    if not unique_paths:
        print(f"{site}: no TIFFs matched for date {date_str}.")
        print(f"{site}: searched patterns: {', '.join(searched_patterns)}")
    return unique_paths


def load_best_frame_from_tiffs(site, tiff_paths, target_dt, frame_interval, color="green", verbose=True):
    """
    Search candidate TIFF tiles and load the frame closest to target_dt.
    Returns (raw_image, normalized_image), both float32 arrays.
    """
    cache_paths = {cache_path for cache_path, _metadata_path in iter_giraff_cache_paths()}
    tiff_lookup_paths = {
        path
        for paths_by_date in GIRAFF_TIFF_PATHS_BY_COLOR_DATE.values()
        for path in paths_by_date.values()
    }
    selected_cache_path = next((Path(path) for path in tiff_paths if Path(path) in cache_paths), None)
    uses_giraff_cache = site == "VEE" and selected_cache_path is not None
    if uses_giraff_cache or (site == "VEE" and any(Path(path) in tiff_lookup_paths for path in tiff_paths)):
        metadata_path = None
        if selected_cache_path is not None:
            for cache_path, candidate_metadata_path in iter_giraff_cache_paths():
                if selected_cache_path == cache_path:
                    metadata_path = candidate_metadata_path
                    break
        cached = load_giraff_cache_frame(target_dt, verbose=verbose, cache_path=selected_cache_path, metadata_path=metadata_path)
        if cached is not None:
            return cached
        if uses_giraff_cache:
            meta = get_giraff_cache_metadata(cache_path=selected_cache_path, metadata_path=metadata_path)
            if meta is not None:
                raise ValueError(
                    "Requested GIRAFF time is outside the local cache window "
                    f"{meta.get('cache_start_time')} to {meta.get('cache_end_time')}"
                )

    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found for {site}")

    candidates = []
    errors = []
    for path in tiff_paths:
        try:
            meta = tiff_timing_metadata(path, frame_interval)
            start_dt = meta["start_dt"]
            end_dt = meta["end_dt"]
            n_frames = meta["n_frames"]
            file_frame_interval = meta["frame_interval"]
            if start_dt is None or end_dt is None or n_frames is None:
                raise ValueError(f"Incomplete timing metadata from {meta['source']}")
            raw_idx = int(round((target_dt - start_dt).total_seconds() / file_frame_interval))
            idx = min(max(raw_idx, 0), n_frames - 1)
            frame_dt = start_dt + dt.timedelta(seconds=idx * file_frame_interval)
            delta_s = abs((frame_dt - target_dt).total_seconds())
            range_tolerance = dt.timedelta(seconds=file_frame_interval / 2)
            in_range = (start_dt - range_tolerance) <= target_dt <= (end_dt + range_tolerance)
            candidates.append(
                {
                    "path": path,
                    "idx": idx,
                    "n_frames": n_frames,
                    "delta_s": delta_s,
                    "frame_dt": frame_dt,
                    "in_range": in_range,
                    "metadata_source": meta["source"],
                }
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if not candidates:
        raise RuntimeError(f"All TIFF candidates failed for {site}: {'; '.join(errors)}")

    in_range_candidates = [c for c in candidates if c["in_range"]]
    if in_range_candidates:
        best = min(in_range_candidates, key=lambda c: c["delta_s"])
    else:
        best = min(candidates, key=lambda c: c["delta_s"])
        if verbose:
            print(f"{site}: requested time outside all tile ranges; using nearest boundary frame.")

    if verbose:
        print(
            f"{site}: {os.path.basename(best['path'])} frame "
            f"{best['idx'] + 1}/{best['n_frames']} "
            f"(delta {best['delta_s']:.2f}s, metadata {best['metadata_source']})"
        )

    with tifffile.TiffFile(best["path"]) as tif:
        im = tif.pages[best["idx"]].asarray()
    if im.ndim == 3:
        im = im[:, :, 0]
    im_raw = im.astype(np.float32)
    return im_raw, im_raw.copy().astype(np.float32)


def build_tiff_metadata(tiff_paths, frame_interval):
    """Cache TIFF time coverage so repeated frame selection avoids rescanning files."""
    metadata = []
    for path in tiff_paths:
        metadata.append(tiff_timing_metadata(path, frame_interval))
    return metadata


def load_best_frame_from_cached_tiffs(site, tiff_metadata, target_dt, frame_interval):
    """Load the frame nearest target_dt using precomputed TIFF coverage metadata."""
    if not tiff_metadata:
        raise FileNotFoundError(f"No TIFF files found for {site}")

    candidates = []
    for meta in tiff_metadata:
        file_frame_interval = meta.get("frame_interval") or frame_interval
        raw_idx = int(round((target_dt - meta["start_dt"]).total_seconds() / file_frame_interval))
        idx = min(max(raw_idx, 0), meta["n_frames"] - 1)
        frame_dt = meta["start_dt"] + dt.timedelta(seconds=idx * file_frame_interval)
        candidates.append(
            {
                "path": meta["path"],
                "idx": idx,
                "delta_s": abs((frame_dt - target_dt).total_seconds()),
                "in_range": (
                    meta["start_dt"] - dt.timedelta(seconds=file_frame_interval / 2)
                    <= target_dt
                    <= meta["end_dt"] + dt.timedelta(seconds=file_frame_interval / 2)
                ),
            }
        )

    in_range_candidates = [c for c in candidates if c["in_range"]]
    best = min(in_range_candidates or candidates, key=lambda c: c["delta_s"])

    if Path(best["path"]).suffix.lower() == ".npy":
        stack = np.load(best["path"], mmap_mode="r")
        im = np.asarray(stack[best["idx"]], dtype=np.float32)
        return im

    with tifffile.TiffFile(best["path"]) as tif:
        im = tif.pages[best["idx"]].asarray()
    if im.ndim == 3:
        im = im[:, :, 0]
    return im.astype(np.float32)
