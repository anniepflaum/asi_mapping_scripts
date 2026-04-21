#!/usr/bin/env python3
"""Shared TIFF discovery and nearest-frame loading helpers."""

import datetime as dt
import os
import re
from glob import glob

import numpy as np
import tifffile


def parse_tiff_start_datetime(tiff_path):
    """Parse TIFF start datetime from filename pattern *_YYYYMMDD_HHMMSS.tiff."""
    fname = os.path.basename(tiff_path)
    m = re.search(r"_(\d{8})_(\d{6})\.tiff$", fname)
    if not m:
        raise ValueError(f"Could not parse start time from TIFF filename: {fname}")
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def get_site_tiff_candidates(site, date_str, color, override_dirs=None):
    """
    Return candidate TIFF paths for a site.
    Priority:
    1) Explicit override directories if provided.
    2) Auto-discovered TIFFs in ../raw_tiffs/<COLOR>/<SITE>/.
       VEE green TIFFs may also live in ../raw_tiffs/<COLOR>/VEE/GNEISS/.
    """
    if isinstance(override_dirs, str):
        override_dirs = [override_dirs]
    site_prefixes = [site]
    dirs_to_search = list(override_dirs) if override_dirs else [f"../raw_tiffs/{color}/{site}"]
    if site == "VEE":
        alt_dir = f"../raw_tiffs/{color}/VEE/GNEISS"
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
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found for {site}")

    candidates = []
    errors = []
    for path in tiff_paths:
        try:
            start_dt = parse_tiff_start_datetime(path)
            with tifffile.TiffFile(path) as tif:
                n_frames = len(tif.pages)
            end_dt = start_dt + dt.timedelta(seconds=(n_frames - 1) * frame_interval)
            raw_idx = int(round((target_dt - start_dt).total_seconds() / frame_interval))
            idx = min(max(raw_idx, 0), n_frames - 1)
            frame_dt = start_dt + dt.timedelta(seconds=idx * frame_interval)
            delta_s = abs((frame_dt - target_dt).total_seconds())
            in_range = start_dt <= target_dt <= end_dt
            candidates.append(
                {
                    "path": path,
                    "idx": idx,
                    "n_frames": n_frames,
                    "delta_s": delta_s,
                    "frame_dt": frame_dt,
                    "in_range": in_range,
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
            f"(delta {best['delta_s']:.2f}s)"
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
        start_dt = parse_tiff_start_datetime(path)
        with tifffile.TiffFile(path) as tif:
            n_frames = len(tif.pages)
        metadata.append(
            {
                "path": path,
                "start_dt": start_dt,
                "n_frames": n_frames,
                "end_dt": start_dt + dt.timedelta(seconds=(n_frames - 1) * frame_interval),
            }
        )
    return metadata


def load_best_frame_from_cached_tiffs(site, tiff_metadata, target_dt, frame_interval):
    """Load the frame nearest target_dt using precomputed TIFF coverage metadata."""
    if not tiff_metadata:
        raise FileNotFoundError(f"No TIFF files found for {site}")

    candidates = []
    for meta in tiff_metadata:
        raw_idx = int(round((target_dt - meta["start_dt"]).total_seconds() / frame_interval))
        idx = min(max(raw_idx, 0), meta["n_frames"] - 1)
        frame_dt = meta["start_dt"] + dt.timedelta(seconds=idx * frame_interval)
        candidates.append(
            {
                "path": meta["path"],
                "idx": idx,
                "delta_s": abs((frame_dt - target_dt).total_seconds()),
                "in_range": meta["start_dt"] <= target_dt <= meta["end_dt"],
            }
        )

    in_range_candidates = [c for c in candidates if c["in_range"]]
    best = min(in_range_candidates or candidates, key=lambda c: c["delta_s"])

    with tifffile.TiffFile(best["path"]) as tif:
        im = tif.pages[best["idx"]].asarray()
    if im.ndim == 3:
        im = im[:, :, 0]
    return im.astype(np.float32)
