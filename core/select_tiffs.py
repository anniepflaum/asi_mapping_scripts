#!/usr/bin/env python3
"""Select sampled GNEISS TIFF frames for ARV, VEE, and BVR."""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.tiff_utils import get_site_tiff_candidates, tiff_timing_metadata
from core.time_utils import parse_date_and_time, sanitize_time_for_filename


SITES = ("ARV", "VEE", "BVR")


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Select frames from GNEISS ARV, VEE, and BVR TIFFs over a time "
            "range and write one multi-page TIFF per site."
        )
    )
    ap.add_argument("--date", default="20260210", help="Image date as YYYYMMDD, for example 20260210")
    ap.add_argument("--start", required=True, help="Start time as HHMMSS or HHMMSS.s")
    ap.add_argument("--end", required=True, help="End time as HHMMSS or HHMMSS.s")
    ap.add_argument(
        "--step",
        default=None,
        type=float,
        help="Sampling step in seconds. Default: native frame cadence for the color (green/blue 0.3 s, red 0.9 s).",
    )
    ap.add_argument("--color", choices=("green", "red", "blue"), default="green", help="ASI color channel")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for selected TIFFs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing selected TIFF outputs.",
    )
    return ap.parse_args()


def frame_interval_for_color(color):
    if color == "red":
        return FRAME_INTERVAL_SECONDS_RED
    return FRAME_INTERVAL_SECONDS_GREEN


def sampled_times(start_dt, end_dt, step_s):
    if step_s <= 0:
        raise ValueError("--step must be > 0")
    if end_dt < start_dt:
        raise ValueError("--end must be at or after --start")

    times = []
    current = start_dt
    step = dt.timedelta(seconds=step_s)
    epsilon = dt.timedelta(microseconds=1)
    while current <= end_dt + epsilon:
        times.append(current)
        current += step
    return times


def build_site_metadata(site, date, color, frame_interval):
    candidates = get_site_tiff_candidates(site, date, color, mission="GNEISS")
    if not candidates:
        raise FileNotFoundError(f"No GNEISS {color} TIFF files found for {site} on {date}")
    return [tiff_timing_metadata(path, frame_interval) for path in candidates]


def best_frame_selection(site, metadata, target_dt, fallback_frame_interval):
    choices = []
    for meta in metadata:
        frame_interval = meta.get("frame_interval") or fallback_frame_interval
        start_dt = meta["start_dt"]
        end_dt = meta["end_dt"]
        n_frames = meta["n_frames"]
        if start_dt is None or end_dt is None or n_frames is None:
            raise ValueError(f"{site}: incomplete timing metadata from {meta['source']}")

        raw_idx = int(round((target_dt - start_dt).total_seconds() / frame_interval))
        idx = min(max(raw_idx, 0), n_frames - 1)
        frame_dt = start_dt + dt.timedelta(seconds=idx * frame_interval)
        tolerance = dt.timedelta(seconds=frame_interval / 2)
        choices.append(
            {
                "path": Path(meta["path"]),
                "idx": idx,
                "frame_dt": frame_dt,
                "delta_s": abs((frame_dt - target_dt).total_seconds()),
                "in_range": start_dt - tolerance <= target_dt <= end_dt + tolerance,
            }
        )

    in_range = [choice for choice in choices if choice["in_range"]]
    return min(in_range or choices, key=lambda choice: choice["delta_s"])


def read_selected_frame(selection):
    path = selection["path"]
    if path.suffix.lower() == ".npy":
        stack = np.load(path, mmap_mode="r")
        frame = np.asarray(stack[selection["idx"]])
    else:
        with tifffile.TiffFile(path) as tif:
            frame = tif.pages[selection["idx"]].asarray()
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame


def output_path_for_site(output_dir, site, date, start, end, _step, color):
    start_token = sanitize_time_for_filename(start)
    end_token = sanitize_time_for_filename(end)
    return output_dir / f"{site}_GNEISS_{color}_{date}_{start_token}_{end_token}.tiff"


def write_site_tiff(site, metadata, times, output_path, frame_interval, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite to replace it)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selections = [best_frame_selection(site, metadata, target_dt, frame_interval) for target_dt in times]

    with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
        for selection in selections:
            writer.write(read_selected_frame(selection))

    first = selections[0]
    last = selections[-1]
    print(
        f"{site}: wrote {len(selections)} frames to {output_path} "
        f"({first['frame_dt'].strftime('%H:%M:%S.%f')[:-3]}.."
        f"{last['frame_dt'].strftime('%H:%M:%S.%f')[:-3]})"
    )
    return output_path


def main():
    args = parse_args()
    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    frame_interval = frame_interval_for_color(args.color)
    if args.step is None:
        args.step = frame_interval
    times = sampled_times(start_dt, end_dt, args.step)
    output_dir = Path("Users/anniepflaum/lab317/rocket_gif")

    outputs = []
    for site in SITES:
        metadata = build_site_metadata(site, args.date, args.color, frame_interval)
        output_path = output_path_for_site(output_dir, site, args.date, args.start, args.end, args.step, args.color)
        outputs.append(write_site_tiff(site, metadata, times, output_path, frame_interval, overwrite=args.overwrite))

    print("Selected TIFFs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
