import csv
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

from core.time_utils import parse_date_and_time


def format_time_arg(t):
    hhmmss = t.strftime("%H%M%S")
    frac = f"{t.microsecond:06d}".rstrip("0")
    return f"{hhmmss}.{frac}" if frac else hhmmss


def count_steps(start_dt, end_dt, step_seconds):
    total_seconds = max((end_dt - start_dt).total_seconds(), 0.0)
    return int(total_seconds / step_seconds) + 1


def print_progress(step_idx, total_steps, time_arg):
    width = 30
    filled = min(width, int(width * step_idx / max(total_steps, 1)))
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r[{bar}] {step_idx:>5}/{total_steps:<5} {time_arg}"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if step_idx >= total_steps:
        sys.stdout.write("\n")


def make_plot_output_path(csv_path, output_arg):
    if output_arg:
        return Path(output_arg)
    return Path(csv_path).with_suffix(".png")


def parse_series_filename(csv_path, prefix, allow_receiver_suffix=False):
    receiver_suffix = r"(?:_receivers_.+)?" if allow_receiver_suffix else ""
    pattern = (
        rf"^{re.escape(prefix)}_(?:(\d{{8}})_)?"
        rf"(\d{{6}}(?:p\d+)?)_(\d{{6}}(?:p\d+)?)_step(\d+(?:p\d+)?)"
        rf"{receiver_suffix}\.csv$"
    )
    match = re.match(pattern, csv_path.name)
    if not match:
        return None
    date_tok, start_tok, end_tok, step_tok = match.groups()
    return {
        "date": date_tok,
        "start_tok": start_tok,
        "end_tok": end_tok,
        "start_time": start_tok.replace("p", "."),
        "end_time": end_tok.replace("p", "."),
        "step": float(step_tok.replace("p", ".")),
    }


def build_requested_iso_times(date, start, end, step):
    start_dt = parse_date_and_time(date, start)
    end_dt = parse_date_and_time(date, end)
    step_td = dt.timedelta(seconds=step)
    requested = []
    t = start_dt
    while t <= end_dt:
        requested.append(t.isoformat())
        t += step_td
    return requested


def csv_contains_requested_times(csv_path, requested_times):
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            available = {row["time"] for row in reader if row.get("time")}
    except Exception:
        return False
    return all(time_value in available for time_value in requested_times)


def load_rows_from_csv(csv_path, required_fields):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"CSV file {csv_path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def csv_has_required_fields(csv_path, required_fields):
    if not required_fields:
        return True
    try:
        load_rows_from_csv(csv_path, required_fields)
    except Exception:
        return False
    return True


def find_reusable_csv(
    preferred_path,
    prefix,
    date,
    start,
    end,
    step,
    required_fields=None,
    allow_receiver_suffix=False,
    glob_pattern=None,
):
    requested_times = build_requested_iso_times(date, start, end, step)
    if (
        preferred_path.exists()
        and csv_contains_requested_times(preferred_path, requested_times)
        and csv_has_required_fields(preferred_path, required_fields)
    ):
        return preferred_path

    request_start_dt = parse_date_and_time(date, start)
    request_end_dt = parse_date_and_time(date, end)
    candidates = []
    for candidate in preferred_path.parent.glob(glob_pattern or f"{prefix}_*.csv"):
        parsed = parse_series_filename(candidate, prefix, allow_receiver_suffix=allow_receiver_suffix)
        if parsed is None:
            continue
        if parsed.get("date") is not None and parsed["date"] != date:
            continue
        try:
            candidate_start_dt = parse_date_and_time(date, parsed["start_time"])
            candidate_end_dt = parse_date_and_time(date, parsed["end_time"])
        except ValueError:
            continue
        if candidate_start_dt > request_start_dt or candidate_end_dt < request_end_dt:
            continue
        if (
            csv_contains_requested_times(candidate, requested_times)
            and csv_has_required_fields(candidate, required_fields)
        ):
            span_seconds = (candidate_end_dt - candidate_start_dt).total_seconds()
            candidates.append((span_seconds, parsed["step"], candidate))
    if not candidates:
        return preferred_path
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    return candidates[0][2]


def parse_frame_datetime_from_url(url):
    match = re.search(r"(\d{8})_(\d{6})", url)
    if not match:
        raise ValueError(f"Could not parse frame timestamp from URL: {url}")
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def load_tiff_frame_with_metadata(site, tiff_metadata, target_dt, frame_interval):
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
                "frame_dt": frame_dt,
                "delta_s": abs((frame_dt - target_dt).total_seconds()),
                "in_range": meta["start_dt"] <= target_dt <= meta["end_dt"],
                "cache_path": meta.get("cache_path"),
            }
        )

    in_range_candidates = [c for c in candidates if c["in_range"]]
    best = min(in_range_candidates or candidates, key=lambda c: c["delta_s"])

    if best.get("cache_path"):
        stack = np.load(best["cache_path"], mmap_mode="r")
        im = stack[best["idx"]]
    else:
        with tifffile.TiffFile(best["path"]) as tif:
            im = tif.pages[best["idx"]].asarray()
    if im.ndim == 3:
        im = im[:, :, 0]
    return im.astype("float32"), {"site": site, "frame_time": best["frame_dt"].isoformat()}
