#!/usr/bin/env python3
"""Run map_HER.py repeatedly over a requested time range."""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import map_HER
from core.series_utils import count_steps, format_time_arg, print_progress
from core.time_utils import parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename


SCRIPT_DIR = Path(__file__).resolve().parent
MAP_SCRIPT = SCRIPT_DIR / "map_HER.py"


def clear_progress_line():
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    sys.stdout.write("\r" + (" " * max(cols - 1, 1)) + "\r")
    sys.stdout.flush()


def format_step_token(step_seconds):
    return str(step_seconds).replace(".", "p").rstrip("0").rstrip("p")


def series_output_dir(args):
    start_token = sanitize_time_for_filename(args.start)
    end_token = sanitize_time_for_filename(args.end)
    step_token = format_step_token(args.step)
    folder_name = f"HER_on_VEE_mapped_{map_HER.DEFAULT_DATE}_{start_token}_to_{end_token}_step_{step_token}"
    return map_HER.mission_output_dir("GNEISS", color="green", date=map_HER.DEFAULT_DATE) / folder_name


def move_frame_to_series_dir(time_arg, output_dir):
    source_path = map_HER.output_path(time_arg)
    if not source_path.exists():
        raise FileNotFoundError(f"Expected mapped frame was not created: {source_path}")
    destination_path = output_dir / source_path.name
    if destination_path.exists():
        destination_path.unlink()
    shutil.move(str(source_path), str(destination_path))
    return destination_path


def build_command(time_arg):
    return ["python3", str(MAP_SCRIPT), "--time", time_arg]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a HER/HERA mapped frame series.")
    parser.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    parser.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    parser.add_argument("--step", type=float, required=True, help="Cadence in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        parse_hhmmss_fractional(args.start)
        parse_hhmmss_fractional(args.end)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if args.step <= 0:
        raise SystemExit("error: --step must be > 0")

    start_dt = parse_date_and_time(map_HER.DEFAULT_DATE, args.start)
    end_dt = parse_date_and_time(map_HER.DEFAULT_DATE, args.end)
    if end_dt < start_dt:
        raise SystemExit("error: --end must be >= --start")

    total_steps = count_steps(start_dt, end_dt, args.step)
    output_dir = series_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving series frames to {output_dir}")

    step_idx = 0
    step_td = dt.timedelta(seconds=args.step)
    t = start_dt
    print_progress(0, total_steps, format_time_arg(start_dt))
    while t <= end_dt:
        time_arg = format_time_arg(t)
        result = subprocess.run(
            build_command(time_arg),
            check=False,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )
        clear_progress_line()
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        destination_path = move_frame_to_series_dir(time_arg, output_dir)
        print(f"Moved frame to {destination_path}")
        step_idx += 1
        print_progress(step_idx, total_steps, time_arg)
        t += step_td


if __name__ == "__main__":
    main()
