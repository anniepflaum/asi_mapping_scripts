#!/usr/bin/env python3
"""
Run map_asi_archive.py repeatedly over a requested time range.

Example:
    python3 map_asi_archive_series.py \
      --date 20260210 \
      --start 101900.0 \
      --end 102900.0 \
      --step 10 \
      --sites ARV BVR VEE \
      --color green \
      --bounds -150 -142 65 69 \
      --colorbar-color monochromatic \
      --plot-ipps
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from core.time_utils import parse_date_and_time, parse_hhmmss_fractional


SCRIPT_DIR = Path(__file__).resolve().parent
MAP_SCRIPT = SCRIPT_DIR / "map_asi_archive.py"


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


def build_command(args, time_arg):
    cmd = [
        "python3",
        str(MAP_SCRIPT),
        "--date",
        args.date,
        "--time",
        time_arg,
        "--sites",
        *args.sites,
        "--color",
        args.color,
        "--colorbar-color",
        args.colorbar_color,
        "--colorbar-scale",
        args.colorbar_scale,
    ]
    if not args.shared_norm:
        cmd.append("--no-shared-norm")
    if args.bounds is not None:
        cmd.extend(["--bounds", *(str(v) for v in args.bounds)])
    if args.pretty:
        cmd.append("--pretty")
    if args.plot_receivers:
        cmd.append("--plot-receivers")
    if args.plot_ipps:
        cmd.append("--plot-ipps")
    if args.plot_geodetic_traj:
        cmd.append("--plot-geodetic-traj")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260210", help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=10.0, help="Cadence in seconds")
    ap.add_argument("--sites", nargs="*", default=["ARV", "BVR", "VEE"], help="Sites to pass to map_asi_archive.py")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help="Optional map bounds override",
    )
    ap.add_argument("--colorbar-scale", choices=["linear", "log"], default="log", help="Colorbar scaling")
    ap.add_argument("--colorbar-color", choices=["viridis", "monochromatic"], default="monochromatic", help="Colorbar colormap")
    ap.add_argument(
        "--no-shared-norm",
        dest="shared_norm",
        action="store_false",
        default=True,
        help="Disable cross-site shared brightness normalization in downstream map calls",
    )
    ap.add_argument("--pretty", action="store_true", help="Use Cartopy plotting")
    ap.add_argument("--plot-receivers", action="store_true", help="Pass --plot-receivers through to map_asi_archive.py")
    ap.add_argument("--plot-ipps", action="store_true", help="Pass --plot-ipps through to map_asi_archive.py")
    ap.add_argument("--plot-geodetic-traj", action="store_true", help="Pass --plot-geodetic-traj through to map_asi_archive.py")
    args = ap.parse_args()

    try:
        parse_hhmmss_fractional(args.start)
        parse_hhmmss_fractional(args.end)
    except ValueError as exc:
        ap.error(str(exc))
    if args.step <= 0:
        ap.error("--step must be > 0")
    if args.bounds is not None:
        lon_min, lon_max, lat_min, lat_max = args.bounds
        if lon_min >= lon_max or lat_min >= lat_max:
            ap.error("--bounds must satisfy LON_MIN < LON_MAX and LAT_MIN < LAT_MAX")

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    total_steps = count_steps(start_dt, end_dt, args.step)
    step_idx = 0
    step_td = dt.timedelta(seconds=args.step)
    t = start_dt
    while t <= end_dt:
        step_idx += 1
        time_arg = format_time_arg(t)
        print_progress(step_idx, total_steps, time_arg)
        subprocess.run(build_command(args, time_arg), check=True, cwd=SCRIPT_DIR)
        t += step_td


if __name__ == "__main__":
    main()
