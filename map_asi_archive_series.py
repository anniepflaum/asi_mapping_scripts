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
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

from core.time_utils import parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from core.paths import mission_output_dir


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


def clear_progress_line():
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    sys.stdout.write("\r" + (" " * max(cols - 1, 1)) + "\r")
    sys.stdout.flush()


def format_step_token(step_seconds):
    return str(step_seconds).replace(".", "p").rstrip("0").rstrip("p")


def effective_sites(args):
    if args.sites is not None:
        sites = [site.upper() for site in args.sites]
    elif args.mission == "GIRAFF":
        sites = ["VEE"]
    else:
        sites = ["ARV", "VEE", "BVR"]
    if args.mission == "GIRAFF":
        return ["VEE"]
    return sorted(sites)


def series_output_dir(args):
    sites_str = "_".join(effective_sites(args))
    start_token = sanitize_time_for_filename(args.start)
    end_token = sanitize_time_for_filename(args.end)
    step_token = format_step_token(args.step)
    folder_name = f"{args.mission}_launch_{args.color}_{sites_str}_{args.date}_{start_token}_to_{end_token}_step_{step_token}"
    return mission_output_dir(args.mission, color=args.color, date=args.date) / folder_name


def frame_output_path(args, time_arg):
    sites_str = "_".join(effective_sites(args))
    time_token = sanitize_time_for_filename(time_arg)
    filename = f"{args.mission}_launch_{args.color}_{sites_str}_{args.date}_{time_token}.png"
    return mission_output_dir(args.mission, color=args.color, date=args.date) / filename


def move_frame_to_series_dir(args, time_arg, output_dir):
    source_path = frame_output_path(args, time_arg)
    if not source_path.exists():
        raise FileNotFoundError(f"Expected mapped frame was not created: {source_path}")
    destination_path = output_dir / source_path.name
    if destination_path.exists():
        destination_path.unlink()
    shutil.move(str(source_path), str(destination_path))
    return destination_path


def build_command(args, time_arg):
    cmd = [
        "python3",
        str(MAP_SCRIPT),
        "--date",
        args.date,
        "--time",
        time_arg,
        "--mission",
        args.mission,
        "--color",
        args.color,
        "--colorbar-color",
        args.colorbar_color,
        "--colorbar-scale",
        args.colorbar_scale,
    ]
    if args.sites is not None:
        cmd.extend(["--sites", *args.sites])
    if args.shared_norm:
        cmd.append("--shared-norm")
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
    ap.add_argument("--date", default=None, help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=10.0, help="Cadence in seconds")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to pass to map_asi_archive.py")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default="GNEISS", help="Mission dataset to use")
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
        "--shared-norm",
        dest="shared_norm",
        action="store_true",
        default=True,
        help="Enable cross-site shared brightness normalization in downstream map calls",
    )
    ap.add_argument("--pretty", action="store_true", help="Use Cartopy plotting")
    ap.add_argument("--plot-receivers", action="store_true", help="Pass --plot-receivers through to map_asi_archive.py")
    ap.add_argument("--plot-ipps", action="store_true", help="Pass --plot-ipps through to map_asi_archive.py")
    ap.add_argument("--plot-geodetic-traj", action="store_true", help="Pass --plot-geodetic-traj through to map_asi_archive.py")
    args = ap.parse_args()
    args.mission = args.mission.upper()
    if args.date is None:
        args.date = "20250202" if args.mission == "GIRAFF" else "20260210"
    if args.mission == "GIRAFF":
        if args.color != "green":
            ap.error("--mission GIRAFF only supports --color green")
        if args.sites is not None:
            selected_sites = {site.upper() for site in args.sites}
            invalid_sites = sorted(selected_sites - {"VEE"})
            if invalid_sites:
                ap.error(f"--mission GIRAFF only supports VEE TIFFs; remove site(s): {', '.join(invalid_sites)}")

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
            build_command(args, time_arg),
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
        destination_path = move_frame_to_series_dir(args, time_arg, output_dir)
        print(f"Moved frame to {destination_path}")
        step_idx += 1
        print_progress(step_idx, total_steps, time_arg)
        t += step_td


if __name__ == "__main__":
    main()
