#!/usr/bin/env python3
"""
Run map_asi_archive.py repeatedly over a requested time range.

Example:
    python3 map_asi_archive_series.py \
      --rocket 397 \
      --step 10 \
      --sites ARV BVR VEE \
      --color green \
      --bounds -150 -142 65 69 \
      --plot-ipps
"""

import argparse
import datetime as dt
import io
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

from core.constants import NORMALIZATION_LOWER_PERCENTILE, NORMALIZATION_UPPER_PERCENTILE
from core.series_utils import count_steps, format_time_arg, print_progress
from core.time_utils import parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from core.missions import default_sites, default_time_range, mission_output_dir, resolve_mission_and_date, validate_color_and_sites
import map_asi_archive


def clear_progress_line():
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    sys.stdout.write("\r" + (" " * max(cols - 1, 1)) + "\r")
    sys.stdout.flush()


def format_step_token(step_seconds):
    return str(step_seconds).replace(".", "p").rstrip("0").rstrip("p")


def wavelength_token(color, red_wavelength="6300"):
    if str(color).lower() == "green":
        return "557p7nm"
    if str(color).lower() == "red":
        red_tokens = {
            "6300": "630nm",
            "8446": "844p6nm",
        }
        return red_tokens.get(str(red_wavelength), f"{str(red_wavelength).replace('.', 'p')}nm")
    return str(color).lower()


def effective_sites(args):
    if args.sites is not None:
        sites = [site.upper() for site in args.sites]
    else:
        sites = default_sites(args.mission)
    return sorted(sites)


def series_output_dir(args):
    sites_str = "_".join(effective_sites(args))
    start_token = sanitize_time_for_filename(args.start)
    end_token = sanitize_time_for_filename(args.end)
    step_token = format_step_token(args.step)
    folder_name = f"{wavelength_token(args.color, args.red_wavelength)}_{sites_str}_{args.date}_{start_token}_to_{end_token}_step_{step_token}"
    return mission_output_dir(args.mission, color=args.color, date=args.date) / folder_name


def frame_output_path(args, time_arg):
    sites_str = "_".join(effective_sites(args))
    time_token = sanitize_time_for_filename(time_arg)
    render_suffix = "" if args.render_mode == "auto" else f"_{args.render_mode}"
    filename = (
        f"calibrated_{args.color}_{sites_str}_{args.date}_{time_token}"
        f"{render_suffix}.png"
    )
    return mission_output_dir(args.mission, color=args.color, date=args.date) / filename


def move_frame_to_series_dir(args, time_arg, output_dir, source_path=None):
    source_path = Path(source_path) if source_path is not None else frame_output_path(args, time_arg)
    if not source_path.exists():
        raise FileNotFoundError(f"Expected mapped frame was not created: {source_path}")
    destination_path = output_dir / source_path.name
    if destination_path.exists():
        destination_path.unlink()
    shutil.move(str(source_path), str(destination_path))
    return destination_path


def build_map_args(args, time_arg):
    map_args = [
        "--time",
        time_arg,
        "--mission",
        args.mission,
        "--color",
        args.color,
        "--red-wavelength",
        args.red_wavelength,
        "--colorbar-scale",
        args.colorbar_scale,
        "--vmax",
        str(args.vmax),
        "--render-mode",
        args.render_mode,
    ]
    if args.rocket is not None:
        map_args.extend(["--rocket", args.rocket])
    if args.sites is not None:
        map_args.extend(["--sites", *args.sites])
    if args.no_shared_norm:
        map_args.append("--no-shared-norm")
    if args.bounds is not None:
        map_args.extend(["--bounds", *(str(v) for v in args.bounds)])
    if args.pretty:
        map_args.append("--pretty")
    if args.plot_receivers:
        map_args.append("--plot-receivers")
    if args.plot_ipps:
        map_args.append("--plot-ipps")
    if args.plot_geodetic_traj:
        map_args.append("--plot-geodetic-traj")
    if args.plot_ezie:
        map_args.append("--plot-ezie")
    return map_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the ASI image date")
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=10.0, help="Cadence in seconds")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to pass to map_asi_archive.py")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--red-wavelength", choices=["6300", "8446"], default="6300", help="Red-channel wavelength directory to pass through to map_asi_archive.py")
    ap.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help="Optional map bounds override",
    )
    ap.add_argument("--colorbar-scale", choices=["linear", "log"], default="log", help="Colorbar scaling")
    ap.add_argument("--vmax", type=float, default=NORMALIZATION_UPPER_PERCENTILE, help="Upper percentile used as vmax for downstream ASI normalization")
    ap.add_argument(
        "--no-shared-norm",
        dest="no_shared_norm",
        action="store_true",
        default=False,
        help="Disable cross-site shared brightness normalization in downstream map calls",
    )
    ap.add_argument("--pretty", action="store_true", help="Use Cartopy plotting")
    ap.add_argument("--plot-receivers", action="store_true", help="Pass --plot-receivers through to map_asi_archive.py")
    ap.add_argument("--plot-ipps", action="store_true", help="Pass --plot-ipps through to map_asi_archive.py")
    ap.add_argument("--plot-geodetic-traj", action="store_true", help="Pass --plot-geodetic-traj through to map_asi_archive.py")
    ap.add_argument("--plot-ezie", action="store_true", help="Pass --plot-ezie through to map_asi_archive.py")
    ap.add_argument(
        "--render-mode",
        choices=["auto", "pcolor", "pcolormesh", "points", "regrid"],
        default="auto",
        help="ASI rendering method passed to map_asi_archive.py",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print the detailed map_asi_archive.py log for every frame",
    )
    args = ap.parse_args()
    try:
        args.mission, args.date = resolve_mission_and_date(args.mission, args.rocket)
    except ValueError as exc:
        ap.error(str(exc))
    if not (NORMALIZATION_LOWER_PERCENTILE < args.vmax <= 100):
        ap.error(f"--vmax must be > {NORMALIZATION_LOWER_PERCENTILE:g} and <= 100")
    default_start, default_end = default_time_range(
        args.mission,
        args.date,
        rocket_tags=[args.rocket] if args.rocket is not None else None,
    )
    if args.start is None:
        args.start = default_start
    if args.end is None:
        args.end = default_end
    if args.sites is not None:
        validate_color_and_sites(ap, args.mission, args.color, args.sites, giraff_message_site="VEE TIFFs and PKR PNGs")
    else:
        validate_color_and_sites(ap, args.mission, args.color, default_sites(args.mission), giraff_message_site="VEE TIFFs and PKR PNGs")

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
        frame_stdout = io.StringIO()
        frame_stderr = io.StringIO()
        try:
            with redirect_stdout(frame_stdout), redirect_stderr(frame_stderr):
                source_path = map_asi_archive.main(build_map_args(args, time_arg))
        except Exception:
            clear_progress_line()
            sys.stdout.write(frame_stdout.getvalue())
            sys.stderr.write(frame_stderr.getvalue())
            raise
        destination_path = move_frame_to_series_dir(
            args, time_arg, output_dir, source_path=source_path
        )
        if args.verbose:
            clear_progress_line()
            sys.stdout.write(frame_stdout.getvalue())
            sys.stderr.write(frame_stderr.getvalue())
            print(f"Moved frame to {destination_path}")
        step_idx += 1
        print_progress(step_idx, total_steps, time_arg)
        t += step_td
    print(f"Saved {step_idx} frames to {output_dir}")


if __name__ == "__main__":
    main()
