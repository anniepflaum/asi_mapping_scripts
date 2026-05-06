#!/usr/bin/env python3
"""
Build a brightness-vs-time dataset for mission trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket position at that time.
3) Samples brightness at rocket position using the same logic as map_asi_archive.py
   (mean of 25 nearest valid pixels, with percentile metadata).
4) Writes one CSV row per timestamp with trajectory brightness values.
"""

import argparse
import csv
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from core.brightness import best_rocket_brightness
from core.constants import DEFAULT_GREEN_ALT_KM, FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.plot_norm import compute_reference_norm_limits, reference_normalization_time
from core.remote_data import retrieve_image
from core.series_utils import (
    build_requested_iso_times,
    count_steps,
    find_reusable_csv,
    format_time_arg,
    load_rows_from_csv,
    load_tiff_frame_with_metadata,
    parse_frame_datetime_from_url,
    print_progress,
)
from core.skymaps import load_skymaps
from core.time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from core.fetch_url import closest_amisr_png_url
from core.missions import default_sites, default_time_range, mission_output_dir, resolve_mission_and_date, trajectory_config_tuples, validate_color_and_sites
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates
from core.traj_utils import (
    build_traj_lookup,
    lookup_traj_geodetic_position,
    lookup_traj_position,
)


def series_prefix(mission):
    return "brightness_vs_time" if str(mission).upper() == "GNEISS" else f"{str(mission).upper()}_brightness_vs_time"


def format_alt_token(alt_km):
    return str(float(alt_km)).replace(".", "p").rstrip("0").rstrip("p")


def green_alt_suffix(color, green_alt):
    if str(color).lower() != "green" or green_alt is None or float(green_alt) == DEFAULT_GREEN_ALT_KM:
        return ""
    return f"_alt_{format_alt_token(green_alt)}km"


def make_output_path(mission, date, start, end, step, color="green", green_alt=None):
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    return Path(f"{series_prefix(mission)}_{date}_{start_tok}_{end_tok}_step{step_tok}{green_alt_suffix(color, green_alt)}.csv")


def make_plot_output_path(csv_path):
    return Path(csv_path).with_suffix(".png")


def plot_brightness_timeseries(times, series_by_key, output_path, title, labels_by_key=None):
    if not times:
        raise ValueError("No rows with iso_time were collected")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"left": "tab:blue", "right": "tab:red"}
    labels_by_key = labels_by_key or {}
    for key, values in series_by_key.items():
        label = labels_by_key.get(key, key)
        color = colors.get(key)
        ax.plot(times, values, linewidth=1.0, color=color, label=f"{label} brightness")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_yscale("log")
    ax.set_ylabel("Brightness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def normalize_reference_brightness(raw_brightness, norm_limits):
    if raw_brightness is None or norm_limits is None:
        return None
    vmin, vmax = norm_limits
    if vmin is None or vmax is None or vmax <= vmin:
        return None
    return (float(raw_brightness) - float(vmin)) / (float(vmax) - float(vmin))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the ASI image date")
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=0.05, help="Step size in seconds")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to include")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--green-alt", type=float, default=None, help="Mapped altitude in km for green-channel skymaps and trajectories")
    ap.add_argument("--no-plot", action="store_true", help="Write the CSV only and skip the PNG plot")
    ap.add_argument("--no-csv", action="store_true", help="Skip CSV generation and plot from an existing CSV instead")
    args = ap.parse_args()
    try:
        args.mission, args.date = resolve_mission_and_date(args.mission, args.rocket)
    except ValueError as exc:
        ap.error(str(exc))
    default_start, default_end = default_time_range(
        args.mission,
        args.date,
        rocket_tags=[args.rocket] if args.rocket is not None else None,
    )
    if args.start is None:
        args.start = default_start
    if args.end is None:
        args.end = default_end
    if args.sites is None:
        args.sites = default_sites(args.mission)
    validate_color_and_sites(ap, args.mission, args.color, args.sites)

    try:
        parse_hhmmss_fractional(args.start)
        parse_hhmmss_fractional(args.end)
    except ValueError as exc:
        ap.error(str(exc))
    if args.step <= 0:
        ap.error("--step must be > 0")
    if args.green_alt is not None and args.green_alt <= 0:
        ap.error("--green-alt must be > 0")

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    selected_sites = set(s.upper() for s in args.sites)
    skymaps = load_skymaps(selected_sites, color=args.color, mission=args.mission, green_alt=args.green_alt)

    build_overlap_masks(skymaps)

    tiff_candidates = {}
    tiff_metadata = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site in selected_sites:
            tiff_candidates[site] = get_site_tiff_candidates(site, args.date, args.color, mission=args.mission)

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED
    for site in ["ARV", "VEE", "BVR"]:
        if site in tiff_candidates:
            tiff_metadata[site] = build_tiff_metadata(tiff_candidates[site], frame_interval)

    traj_configs = trajectory_config_tuples(args.mission, date=args.date)
    traj_lookups = {
        key: build_traj_lookup(str(path), color=args.color, green_alt=args.green_alt)
        for key, _tag, _label, path in traj_configs
    }
    traj_tags = {key: tag for key, tag, _label, _path in traj_configs}
    traj_labels = {key: label for key, _tag, label, _path in traj_configs}

    out_path = make_output_path(args.mission, args.date, args.start, args.end, args.step, color=args.color, green_alt=args.green_alt)
    if not out_path.is_absolute():
        out_path = mission_output_dir(args.mission, color=args.color, date=args.date) / out_path

    if args.no_csv:
        if args.no_plot:
            ap.error("--no-csv cannot be combined with --no-plot")
        csv_path = find_reusable_csv(out_path, series_prefix(args.mission), args.date, args.start, args.end, args.step)
        rows = load_rows_from_csv(
            csv_path,
            [
                "time",
                *([field for key in traj_lookups for field in (f"{key}_frame_site", f"{key}_brightness")]),
            ],
        )
        requested_times = set(build_requested_iso_times(args.date, args.start, args.end, args.step))
        rows = [row for row in rows if row.get("time") in requested_times]
        plot_times = [dt.datetime.fromisoformat(row["time"]) for row in rows if row.get("time")]
        plot_series = {
            key: [
                float(row[f"{key}_reference_norm_brightness"])
                if args.mission == "GIRAFF" and row.get(f"{key}_reference_norm_brightness")
                else (float(row[f"{key}_brightness"]) if row.get(f"{key}_brightness") else None)
                for row in rows
            ]
            for key in traj_lookups
        }
        plot_output = make_plot_output_path(out_path)
        plot_title = f"Brightness vs Time ({args.color})"
        plot_brightness_timeseries(
            plot_times,
            plot_series,
            plot_output,
            plot_title,
            labels_by_key=traj_labels,
        )
        print(f"Plotted from existing CSV {csv_path}")
        return

    fieldnames = ["time"]
    for key in traj_lookups:
        fieldnames.extend(
            [
                f"{key}_frame_site",
                f"{key}_frame_time",
                f"{key}_rocket_lat",
                f"{key}_rocket_lon",
                f"{key}_rocket_alt_km",
                f"{key}_percentile",
                f"{key}_brightness",
            ]
        )
        if args.mission == "GIRAFF":
            fieldnames.append(f"{key}_reference_norm_brightness")
    rows = []
    plot_times = []
    plot_series = {key: [] for key in traj_lookups}
    reference_norm_limits = None
    reference_norm_time = None
    if args.mission == "GIRAFF":
        reference_norm_time = reference_normalization_time(args.mission, args.date, args.start)
        reference_norm_limits = compute_reference_norm_limits(
            skymaps,
            selected_sites,
            args.date,
            reference_norm_time,
            args.color,
            frame_interval,
            colorbar_scale="linear",
            mission=args.mission,
        )
    step_td = dt.timedelta(seconds=args.step)
    total_steps = count_steps(start_dt, end_dt, args.step)
    step_idx = 0
    t = start_dt
    while t <= end_dt:
        step_idx += 1
        time_arg = format_time_arg(t)
        print_progress(step_idx, total_steps, time_arg)
        imgs_raw = {}
        frame_info = {}

        for site in ["ARV", "VEE", "BVR"]:
            if site not in selected_sites:
                continue
            try:
                im_raw, site_frame_info = load_tiff_frame_with_metadata(
                    site,
                    tiff_metadata.get(site, []),
                    t,
                    frame_interval=frame_interval,
                )
                imgs_raw[site] = im_raw
                frame_info[site] = site_frame_info
            except Exception as exc:
                print(f"{time_arg} {site}: frame load failed: {exc}")

        if "PKR" in selected_sites:
            try:
                pkr_lookup_time = t.strftime("%H%M%S")
                url_pkr = closest_amisr_png_url("PKR", args.date, pkr_lookup_time, color=args.color)
                imgs_raw["PKR"] = retrieve_image(url_pkr)
                frame_info["PKR"] = {"site": "PKR", "frame_time": parse_frame_datetime_from_url(url_pkr).isoformat()}
            except Exception as exc:
                print(f"{time_arg} PKR: frame load failed: {exc}")

        plot_times.append(t)
        row = {"time": t.isoformat()}
        for key, traj_lookup in traj_lookups.items():
            lat, lon = lookup_traj_position(traj_lookup, time_arg)
            geo = lookup_traj_geodetic_position(traj_lookup, time_arg)
            sample = best_rocket_brightness(lat, lon, skymaps, imgs_raw) if lat is not None and lon is not None else None
            rocket_lat, rocket_lon, rocket_alt_km = geo
            reference_norm_brightness = (
                normalize_reference_brightness(sample["raw_brightness"], reference_norm_limits)
                if sample
                else None
            )
            plot_series[key].append(reference_norm_brightness if reference_norm_brightness is not None else (sample["raw_brightness"] if sample else None))
            row[f"{key}_frame_site"] = frame_info[sample["site"]]["site"] if sample and sample["site"] in frame_info else ""
            row[f"{key}_frame_time"] = frame_info[sample["site"]]["frame_time"] if sample and sample["site"] in frame_info else ""
            row[f"{key}_rocket_lat"] = f"{rocket_lat:.6f}" if rocket_lat is not None else ""
            row[f"{key}_rocket_lon"] = f"{rocket_lon:.6f}" if rocket_lon is not None else ""
            row[f"{key}_rocket_alt_km"] = f"{rocket_alt_km:.6f}" if rocket_alt_km is not None else ""
            row[f"{key}_percentile"] = f"{sample['percentile']:.3f}" if sample else ""
            row[f"{key}_brightness"] = f"{sample['raw_brightness']:.3f}" if sample else ""
            if args.mission == "GIRAFF":
                row[f"{key}_reference_norm_brightness"] = f"{reference_norm_brightness:.6f}" if reference_norm_brightness is not None else ""
        rows.append(row)
        t += step_td

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if not args.no_plot:
        plot_output = make_plot_output_path(out_path)
        plot_title = f"Brightness vs Time ({args.color})"
        plot_brightness_timeseries(
            plot_times,
            plot_series,
            plot_output,
            plot_title,
            labels_by_key=traj_labels,
        )


if __name__ == "__main__":
    main()
