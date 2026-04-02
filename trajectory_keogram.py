#!/usr/bin/env python3
"""Build brightness keograms along the left and right trajectories."""

import argparse
import datetime as dt
from pathlib import Path

from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.paths import LEFT_TRAJECTORY_PATH, RIGHT_TRAJECTORY_PATH
from core.skymaps import load_skymaps
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.spatial import cKDTree

from core.time_utils import format_time_label, parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from brightness_vs_time import format_time_arg
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates, load_best_frame_from_cached_tiffs
from core.traj_utils import build_traj_lookup, get_launch_start_from_traj_csv, resample_traj_by_time


def build_output_path(output_arg, date_str, start, end, color):
    if output_arg:
        return Path(output_arg)
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    return Path(f"../mapped/{color}/trajectory_keogram_{color}_{date_str}_{start_tok}_{end_tok}.png")

def build_site_sampler(skymaps, site):
    lat_grid = skymaps[site]["lat"]
    lon_grid = skymaps[site]["lon"]
    valid = (~skymaps[site]["mask"]) & np.isfinite(lat_grid) & np.isfinite(lon_grid)
    for mask in skymaps[site].get("extra_masks", {}).values():
        valid &= ~mask
    flat_idx = np.flatnonzero(valid.ravel())
    if flat_idx.size == 0:
        raise ValueError(f"{site} has no valid mapped pixels after overlap masking")

    lat_valid = lat_grid.ravel()[flat_idx]
    lon_valid = lon_grid.ravel()[flat_idx]
    mean_lat = np.deg2rad(np.nanmean(lat_valid))
    coords = np.column_stack((lon_valid * np.cos(mean_lat), lat_valid))
    tree = cKDTree(coords)
    return {
        "flat_idx": flat_idx,
        "tree": tree,
        "cos_lat": np.cos(mean_lat),
        "k": min(25, flat_idx.size),
    }


def sample_profile_from_site(raw_img, site_sampler, sample_lats, sample_lons):
    coords = np.column_stack((sample_lons * site_sampler["cos_lat"], sample_lats))
    distances, idx = site_sampler["tree"].query(coords, k=site_sampler["k"])
    if site_sampler["k"] == 1:
        distances = distances[:, np.newaxis]
        idx = idx[:, np.newaxis]
    flat = raw_img.ravel()
    sampled = flat[site_sampler["flat_idx"][idx]]
    brightness = np.mean(sampled, axis=1).astype(float)
    mean_distance = np.mean(distances, axis=1).astype(float)
    return brightness, mean_distance


def build_combined_profile(sample_lats, sample_lons, imgs_raw, samplers, selected_sites):
    best_brightness = np.full(sample_lats.shape, np.nan, dtype=float)
    best_distance = np.full(sample_lats.shape, np.inf, dtype=float)
    best_site = np.full(sample_lats.shape, "", dtype=object)

    for site in selected_sites:
        if site not in imgs_raw or site not in samplers:
            continue
        brightness, distance = sample_profile_from_site(imgs_raw[site], samplers[site], sample_lats, sample_lons)
        take = np.isfinite(brightness) & (distance < best_distance)
        best_brightness[take] = brightness[take]
        best_distance[take] = distance[take]
        best_site[take] = site

    best_brightness[~np.isfinite(best_distance)] = np.nan
    return best_brightness, best_site


def build_flight_time_line(times, launch_dt, traj_lookup):
    """Return flight-time values for plotting, masked outside the trajectory file range."""
    traj_times = np.asarray(traj_lookup["times"], dtype=float)
    if traj_times.size == 0:
        return np.full(len(times), np.nan, dtype=float)
    line = np.array([(t - launch_dt).total_seconds() for t in times], dtype=float)
    valid = (line >= float(traj_times[0])) & (line <= float(traj_times[-1]))
    line[~valid] = np.nan
    return line


def compute_flight_time_bounds(start_dt, end_dt, launch_dt, traj_lookup):
    """Return flight-time y-bounds cropped independently by requested start/end when in-flight."""
    traj_times = np.asarray(traj_lookup["times"], dtype=float)
    if traj_times.size == 0:
        raise ValueError("trajectory lookup has no time samples")
    y_min = float(traj_times[0])
    y_max = float(traj_times[-1])
    req_start = (start_dt - launch_dt).total_seconds()
    req_end = (end_dt - launch_dt).total_seconds()
    if y_min <= req_start <= y_max:
        y_min = req_start
    if y_min <= req_end <= y_max:
        y_max = req_end
    return y_min, y_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260210", help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=0.3, help="Step size in seconds")
    ap.add_argument("--samples", type=int, default=480, help="Number of equal-time samples along each trajectory")
    ap.add_argument("--sites", nargs="*", default=["ARV", "BVR", "VEE"], help="Sites to merge into the keogram")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output PNG path")
    ap.add_argument("--arv-tiffs", nargs="*", default=None, help="Optional ARV TIFF folder(s)")
    ap.add_argument("--vee-tiffs", nargs="*", default=None, help="Optional VEE TIFF folder(s)")
    ap.add_argument("--bvr-tiffs", nargs="*", default=None, help="Optional BVR TIFF folder(s)")
    args = ap.parse_args()

    try:
        parse_hhmmss_fractional(args.start)
        parse_hhmmss_fractional(args.end)
    except ValueError as exc:
        ap.error(str(exc))
    if args.step <= 0:
        ap.error("--step must be > 0")
    if args.samples < 2:
        ap.error("--samples must be at least 2")

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    selected_sites = [s.upper() for s in args.sites]
    skymaps = load_skymaps(set(selected_sites), color=args.color)
    build_overlap_masks(skymaps)
    samplers = {site: build_site_sampler(skymaps, site) for site in selected_sites if site in skymaps}

    tiff_overrides = {"ARV": args.arv_tiffs, "VEE": args.vee_tiffs, "BVR": args.bvr_tiffs}
    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED
    tiff_metadata = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site in selected_sites:
            candidates = get_site_tiff_candidates(site, args.date, args.color, tiff_overrides[site])
            tiff_metadata[site] = build_tiff_metadata(candidates, frame_interval)

    left_traj = build_traj_lookup(str(LEFT_TRAJECTORY_PATH), color=args.color)
    right_traj = build_traj_lookup(str(RIGHT_TRAJECTORY_PATH), color=args.color)
    left_lats, left_lons, left_flight_times = resample_traj_by_time(left_traj, args.samples)
    right_lats, right_lons, right_flight_times = resample_traj_by_time(right_traj, args.samples)

    times = []
    left_cols = []
    right_cols = []
    step_td = dt.timedelta(seconds=args.step)
    t = start_dt
    frame_idx = 0
    while t <= end_dt:
        time_arg = format_time_arg(t)
        imgs_raw = {}
        for site in ["ARV", "VEE", "BVR"]:
            if site not in selected_sites:
                continue
            try:
                imgs_raw[site] = load_best_frame_from_cached_tiffs(
                    site,
                    tiff_metadata.get(site, []),
                    t,
                    frame_interval=frame_interval,
                )
            except Exception as exc:
                print(f"{time_arg} {site}: frame load failed: {exc}")

        left_profile, _left_site = build_combined_profile(left_lats, left_lons, imgs_raw, samplers, selected_sites)
        right_profile, _right_site = build_combined_profile(right_lats, right_lons, imgs_raw, samplers, selected_sites)
        left_cols.append(left_profile)
        right_cols.append(right_profile)
        times.append(t)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames through {time_arg}")
        t += step_td

    left_img = np.column_stack(left_cols)
    right_img = np.column_stack(right_cols)
    all_vals = np.concatenate([left_img[np.isfinite(left_img)], right_img[np.isfinite(right_img)]])
    if all_vals.size == 0:
        raise ValueError("No valid keogram brightness samples were found")
    pos_vals = all_vals[all_vals > 0]
    if pos_vals.size > 0:
        vmin = float(np.percentile(pos_vals, 1))
        vmax = float(np.percentile(pos_vals, 99))
    else:
        vmin = float(np.percentile(all_vals, 1))
        vmax = float(np.percentile(all_vals, 99))

    output_path = build_output_path(args.output, args.date, args.start, args.end, args.color)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    left_launch_start = get_launch_start_from_traj_csv(str(LEFT_TRAJECTORY_PATH))
    right_launch_start = get_launch_start_from_traj_csv(str(RIGHT_TRAJECTORY_PATH))
    left_launch_dt = parse_date_and_time(args.date, left_launch_start)
    right_launch_dt = parse_date_and_time(args.date, right_launch_start)
    log_norm = mpl.colors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, max(vmin, 1e-6) * 1.0001))

    panels = [
        (
            axes[0],
            left_img,
            left_flight_times,
            build_flight_time_line(times, left_launch_dt, left_traj),
            compute_flight_time_bounds(start_dt, end_dt, left_launch_dt, left_traj),
            "36.397 Trajectory Keogram",
            left_launch_start,
        ),
        (
            axes[1],
            right_img,
            right_flight_times,
            build_flight_time_line(times, right_launch_dt, right_traj),
            compute_flight_time_bounds(start_dt, end_dt, right_launch_dt, right_traj),
            "36.398 Trajectory Keogram",
            right_launch_start,
        ),
    ]
    image_handle = None
    time_nums = mdates.date2num(times)
    x_min = mdates.date2num(start_dt)
    x_max = mdates.date2num(end_dt)
    for ax, img, flight_times, line_y, y_bounds, title, launch_start in panels:
        y_min, y_max = y_bounds
        start_idx = int(np.searchsorted(flight_times, y_min, side="left"))
        end_idx = int(np.searchsorted(flight_times, y_max, side="right"))
        start_idx = max(0, min(start_idx, len(flight_times) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(flight_times)))
        img_to_plot = img[start_idx:end_idx, :]
        flight_times_to_plot = flight_times[start_idx:end_idx]
        extent = [x_min, x_max, float(y_min), float(y_max)]
        image_handle = ax.imshow(
            img_to_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="Greens",
            norm=log_norm,
        )
        t0_label = format_time_label(launch_start) if launch_start else "unknown"
        ax.set_title(f"{title} | T0 {t0_label}")
        ax.set_ylabel("Flight Time Since Launch (s)")
        valid_line = np.isfinite(line_y) & (line_y >= y_min) & (line_y <= y_max)
        if np.any(valid_line):
            ax.plot(
                time_nums[valid_line],
                line_y[valid_line],
                color="#ff3b30",
                linewidth=1.4,
                alpha=1.0,
                label="Rocket trajectory",
            )
            ax.legend(loc="upper right")
        ax.set_ylim(float(y_min), float(y_max))

    axes[-1].set_xlabel("Time UTC")
    minute_locator = mdates.MinuteLocator(interval=1)
    axes[-1].xaxis.set_major_locator(minute_locator)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: mdates.num2date(value).strftime("%H:%M")))

    cbar = fig.colorbar(image_handle, ax=axes, orientation="vertical", shrink=0.95)
    cbar.set_label(f"{args.color.capitalize()} Channel Intensity")

    date_label = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    fig.suptitle(
        f"Trajectory Keogram\n"
        f"{date_label} {format_time_label(args.start)} to {format_time_label(args.end)}",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved keogram to {output_path}")


if __name__ == "__main__":
    main()
