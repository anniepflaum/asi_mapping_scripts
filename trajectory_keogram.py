#!/usr/bin/env python3
"""Build brightness keograms along the left and right trajectories."""

import argparse
import datetime as dt
from pathlib import Path

import check_intersect as ci
import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.spatial import cKDTree

from asi_time_utils import format_time_label, parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from brightness_vs_time_dataset import format_time_arg
from map_asi_archive import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED, load_skymaps
from tiff_utils import build_tiff_metadata, get_site_tiff_candidates, load_best_frame_from_cached_tiffs
from traj_utils import build_traj_lookup, get_launch_start_from_traj_csv, lookup_traj_position, resample_traj_by_distance


def build_output_path(output_arg, date_str, start, end, color):
    if output_arg:
        return Path(output_arg)
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    return Path(f"../mapped/{color}/trajectory_keogram_{color}_{date_str}_{start_tok}_{end_tok}.png")


def build_intersection_masks(skymaps):
    sites = list(skymaps.keys())
    for s0 in sites:
        red_sites = sites.copy()
        red_sites.remove(s0)
        skymaps[s0]["extra_masks"] = {}
        for s1 in red_sites:
            m0, _m1 = ci.calculate_masks(
                skymaps[s0]["site_lat"],
                skymaps[s0]["site_lon"],
                skymaps[s0]["azmt"],
                skymaps[s0]["elev"],
                skymaps[s1]["site_lat"],
                skymaps[s1]["site_lon"],
                skymaps[s1]["azmt"],
                skymaps[s1]["elev"],
            )
            skymaps[s0]["extra_masks"][s1] = m0


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


def build_traj_axis_projector(sample_lats, sample_lons):
    """Project mapped lat/lon positions onto the resampled along-track sample index."""
    mean_lat = np.deg2rad(np.nanmean(sample_lats))
    coords = np.column_stack((sample_lons * np.cos(mean_lat), sample_lats))
    return {
        "tree": cKDTree(coords),
        "cos_lat": np.cos(mean_lat),
    }


def project_position_to_traj_index(projector, lat, lon):
    """Return the nearest resampled trajectory sample index for a mapped position."""
    if lat is None or lon is None:
        return np.nan
    _distance, idx = projector["tree"].query([lon * projector["cos_lat"], lat], k=1)
    return float(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260210", help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=0.3, help="Step size in seconds")
    ap.add_argument("--samples", type=int, default=480, help="Number of equal-distance samples along each trajectory")
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
    build_intersection_masks(skymaps)
    samplers = {site: build_site_sampler(skymaps, site) for site in selected_sites if site in skymaps}

    tiff_overrides = {"ARV": args.arv_tiffs, "VEE": args.vee_tiffs, "BVR": args.bvr_tiffs}
    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED
    tiff_metadata = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site in selected_sites:
            candidates = get_site_tiff_candidates(site, args.date, args.color, tiff_overrides[site])
            tiff_metadata[site] = build_tiff_metadata(candidates, frame_interval)

    left_traj = build_traj_lookup("36.397_AttitudeSolution.csv")
    right_traj = build_traj_lookup("36.398_AttitudeSolution.csv")
    left_lats, left_lons, _left_s = resample_traj_by_distance(left_traj, args.samples)
    right_lats, right_lons, _right_s = resample_traj_by_distance(right_traj, args.samples)
    left_projector = build_traj_axis_projector(left_lats, left_lons)
    right_projector = build_traj_axis_projector(right_lats, right_lons)

    times = []
    left_cols = []
    right_cols = []
    left_line = []
    right_line = []
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
        left_pos = lookup_traj_position(left_traj, time_arg)
        right_pos = lookup_traj_position(right_traj, time_arg)
        left_cols.append(left_profile)
        right_cols.append(right_profile)
        left_line.append(project_position_to_traj_index(left_projector, *left_pos))
        right_line.append(project_position_to_traj_index(right_projector, *right_pos))
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
    extent = [mdates.date2num(times[0]), mdates.date2num(times[-1]), 0, args.samples]
    log_norm = mpl.colors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, max(vmin, 1e-6) * 1.0001))

    panels = [
        (
            axes[0],
            left_img,
            np.asarray(left_line, dtype=float),
            "36.397 Trajectory Keogram",
            get_launch_start_from_traj_csv("36.397_AttitudeSolution.csv"),
        ),
        (
            axes[1],
            right_img,
            np.asarray(right_line, dtype=float),
            "36.398 Trajectory Keogram",
            get_launch_start_from_traj_csv("36.398_AttitudeSolution.csv"),
        ),
    ]
    image_handle = None
    time_nums = mdates.date2num(times)
    for ax, img, line_y, title, launch_start in panels:
        image_handle = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            norm=log_norm,
        )
        t0_label = format_time_label(launch_start) if launch_start else "unknown"
        ax.set_title(f"{title} | T0 {t0_label}")
        ax.set_ylabel("Distance-Along-Trajectory Sample")
        valid_line = np.isfinite(line_y)
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

    axes[-1].set_xlabel("Time UTC")
    minute_locator = mdates.MinuteLocator(interval=1)
    axes[-1].xaxis.set_major_locator(minute_locator)
    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: mdates.num2date(value).strftime("%H:%M")
            if mdates.num2date(value).minute % 5 == 0
            else ""
        )
    )

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
