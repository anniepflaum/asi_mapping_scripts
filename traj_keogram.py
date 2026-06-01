#!/usr/bin/env python3
"""Build brightness keograms along mission trajectories."""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.missions import (
    default_sites,
    default_time_range,
    mission_output_dir,
    resolve_mission_and_date,
    trajectory_config_tuples,
    validate_color_and_sites,
)
from core.skymaps import load_skymaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from core.time_utils import format_time_label, parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from traj_brightness_series import format_time_arg
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates, load_best_frame_from_cached_tiffs
from core.traj_utils import build_traj_lookup, get_launch_start_from_traj_csv, resample_traj_by_time


GNEISS_MAGLAT_CSV = mission_output_dir("GNEISS", color="green") / "tg_to_maglat.csv"


def build_output_path(output_arg, mission, date_str, start, end, color):
    if output_arg:
        return Path(output_arg)
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    prefix = "trajectory_keogram" if mission == "GNEISS" else f"{mission}_trajectory_keogram"
    return mission_output_dir(mission, color=color, date=date_str) / f"{prefix}_{color}_{date_str}_{start_tok}_{end_tok}.png"


def build_data_output_path(data_output_arg, output_path):
    if data_output_arg:
        return Path(data_output_arg)
    return Path(output_path).with_suffix(".npz")


def build_maglat_output_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_maglat{output_path.suffix}")


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
    """Return flight-time y-bounds with launch fixed at 0 seconds."""
    traj_times = np.asarray(traj_lookup["times"], dtype=float)
    if traj_times.size == 0:
        raise ValueError("trajectory lookup has no time samples")
    y_min = 0.0
    y_max = float(traj_times[-1])
    req_end = (end_dt - launch_dt).total_seconds()
    if y_min <= req_end <= y_max:
        y_max = req_end
    return y_min, y_max


def tg_reference_datetime(mission, date):
    """Return the mission TG reference time used for x-axis elapsed seconds."""
    if mission == "GNEISS":
        tg_start, _tg_end = default_time_range(mission, date, rocket_tags=["397", "398"])
    else:
        tg_start, _tg_end = default_time_range(mission, date)
    return parse_date_and_time(date, tg_start)


def load_gneiss_maglat_lookup(csv_path=GNEISS_MAGLAT_CSV):
    """Load TG-relative magnetic latitude series for each GNEISS rocket."""
    lookup = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"time_since_TG_s", "397_magnetic_lat_deg", "398_magnetic_lat_deg"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{csv_path} missing column(s): {', '.join(missing)}")
        rows = list(reader)
    for tag in ("397", "398"):
        times = []
        maglats = []
        field = f"{tag}_magnetic_lat_deg"
        for row in rows:
            if row.get("time_since_TG_s") and row.get(field):
                times.append(float(row["time_since_TG_s"]))
                maglats.append(float(row[field]))
        if not times:
            raise ValueError(f"{csv_path} has no magnetic latitude samples for rocket {tag}")
        lookup[tag] = (np.asarray(times, dtype=float), np.asarray(maglats, dtype=float))
    return lookup


def interpolate_maglat_arrays(time_since_tg, tags, maglat_lookup):
    """Interpolate magnetic latitude onto keogram times for each rocket."""
    arrays = {}
    for tag in tags:
        source_times, source_maglats = maglat_lookup[str(tag)]
        values = np.full(time_since_tg.shape, np.nan, dtype=float)
        valid = (time_since_tg >= source_times[0]) & (time_since_tg <= source_times[-1])
        values[valid] = np.interp(time_since_tg[valid], source_times, source_maglats)
        arrays[str(tag)] = values
    return arrays


def save_keogram_data(
    data_output_path,
    args,
    panels,
    time_since_tg,
    x_min,
    x_max,
    vmin,
    vmax,
    source_data_files,
    maglat_by_tag=None,
):
    """Save the plotted keogram arrays so another script can redraw the panel."""
    data = {
        "mission": np.asarray(args.mission),
        "date": np.asarray(args.date),
        "color": np.asarray(args.color),
        "start": np.asarray(args.start),
        "end": np.asarray(args.end),
        "time_since_tg_s": np.asarray(time_since_tg, dtype=float),
        "x_limits_s": np.asarray([x_min, x_max], dtype=float),
        "brightness_limits": np.asarray([max(vmin, 1e-6), max(vmax, max(vmin, 1e-6) * 1.0001)], dtype=float),
        "tags": np.asarray([str(panel[-1]) for panel in panels]),
        "metadata_json": np.asarray(json.dumps({"source_data_file": source_data_files})),
    }
    for _ax, img, flight_times, line_y, y_bounds, _title, _launch_start, tag in panels:
        tag = str(tag)
        y_min, y_max = y_bounds
        start_idx = int(np.searchsorted(flight_times, y_min, side="left"))
        end_idx = int(np.searchsorted(flight_times, y_max, side="right"))
        start_idx = max(0, min(start_idx, len(flight_times) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(flight_times)))
        data[f"brightness_{tag}"] = np.asarray(img[start_idx:end_idx, :], dtype=float)
        data[f"flight_time_{tag}_s"] = np.asarray(flight_times[start_idx:end_idx], dtype=float)
        data[f"trajectory_line_{tag}_s"] = np.asarray(line_y, dtype=float)
        data[f"y_limits_{tag}_s"] = np.asarray([y_min, y_max], dtype=float)
        if maglat_by_tag is not None and tag in maglat_by_tag:
            data[f"magnetic_latitude_{tag}_deg"] = np.asarray(maglat_by_tag[tag], dtype=float)

    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_output_path, **data)
    print(f"Saved keogram data to {data_output_path}")


def plot_maglat_keogram(output_path, panels, maglat_by_tag, log_norm, color):
    """Plot GNEISS keograms with magnetic latitude on the x-axis."""
    fig_height = 5 if len(panels) == 1 else 5 * len(panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, fig_height), sharex=True, constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    trajectory_line_colors = {
        "397": "tab:blue",
        "398": "tab:orange",
    }
    image_handle = None
    for ax, panel in zip(axes, panels):
        _time_ax, img, flight_times, line_y, y_bounds, title, _launch_start, tag = panel
        tag = str(tag)
        maglats = maglat_by_tag[tag]
        y_min, y_max = y_bounds
        start_idx = int(np.searchsorted(flight_times, y_min, side="left"))
        end_idx = int(np.searchsorted(flight_times, y_max, side="right"))
        start_idx = max(0, min(start_idx, len(flight_times) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(flight_times)))
        valid_cols = np.isfinite(maglats)
        image_handle = ax.pcolormesh(
            maglats[valid_cols],
            flight_times[start_idx:end_idx],
            img[start_idx:end_idx, :][:, valid_cols],
            cmap="Greens",
            norm=log_norm,
            shading="auto",
        )
        valid_line = valid_cols & np.isfinite(line_y) & (line_y >= y_min) & (line_y <= y_max)
        if np.any(valid_line):
            ax.plot(
                maglats[valid_line],
                line_y[valid_line],
                color=trajectory_line_colors.get(tag, "#ff3b30"),
                linewidth=1.4,
                alpha=1.0,
                label=f"{tag} trajectory",
            )
            ax.legend(loc="upper right")
        ax.set_title(title)
        ax.set_ylabel("Flight Time Since Launch (s)")
        ax.set_ylim(float(y_min), float(y_max))

    axes[-1].set_xlabel("Magnetic Latitude (deg)")
    cbar = fig.colorbar(image_handle, ax=axes, orientation="vertical", shrink=0.95)
    cbar.set_label(f"{color.capitalize()} Channel Intensity")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved magnetic-latitude keogram to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the mission date and default time window")
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=0.3, help="Step size in seconds")
    ap.add_argument("--samples", type=int, default=480, help="Number of equal-time samples along each trajectory")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to merge into the keogram")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output PNG path")
    ap.add_argument("--data-output", default=None, help="Output NPZ path for redrawable keogram data; defaults to the PNG path with .npz")
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
    if args.samples < 2:
        ap.error("--samples must be at least 2")

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    selected_sites = [s.upper() for s in args.sites]
    skymaps = load_skymaps(set(selected_sites), color=args.color, mission=args.mission)
    build_overlap_masks(skymaps)
    samplers = {site: build_site_sampler(skymaps, site) for site in selected_sites if site in skymaps}

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED
    tiff_metadata = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site in selected_sites:
            candidates = get_site_tiff_candidates(site, args.date, args.color, mission=args.mission)
            tiff_metadata[site] = build_tiff_metadata(candidates, frame_interval)

    traj_configs = trajectory_config_tuples(args.mission, date=args.date)
    if args.rocket is not None:
        traj_configs = [
            cfg
            for cfg in traj_configs
            if cfg[2].replace(".", "").endswith(args.rocket) or str(cfg[1]).endswith(args.rocket)
        ]
        if not traj_configs:
            ap.error(f"No trajectory is configured for --rocket {args.rocket}")
    traj_data = []
    for key, tag, label, path in traj_configs:
        lookup = build_traj_lookup(str(path), color=args.color)
        lats, lons, flight_times = resample_traj_by_time(lookup, args.samples)
        launch_start = get_launch_start_from_traj_csv(str(path))
        launch_dt = parse_date_and_time(args.date, launch_start) if launch_start else None
        traj_data.append(
            {
                "key": key,
                "tag": tag,
                "label": label,
                "path": path,
                "lookup": lookup,
                "lats": lats,
                "lons": lons,
                "flight_times": flight_times,
                "launch_start": launch_start,
                "launch_dt": launch_dt,
                "cols": [],
            }
        )
    if args.mission == "GNEISS":
        plot_order = {"398": 0, "397": 1}
        traj_data.sort(key=lambda traj: plot_order.get(str(traj["tag"]), 99))

    times = []
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

        for traj in traj_data:
            profile, _site = build_combined_profile(traj["lats"], traj["lons"], imgs_raw, samplers, selected_sites)
            traj["cols"].append(profile)
        times.append(t)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames through {time_arg}")
        t += step_td

    for traj in traj_data:
        traj["img"] = np.column_stack(traj["cols"])
    finite_arrays = [traj["img"][np.isfinite(traj["img"])] for traj in traj_data]
    all_vals = np.concatenate(finite_arrays)
    if all_vals.size == 0:
        raise ValueError("No valid keogram brightness samples were found")
    pos_vals = all_vals[all_vals > 0]
    if pos_vals.size > 0:
        vmin = float(np.percentile(pos_vals, 1))
        vmax = float(np.percentile(pos_vals, 99))
    else:
        vmin = float(np.percentile(all_vals, 1))
        vmax = float(np.percentile(all_vals, 99))

    output_path = build_output_path(args.output, args.mission, args.date, args.start, args.end, args.color)
    fig_height = 5 if len(traj_data) == 1 else 5 * len(traj_data)
    fig, axes = plt.subplots(len(traj_data), 1, figsize=(14, fig_height), sharex=True, constrained_layout=True)
    if len(traj_data) == 1:
        axes = [axes]
    log_norm = mpl.colors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, max(vmin, 1e-6) * 1.0001))

    panels = []
    for ax, traj in zip(axes, traj_data):
        if traj["launch_dt"] is None:
            line_y = np.full(len(times), np.nan, dtype=float)
            y_bounds = (0.0, float(traj["flight_times"][-1]))
        else:
            line_y = build_flight_time_line(times, traj["launch_dt"], traj["lookup"])
            y_bounds = compute_flight_time_bounds(start_dt, end_dt, traj["launch_dt"], traj["lookup"])
        panels.append(
            (
                ax,
                traj["img"],
                traj["flight_times"],
                line_y,
                y_bounds,
                f"{traj['tag']} Trajectory Keogram",
                traj["launch_start"],
                traj["tag"],
            )
        )
    image_handle = None
    tg_dt = tg_reference_datetime(args.mission, args.date)
    time_since_tg = np.asarray([(time - tg_dt).total_seconds() for time in times], dtype=float)
    x_min = (start_dt - tg_dt).total_seconds()
    x_max = (end_dt - tg_dt).total_seconds()
    if x_min == x_max:
        pad = max(args.step, 1.0)
        x_min -= pad / 2.0
        x_max += pad / 2.0
    trajectory_line_colors = {
        "397": "tab:blue",
        "398": "tab:orange",
    }
    for ax, img, flight_times, line_y, y_bounds, title, launch_start, tag in panels:
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
        ax.set_title(f"{title}")
        ax.set_ylabel("Flight Time Since Launch (s)")
        valid_line = np.isfinite(line_y) & (line_y >= y_min) & (line_y <= y_max)
        if np.any(valid_line):
            ax.plot(
                time_since_tg[valid_line],
                line_y[valid_line],
                color=trajectory_line_colors.get(str(tag), "#ff3b30"),
                linewidth=1.4,
                alpha=1.0,
                label=f"{tag} trajectory",
            )
            ax.legend(loc="upper right")
        ax.set_ylim(float(y_min), float(y_max))

    axes[-1].set_xlabel("Time since TG (s)")

    cbar = fig.colorbar(image_handle, ax=axes, orientation="vertical", shrink=0.95)
    cbar.set_label(f"{args.color.capitalize()} Channel Intensity")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_output_path = build_data_output_path(args.data_output, output_path)
    maglat_by_tag = None
    if args.mission == "GNEISS":
        maglat_lookup = load_gneiss_maglat_lookup()
        maglat_by_tag = interpolate_maglat_arrays(time_since_tg, [panel[-1] for panel in panels], maglat_lookup)
    source_data_files = [Path(traj["path"]).name for traj in traj_data]
    save_keogram_data(
        data_output_path,
        args,
        panels,
        time_since_tg,
        x_min,
        x_max,
        vmin,
        vmax,
        source_data_files,
        maglat_by_tag=maglat_by_tag,
    )
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved keogram to {output_path}")
    if maglat_by_tag is not None:
        plot_maglat_keogram(build_maglat_output_path(output_path), panels, maglat_by_tag, log_norm, args.color)


if __name__ == "__main__":
    main()
