#!/usr/bin/env python3
"""Build latitude-sampled trajectory brightness CSVs."""

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.missions import (
    default_sites,
    default_time_range,
    mission_output_dir,
    resolve_mission_and_date,
    rocket_launch_start,
    ROCKET_TIME_WINDOWS,
    trajectory_config_tuples,
    validate_color_and_sites,
)
from core.skymaps import load_skymaps
from core.time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates, load_best_frame_from_cached_tiffs
from core.traj_utils import build_traj_lookup
from traj_brightness_series import format_time_arg
from traj_keogram import build_combined_profile, build_site_sampler


def csv_prefix(mission, color):
    prefix = "trajectory_lat_samples" if str(mission).upper() == "GNEISS" else f"{str(mission).upper()}_trajectory_lat_samples"
    return f"{prefix}_{color}"


def build_csv_output_path(mission, date_str, start, end, step, color):
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    return mission_output_dir(mission, color=color, date=date_str) / f"{csv_prefix(mission, color)}_{date_str}_{start_tok}_{end_tok}_step{step_tok}.csv"


def build_rocket_csv_output_path(mission, date_str, rocket_tag, start, end, step, color):
    output_path = build_csv_output_path(mission, date_str, start, end, step, color)
    return output_path.with_name(f"{output_path.stem}_{rocket_tag}{output_path.suffix}")


def parse_lat_sample_filename(csv_path, prefix):
    pattern = (
        rf"^{re.escape(prefix)}_(\d{{8}})_"
        rf"(\d{{6}}(?:p\d+)?)_(\d{{6}}(?:p\d+)?)_step(\d+(?:p\d+)?)"
        rf"(?:_([^.]+))?\.csv$"
    )
    match = re.match(pattern, csv_path.name)
    if not match:
        return None
    date_tok, start_tok, end_tok, step_tok, suffix = match.groups()
    return {
        "date": date_tok,
        "start_time": start_tok.replace("p", "."),
        "end_time": end_tok.replace("p", "."),
        "step": float(step_tok.replace("p", ".")),
        "suffix": suffix,
    }


def format_elapsed_seconds(seconds):
    return f"{float(seconds):.6f}".rstrip("0").rstrip(".")


def time_since_header_start(header_value, date, fallback_dt):
    prefix = "time since "
    if not str(header_value).startswith(prefix):
        return fallback_dt
    try:
        return parse_date_and_time(date, str(header_value)[len(prefix) :])
    except ValueError:
        return fallback_dt


def requested_time_tokens_since(base_dt, start_dt, end_dt, step):
    tokens = []
    t = start_dt
    step_td = dt.timedelta(seconds=step)
    while t <= end_dt:
        tokens.append(format_elapsed_seconds((t - base_dt).total_seconds()))
        t += step_td
    return tokens


def csv_contains_requested_time_tokens(csv_path, date, start_dt, end_dt, step, fallback_base_dt):
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or not header[0].startswith("time since "):
                return False
            base_dt = time_since_header_start(header[0], date, fallback_base_dt)
            requested_tokens = requested_time_tokens_since(base_dt, start_dt, end_dt, step)
            available = {row[0] for row in reader if row}
    except Exception:
        return False
    return all(token in available for token in requested_tokens)


def find_reusable_lat_sample_csvs(preferred_path, prefix, date, start, end, step, rocket=None):
    start_dt = parse_date_and_time(date, start)
    end_dt = parse_date_and_time(date, end)
    if (
        rocket is not None
        and preferred_path.exists()
        and csv_contains_requested_time_tokens(preferred_path, date, start_dt, end_dt, step, start_dt)
    ):
        return [preferred_path]

    candidates = []
    for candidate in preferred_path.parent.glob(f"{prefix}_*.csv"):
        parsed = parse_lat_sample_filename(candidate, prefix)
        if parsed is None or parsed["date"] != date:
            continue
        if rocket is not None and parsed.get("suffix") not in (None, str(rocket)):
            continue
        try:
            candidate_start_dt = parse_date_and_time(date, parsed["start_time"])
            candidate_end_dt = parse_date_and_time(date, parsed["end_time"])
        except ValueError:
            continue
        if candidate_start_dt > start_dt or candidate_end_dt < end_dt:
            continue
        if csv_contains_requested_time_tokens(candidate, date, start_dt, end_dt, step, candidate_start_dt):
            span_seconds = (candidate_end_dt - candidate_start_dt).total_seconds()
            candidates.append((span_seconds, parsed["step"], parsed.get("suffix") or "", candidate))
    if not candidates:
        return [preferred_path]
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].name))
    if rocket is not None:
        return [candidates[0][3]]

    best_by_suffix = {}
    for span_seconds, parsed_step, suffix, candidate in candidates:
        best_by_suffix.setdefault(suffix, (span_seconds, parsed_step, candidate))
    return [item[2] for _suffix, item in sorted(best_by_suffix.items())]


def rocket_window_for_tag(tag, fallback_start, fallback_end):
    rocket = rocket_id_from_tag(tag)
    return ROCKET_TIME_WINDOWS.get(str(rocket), (fallback_start, fallback_end))


def per_rocket_output_paths(output_path, traj_data):
    output_path = Path(output_path)
    if len(traj_data) == 1:
        return {traj_data[0]["tag"]: output_path}
    return {
        traj["tag"]: output_path.with_name(f"{output_path.stem}_{traj['tag']}{output_path.suffix}")
        for traj in traj_data
    }


def rocket_id_from_tag(tag):
    tag = str(tag)
    if tag.startswith("36") and tag[2:] in {"380", "381", "397", "398"}:
        return tag[2:]
    return tag


def rocket_launch_dt_for_tag(tag, date, fallback_dt):
    rocket = rocket_id_from_tag(tag)
    try:
        return parse_date_and_time(date, rocket_launch_start(rocket))
    except Exception:
        return fallback_dt


def resample_traj_by_latitude(traj_lookup, n_samples):
    """Resample a mapped trajectory to equally spaced latitude points."""
    lats = np.asarray(traj_lookup["lats"], dtype=float)
    lons = np.asarray(traj_lookup["lons"], dtype=float)
    valid = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[valid]
    lons = lons[valid]
    if lats.size < 2:
        raise ValueError("trajectory lookup has too few finite latitude samples")

    order = np.argsort(lats)
    lats = lats[order]
    lons = lons[order]
    unique_lats, unique_idx = np.unique(lats, return_index=True)
    if unique_lats.size < 2:
        raise ValueError("trajectory lookup has too few unique latitude samples")

    lons = lons[unique_idx]
    target_lats = np.linspace(float(unique_lats[0]), float(unique_lats[-1]), n_samples)
    target_lons = np.interp(target_lats, unique_lats, lons)
    return target_lats, target_lons


def save_lat_sample_csv(output_path, traj_data, times, start_dt):
    """Save per-rocket CSVs: time rows with latitude brightness columns."""
    output_paths = per_rocket_output_paths(output_path, traj_data)
    for traj in traj_data:
        rocket_output_path = output_paths[traj["tag"]]
        rocket_output_path.parent.mkdir(parents=True, exist_ok=True)
        header = [f"time since {format_time_arg(start_dt)}"] + [f"{float(lat):.8f}" for lat in traj["lats"]]
        brightness = np.asarray(traj["img"], dtype=float)
        with open(rocket_output_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for time_idx, sample_time in enumerate(times):
                values = [
                    "" if not np.isfinite(value) else f"{float(value):.8g}"
                    for value in brightness[:, time_idx]
                ]
                writer.writerow([format_elapsed_seconds((sample_time - start_dt).total_seconds())] + values)
        print(f"Saved latitude-sampled CSV to {rocket_output_path}")
    return output_paths


def save_single_lat_sample_csv(output_path, traj, times, start_dt):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"time since {format_time_arg(start_dt)}"] + [f"{float(lat):.8f}" for lat in traj["lats"]]
    brightness = np.asarray(traj["img"], dtype=float)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for time_idx, sample_time in enumerate(times):
            values = [
                "" if not np.isfinite(value) else f"{float(value):.8g}"
                for value in brightness[:, time_idx]
            ]
            writer.writerow([format_elapsed_seconds((sample_time - start_dt).total_seconds())] + values)
    print(f"Saved latitude-sampled CSV to {output_path}")
    return output_path


def load_lat_sample_csv(csv_path, start_dt, date, rocket_hint=None):
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header or len(header) < 2 or not header[0].startswith("time since "):
            raise ValueError(f"{csv_path} is not a latitude-sampled CSV")
        base_dt = time_since_header_start(header[0], date, start_dt)
        lats = np.asarray([float(value) for value in header[1:]], dtype=float)
        rows = []
        times = []
        for row in reader:
            if not row:
                continue
            times.append(base_dt + dt.timedelta(seconds=float(row[0])))
            values = [np.nan if value == "" else float(value) for value in row[1 : len(header)]]
            if len(values) < lats.size:
                values.extend([np.nan] * (lats.size - len(values)))
            rows.append(values[: lats.size])
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")
    parsed = None
    for prefix in (
        "trajectory_lat_samples_green",
        "trajectory_lat_samples_red",
        "GIRAFF_trajectory_lat_samples_green",
        "GIRAFF_trajectory_lat_samples_red",
    ):
        parsed = parse_lat_sample_filename(Path(csv_path), prefix)
        if parsed is not None:
            break
    rocket = (parsed.get("suffix") if parsed else None) or rocket_hint or Path(csv_path).stem
    launch_dt = rocket_launch_dt_for_tag(rocket, date, start_dt)
    return times, {
        "tag": rocket,
        "launch_dt": launch_dt,
        "lats": lats,
        "img": np.asarray(rows, dtype=float).T,
    }


def save_minute_marker_plots(output_path, traj_data, times, start_dt, end_dt, color):
    """Plot brightness along each trajectory at each full minute after each rocket T0."""
    output_paths = per_rocket_output_paths(output_path, traj_data)

    for traj in traj_data:
        launch_dt = traj.get("launch_dt") or start_dt
        minute_offsets = []
        offset = 60.0
        while launch_dt + dt.timedelta(seconds=offset) <= end_dt + dt.timedelta(seconds=1e-9):
            if launch_dt + dt.timedelta(seconds=offset) >= start_dt - dt.timedelta(seconds=1e-9):
                minute_offsets.append(offset)
            offset += 60.0
        if not minute_offsets:
            print(f"{traj['tag']}: no nonzero minute markers fall inside the requested time range; skipped minute-marker plot.")
            continue

        time_offsets = np.asarray([(time - launch_dt).total_seconds() for time in times], dtype=float)
        brightness = np.asarray(traj["img"], dtype=float)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for offset in minute_offsets:
            time_idx = int(np.argmin(np.abs(time_offsets - offset)))
            ax.plot(
                traj["lats"],
                brightness[:, time_idx],
                linewidth=1.2,
                label=f"T+{offset / 60:.0f} min",
            )

        ax.set_title(f"{traj['tag']} Brightness Along Trajectory at Minute Markers")
        ax.set_xlabel("Latitude")
        ax.set_ylabel(f"{color.capitalize()} Channel Intensity")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Time")

        rocket_csv_path = output_paths[traj["tag"]]
        plot_path = rocket_csv_path.with_name(f"{rocket_csv_path.stem}_minute_markers.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved minute-marker plot to {plot_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the mission date and default time window")
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=0.3, help="Step size in seconds")
    ap.add_argument("--samples", type=int, default=480, help="Number of equal-latitude samples along each trajectory")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to merge into the brightness samples")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--no-csv", action="store_true", help="Skip CSV generation and make minute-marker plots from an existing CSV covering the requested range")
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

    if args.no_csv:
        traj_configs = trajectory_config_tuples(args.mission, date=args.date)
        if args.rocket is not None:
            traj_configs = [
                cfg
                for cfg in traj_configs
                if cfg[2].replace(".", "").endswith(args.rocket) or str(cfg[1]).endswith(args.rocket)
            ]
            if not traj_configs:
                ap.error(f"No trajectory is configured for --rocket {args.rocket}")

        csv_paths = []
        csv_plot_ranges = {}
        for _key, tag, _label, _path in traj_configs:
            rocket_start, rocket_end = rocket_window_for_tag(tag, args.start, args.end)
            rocket_start_dt = max(parse_date_and_time(args.date, rocket_start), start_dt)
            rocket_end_dt = min(parse_date_and_time(args.date, rocket_end), end_dt)
            if rocket_end_dt < rocket_start_dt:
                continue
            rocket_start_arg = format_time_arg(rocket_start_dt)
            rocket_end_arg = format_time_arg(rocket_end_dt)
            rocket_output_path = build_rocket_csv_output_path(
                args.mission,
                args.date,
                tag,
                rocket_start_arg,
                rocket_end_arg,
                args.step,
                args.color,
            )
            found = find_reusable_lat_sample_csvs(
                rocket_output_path,
                csv_prefix(args.mission, args.color),
                args.date,
                rocket_start_arg,
                rocket_end_arg,
                args.step,
                rocket=tag,
            )
            csv_paths.extend(found)
            for path in found:
                csv_plot_ranges[path] = (rocket_start_dt, rocket_end_dt, tag)

        missing = [path for path in csv_paths if not path.exists()]
        if missing:
            ap.error(f"No reusable latitude-sampled CSV found for requested range; expected {missing[0]}")
        for csv_path in csv_paths:
            rocket_start_dt, rocket_end_dt, tag = csv_plot_ranges[csv_path]
            csv_times, csv_traj = load_lat_sample_csv(csv_path, rocket_start_dt, args.date, rocket_hint=tag)
            save_minute_marker_plots(csv_path, [csv_traj], csv_times, rocket_start_dt, rocket_end_dt, args.color)
        print("Plotted from existing CSV " + ", ".join(str(path) for path in csv_paths))
        return

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
        lats, lons = resample_traj_by_latitude(lookup, args.samples)
        traj_data.append(
            {
                "key": key,
                "tag": tag,
                "label": label,
                "path": path,
                "launch_dt": rocket_launch_dt_for_tag(tag, args.date, start_dt),
                "lats": lats,
                "lons": lons,
                "cols": [],
            }
        )

    for traj in traj_data:
        rocket_start, rocket_end = rocket_window_for_tag(traj["tag"], args.start, args.end)
        rocket_start_dt = parse_date_and_time(args.date, rocket_start)
        rocket_end_dt = parse_date_and_time(args.date, rocket_end)
        if rocket_start_dt < start_dt:
            rocket_start_dt = start_dt
            rocket_start = args.start
        if rocket_end_dt > end_dt:
            rocket_end_dt = end_dt
            rocket_end = args.end
        if rocket_end_dt < rocket_start_dt:
            print(f"{traj['tag']}: requested range does not overlap rocket window; skipped.")
            continue

        traj["cols"] = []
        times = []
        step_td = dt.timedelta(seconds=args.step)
        t = rocket_start_dt
        frame_idx = 0
        while t <= rocket_end_dt:
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

            profile, _site = build_combined_profile(traj["lats"], traj["lons"], imgs_raw, samplers, selected_sites)
            traj["cols"].append(profile)
            times.append(t)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"{traj['tag']}: processed {frame_idx} frames through {time_arg}")
            t += step_td

        traj["img"] = np.column_stack(traj["cols"])
        rocket_output_path = build_rocket_csv_output_path(
            args.mission,
            args.date,
            traj["tag"],
            rocket_start,
            rocket_end,
            args.step,
            args.color,
        )
        save_single_lat_sample_csv(rocket_output_path, traj, times, rocket_start_dt)
        save_minute_marker_plots(rocket_output_path, [traj], times, rocket_start_dt, rocket_end_dt, args.color)


if __name__ == "__main__":
    main()
