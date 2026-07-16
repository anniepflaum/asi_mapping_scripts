#!/usr/bin/env python3
"""
Build a brightness-vs-time dataset for mission trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket position at that time.
3) Samples brightness at rocket position using the same logic as map_asi_archive.py.
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
from core.constants import DEFAULT_GREEN_ALT_KM, DEFAULT_RED_ALT_KM, FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.remote_data import load_pkr_image
from core.series_utils import (
    build_requested_iso_times,
    count_steps,
    find_reusable_csv,
    format_time_arg,
    load_rows_from_csv,
    load_tiff_frame_with_metadata,
    print_progress,
)
from core.skymaps import load_skymaps
from core.time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from core.missions import default_sites, default_time_range, mission_output_dir, resolve_mission_and_date, rocket_launch_datetime, rocket_time_window_datetimes, trajectory_config_tuples, validate_color_and_sites
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates
from core.traj_utils import (
    build_traj_lookup,
    lookup_traj_geodetic_position,
    lookup_traj_position,
)


def series_prefix(mission):
    return "brightness_vs_time" if str(mission).upper() == "GNEISS" else f"{str(mission).upper()}_brightness_vs_time"


GNEISS_MAGLAT_CSV = mission_output_dir("GNEISS", color="green") / "tg_to_maglat.csv"


def format_alt_token(alt_km):
    return str(float(alt_km)).replace(".", "p").rstrip("0").rstrip("p")


def nondefault_site_suffix(mission, sites):
    if not sites:
        return ""
    default_site_set = {site.upper() for site in default_sites(mission)}
    extra_sites = sorted({site.upper() for site in sites} - default_site_set)
    if not extra_sites:
        return ""
    return "_sites_" + "_".join(extra_sites)


def brightness_altitudes(color):
    return (DEFAULT_RED_ALT_KM,) if str(color).lower() == "red" else (DEFAULT_GREEN_ALT_KM,)


def alt_column_prefix(csv_key, alt_km):
    return f"{csv_key}_{format_alt_token(alt_km)}"


def brightness_plot_title(color):
    title = f"Brightness vs Time ({color})"
    if str(color).lower() == "green":
        title += f"\nGreen mapped altitude: {format_alt_token(DEFAULT_GREEN_ALT_KM)} km"
    return title


def make_output_path(mission, date, start, end, step, color="green", sites=None):
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    fit_suffix = "_FIT" if str(mission).upper() == "GIRAFF" else ""
    return Path(
        f"{series_prefix(mission)}_{date}_{start_tok}_{end_tok}_step{step_tok}"
        f"{nondefault_site_suffix(mission, sites)}{fit_suffix}.csv"
    )


def make_plot_output_path(csv_path):
    path = Path(csv_path)
    return path.with_suffix(".png")


def series_rocket_key(series_key):
    return str(series_key).split("_", 1)[0]


def plot_brightness_timeseries(times, series_by_key, output_path, title, labels_by_key=None, x_label="Time"):
    if not times:
        raise ValueError("No rows with iso_time were collected")
    rocket_order = []
    fig, axes = plt.subplots(figsize=(12, 5))
    axes = [axes]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    labels_by_key = labels_by_key or {}
    if rocket_order:
        grouped_items = [
            (rocket, [(key, values) for key, values in series_by_key.items() if series_rocket_key(key) == rocket])
            for rocket in rocket_order
        ]
    else:
        grouped_items = [(None, list(series_by_key.items()))]
    for ax, (rocket, items) in zip(axes, grouped_items):
        for idx, (key, values) in enumerate(items):
            label = labels_by_key.get(key, key)
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            ax.plot(times, values, linewidth=1.4, color=color, label=f"{label} brightness")
        ax.set_title(f"{title} | {rocket}" if rocket else title)
        ax.set_yscale("log")
        ax.set_ylabel("Brightness")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if times and isinstance(times[0], dt.datetime):
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel(x_label)
    if times and isinstance(times[0], dt.datetime):
        fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def csv_column_prefix(traj_key, traj_tags):
    tag = str(traj_tags.get(traj_key, traj_key))
    if tag.startswith("36") and tag[2:] in {"380", "381"}:
        return tag[2:]
    return tag


def first_existing_field(row, field_names):
    for field_name in field_names:
        if row.get(field_name):
            return row[field_name]
    return ""


def brightness_field_candidates(csv_key, traj_key, alt_km, normalized=False):
    suffix = "brightness"
    alt_prefix = alt_column_prefix(csv_key, alt_km)
    candidates = [f"{alt_prefix}_{suffix}"]
    if alt_km == DEFAULT_GREEN_ALT_KM:
        candidates.extend([f"{csv_key}_{suffix}", f"{traj_key}_{suffix}"])
    return candidates


def in_rocket_time_window(sample_dt, mission, rocket_tag):
    if str(mission).upper() != "GNEISS":
        return True
    start_dt, end_dt = rocket_time_window_datetimes(rocket_tag)
    return start_dt <= sample_dt <= end_dt


def suppress_outside_rocket_window(value, sample_dt, mission, rocket_tag):
    if value is None or str(mission).upper() != "GNEISS":
        return value
    if not in_rocket_time_window(sample_dt, mission, rocket_tag):
        return None
    return value


def load_gneiss_maglat_lookup(csv_path=GNEISS_MAGLAT_CSV):
    """Load TG-relative magnetic latitude series for the GNEISS rockets."""
    times_by_rocket = {"397": [], "398": []}
    maglats_by_rocket = {"397": [], "398": []}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"time_since_TG_s", "397_magnetic_lat_deg", "398_magnetic_lat_deg"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{csv_path} missing column(s): {', '.join(missing)}")
        for row in reader:
            tg_value = row.get("time_since_TG_s", "")
            if not tg_value:
                continue
            tg_time = float(tg_value)
            for rocket in ("397", "398"):
                maglat_value = row.get(f"{rocket}_magnetic_lat_deg", "")
                if maglat_value:
                    times_by_rocket[rocket].append(tg_time)
                    maglats_by_rocket[rocket].append(float(maglat_value))
    lookup = {}
    for rocket in ("397", "398"):
        if not times_by_rocket[rocket]:
            raise ValueError(f"{csv_path} has no magnetic latitude samples for rocket {rocket}")
        lookup[rocket] = (
            np.asarray(times_by_rocket[rocket], dtype=float),
            np.asarray(maglats_by_rocket[rocket], dtype=float),
        )
    return lookup


def interpolate_gneiss_maglat(maglat_lookup, rocket, tg_time):
    """Return magnetic latitude at TG time, or None outside the source range."""
    times, maglats = maglat_lookup[str(rocket)]
    if tg_time < times[0] or tg_time > times[-1]:
        return None
    return float(np.interp(tg_time, times, maglats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the ASI image date")
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=0.05, help="Step size in seconds")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to include")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
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

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    selected_sites = set(s.upper() for s in args.sites)
    alts = brightness_altitudes(args.color)
    skymaps_by_alt = {
        alt: load_skymaps(
            selected_sites,
            color=args.color,
            mission=args.mission,
        )
        for alt in alts
    }
    for skymaps in skymaps_by_alt.values():
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
    traj_lookups_by_alt = {
        alt: {
            key: build_traj_lookup(str(path), color=args.color)
            for key, _tag, _label, path in traj_configs
        }
        for alt in alts
    }
    traj_lookups = traj_lookups_by_alt[alts[0]]
    traj_tags = {key: tag for key, tag, _label, _path in traj_configs}
    traj_labels = {key: label for key, _tag, label, _path in traj_configs}
    csv_prefixes = {key: csv_column_prefix(key, traj_tags) for key in traj_lookups}

    out_path = make_output_path(args.mission, args.date, args.start, args.end, args.step, color=args.color, sites=args.sites)
    if not out_path.is_absolute():
        out_path = mission_output_dir(args.mission, color=args.color, date=args.date) / out_path

    if args.no_csv:
        if args.no_plot:
            ap.error("--no-csv cannot be combined with --no-plot")
        csv_path = find_reusable_csv(out_path, series_prefix(args.mission), args.date, args.start, args.end, args.step)
        rows = load_rows_from_csv(csv_path, ["time"])
        requested_times = set(build_requested_iso_times(args.date, args.start, args.end, args.step))
        rows = [row for row in rows if row.get("time") in requested_times]
        if args.mission == "GNEISS":
            gneiss_t0 = rocket_launch_datetime("397")
            plot_times = [
                float(row["TG"]) if row.get("TG") else (dt.datetime.fromisoformat(row["time"]) - gneiss_t0).total_seconds()
                for row in rows
                if row.get("time")
            ]
            x_label = "TG (s)"
        else:
            plot_times = [dt.datetime.fromisoformat(row["time"]) for row in rows if row.get("time")]
            x_label = "Time"
        plot_series = {}
        plot_labels = {}
        for key in traj_lookups:
            csv_key = csv_prefixes[key]
            for alt in brightness_altitudes(args.color):
                series_key = f"{csv_key}_{format_alt_token(alt)}"
                plot_labels[series_key] = f"{traj_labels.get(key, csv_key)} {format_alt_token(alt)} km"
                values = []
                for row in rows:
                    sample_dt = dt.datetime.fromisoformat(row["time"])
                    value = first_existing_field(row, brightness_field_candidates(csv_key, key, alt, normalized=False))
                    plot_value = float(value) if value else None
                    values.append(suppress_outside_rocket_window(plot_value, sample_dt, args.mission, csv_key))
                plot_series[series_key] = values
        plot_output = make_plot_output_path(out_path)
        plot_title = brightness_plot_title(args.color)
        plot_brightness_timeseries(
            plot_times,
            plot_series,
            plot_output,
            plot_title,
            labels_by_key=plot_labels,
            x_label=x_label,
        )
        print(f"Plotted from existing CSV {csv_path}")
        return

    fieldnames = ["time"]
    gneiss_t0 = None
    gneiss_maglat_lookup = None
    if args.mission == "GNEISS":
        fieldnames.append("TG")
        gneiss_t0 = rocket_launch_datetime("397")
        gneiss_maglat_lookup = load_gneiss_maglat_lookup()
    for key in traj_lookups:
        csv_key = csv_prefixes[key]
        fieldnames.extend(
            [
                f"{csv_key}_rocket_lat",
                f"{csv_key}_rocket_lon",
                f"{csv_key}_rocket_alt_km",
            ]
        )
        if args.mission == "GNEISS":
            fieldnames.append(f"{csv_key}_maglat")
        for alt in alts:
            alt_prefix = alt_column_prefix(csv_key, alt)
            fieldnames.append(f"{alt_prefix}_brightness")
    rows = []
    plot_times = []
    plot_series = {}
    plot_labels = {}
    for key in traj_lookups:
        csv_key = csv_prefixes[key]
        for alt in brightness_altitudes(args.color):
            series_key = f"{csv_key}_{format_alt_token(alt)}"
            plot_series[series_key] = []
            plot_labels[series_key] = f"{traj_labels.get(key, csv_key)} {format_alt_token(alt)} km"
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
                pkr_img, _pkr_source, _pkr_frame_dt = load_pkr_image(args.date, pkr_lookup_time, color=args.color, verbose=False)
                imgs_raw["PKR"] = pkr_img
            except Exception as exc:
                print(f"{time_arg} PKR: frame load failed: {exc}")

        plot_times.append(t)
        row = {"time": t.isoformat()}
        tg_time = None
        if args.mission == "GNEISS":
            tg_time = (t - gneiss_t0).total_seconds()
            row["TG"] = f"{tg_time:.6f}"
            plot_times[-1] = tg_time
        for key, traj_lookup in traj_lookups.items():
            csv_key = csv_prefixes[key]
            in_window = in_rocket_time_window(t, args.mission, csv_key)
            geo = lookup_traj_geodetic_position(traj_lookup, time_arg)
            rocket_lat, rocket_lon, rocket_alt_km = geo
            row[f"{csv_key}_rocket_lat"] = f"{rocket_lat:.6f}" if in_window and rocket_lat is not None else ""
            row[f"{csv_key}_rocket_lon"] = f"{rocket_lon:.6f}" if in_window and rocket_lon is not None else ""
            row[f"{csv_key}_rocket_alt_km"] = f"{rocket_alt_km:.6f}" if in_window and rocket_alt_km is not None else ""
            if args.mission == "GNEISS":
                maglat = interpolate_gneiss_maglat(gneiss_maglat_lookup, csv_key, tg_time)
                row[f"{csv_key}_maglat"] = f"{maglat:.8f}" if in_window and maglat is not None else ""
            for alt in alts:
                if in_window:
                    alt_lookup = traj_lookups_by_alt[alt][key]
                    lat, lon = lookup_traj_position(alt_lookup, time_arg)
                    sample = best_rocket_brightness(lat, lon, skymaps_by_alt[alt], imgs_raw) if lat is not None and lon is not None else None
                    outside_footprint = bool(sample and sample.get("outside_footprint", False))
                    if sample and not outside_footprint:
                        plot_value = sample["raw_brightness"]
                    else:
                        plot_value = None
                else:
                    sample = None
                    plot_value = None
                alt_prefix = alt_column_prefix(csv_key, alt)
                row[f"{alt_prefix}_brightness"] = f"{sample['raw_brightness']:.3f}" if sample and not outside_footprint else ""
                series_key = f"{csv_key}_{format_alt_token(alt)}"
                if series_key in plot_series:
                    plot_series[series_key].append(plot_value)
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
        plot_title = brightness_plot_title(args.color)
        plot_brightness_timeseries(
            plot_times,
            plot_series,
            plot_output,
            plot_title,
            labels_by_key=plot_labels,
            x_label="TG (s)" if args.mission == "GNEISS" else "Time",
        )


if __name__ == "__main__":
    main()
