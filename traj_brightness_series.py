#!/usr/bin/env python3
"""
Build a brightness-vs-time dataset for mission trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket position at that time.
3) Samples brightness at rocket position using the same logic as map_asi_archive.py.
4) Writes calibrated trajectory brightness arrays and metadata to HDF5.
"""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import h5py
import numpy as np

from core.brightness import best_rocket_brightness
from core.calibration import (
    BACKGROUND_EDGE_BUFFER_PX,
    calibration_metadata,
    calibrate_image_cached,
    calibration_factor,
    exposure_time_s,
    partition_calibrated_sites,
)
from core.constants import DEFAULT_GREEN_ALT_KM, DEFAULT_RED_ALT_KM, FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.series_utils import (
    count_steps,
    format_time_arg,
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
    title = f"Calibrated Brightness vs Time ({color})"
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
        f"{nondefault_site_suffix(mission, sites)}{fit_suffix}.h5"
    )


def make_plot_output_path(data_path):
    path = Path(data_path)
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
        ax.set_ylabel("Brightness (Rayleighs)")
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


def in_rocket_time_window(sample_dt, mission, rocket_tag):
    if str(mission).upper() != "GNEISS":
        return True
    start_dt, end_dt = rocket_time_window_datetimes(rocket_tag)
    return start_dt <= sample_dt <= end_dt


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


def row_float_array(rows, field):
    return np.asarray(
        [float(row[field]) if row.get(field) not in {None, ""} else np.nan for row in rows],
        dtype=float,
    )


def altitude_dataset_name(alt_km):
    return f"{format_alt_token(alt_km)}_km"


def compressed_dataset(group, name, values, units=None):
    values = np.asarray(values)
    options = {}
    if values.ndim > 0 and values.size > 0:
        options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    dataset = group.create_dataset(name, data=values, **options)
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def write_brightness_hdf5(path, args, rows, traj_configs, csv_prefixes, alts):
    """Write the calibrated trajectory series as structured HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    calibration = calibration_metadata(args.sites, args.color)

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "trajectory_brightness_series"
        h5.attrs["schema_version"] = "1.0"
        h5.attrs["mission"] = args.mission
        h5.attrs["date"] = args.date
        h5.attrs["color"] = args.color
        h5.attrs["start"] = args.start
        h5.attrs["end"] = args.end
        h5.attrs["step_s"] = float(args.step)
        h5.attrs["brightness_units"] = "Rayleighs"
        h5.attrs["selected_sites_json"] = json.dumps(args.sites)
        h5.attrs["calibration_json"] = json.dumps(calibration, sort_keys=True)

        time_iso = np.asarray([row["time"] for row in rows], dtype=object)
        h5.create_dataset("time_iso", data=time_iso, dtype=string_dtype)
        if args.mission == "GNEISS":
            compressed_dataset(h5, "time_since_tg_s", row_float_array(rows, "TG"), units="s")

        rockets_group = h5.create_group("rockets")
        for key, tag, label, source_path in traj_configs:
            if key not in csv_prefixes:
                continue
            rocket = csv_prefixes[key]
            group = rockets_group.create_group(rocket)
            group.attrs["trajectory_key"] = str(key)
            group.attrs["trajectory_tag"] = str(tag)
            group.attrs["label"] = str(label)
            group.attrs["source_trajectory_file"] = Path(source_path).name
            compressed_dataset(
                group,
                "geodetic_latitude_deg",
                row_float_array(rows, f"{rocket}_rocket_lat"),
                units="degrees_north",
            )
            compressed_dataset(
                group,
                "geodetic_longitude_deg",
                row_float_array(rows, f"{rocket}_rocket_lon"),
                units="degrees_east",
            )
            compressed_dataset(
                group,
                "geodetic_altitude_km",
                row_float_array(rows, f"{rocket}_rocket_alt_km"),
                units="km",
            )
            if args.mission == "GNEISS":
                compressed_dataset(
                    group,
                    "magnetic_latitude_deg",
                    row_float_array(rows, f"{rocket}_maglat"),
                    units="degrees",
                )
            brightness_group = group.create_group("brightness")
            for alt in alts:
                dataset = compressed_dataset(
                    brightness_group,
                    altitude_dataset_name(alt),
                    row_float_array(rows, f"{alt_column_prefix(rocket, alt)}_brightness"),
                    units="Rayleighs",
                )
                dataset.attrs["mapped_altitude_km"] = float(alt)

    print(f"Wrote {len(rows)} samples to {path}")


def load_hdf5_plot_series(path, mission, traj_lookups, csv_prefixes, traj_labels, alts):
    """Load plotting arrays from an existing trajectory-series HDF5 file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != "trajectory_brightness_series":
            raise ValueError(f"{path} is not a trajectory brightness HDF5 file")
        if h5.attrs.get("brightness_units") != "Rayleighs":
            raise ValueError(f"{path} does not contain Rayleigh-calibrated brightness")
        time_iso = [dt.datetime.fromisoformat(value) for value in h5["time_iso"].asstr()[:]]
        if mission == "GNEISS":
            plot_times = np.asarray(h5["time_since_tg_s"], dtype=float).tolist()
            x_label = "TG (s)"
        else:
            plot_times = time_iso
            x_label = "Time"
        plot_series = {}
        plot_labels = {}
        for key in traj_lookups:
            rocket = csv_prefixes[key]
            for alt in alts:
                series_key = f"{rocket}_{format_alt_token(alt)}"
                values = np.asarray(
                    h5[f"rockets/{rocket}/brightness/{altitude_dataset_name(alt)}"],
                    dtype=float,
                )
                plot_series[series_key] = values.tolist()
                plot_labels[series_key] = (
                    f"{traj_labels.get(key, rocket)} {format_alt_token(alt)} km"
                )
    return plot_times, plot_series, plot_labels, x_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the ASI image date")
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=0.05, help="Step size in seconds")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to include")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output HDF5 path")
    ap.add_argument("--no-plot", action="store_true", help="Write the HDF5 file only and skip the PNG plot")
    ap.add_argument(
        "--plot-existing",
        action="store_true",
        help="Skip data generation and plot from the requested existing HDF5 file",
    )
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
    calibrated_sites, unsupported_sites = partition_calibrated_sites(
        [site.upper() for site in args.sites],
        args.color,
    )
    if unsupported_sites:
        print(
            f"Skipping sites without {args.color} Rayleigh calibration: "
            f"{', '.join(unsupported_sites)}"
        )
    if not calibrated_sites:
        ap.error(f"none of the requested sites has a {args.color} Rayleigh calibration")
    args.sites = calibrated_sites
    for site in args.sites:
        print(
            f"{site}: calibration factor={calibration_factor(site, args.color):g} R s/count, "
            f"exposure={exposure_time_s(args.color):g}s"
        )

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

    selected_sites = set(args.sites)
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

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() not in {".h5", ".hdf5"}:
            out_path = out_path.with_suffix(".h5")
    else:
        out_path = make_output_path(
            args.mission,
            args.date,
            args.start,
            args.end,
            args.step,
            color=args.color,
            sites=args.sites,
        )
        out_path = mission_output_dir(args.mission, color=args.color, date=args.date) / out_path

    if args.plot_existing:
        if args.no_plot:
            ap.error("--plot-existing cannot be combined with --no-plot")
        plot_times, plot_series, plot_labels, x_label = load_hdf5_plot_series(
            out_path,
            args.mission,
            traj_lookups,
            csv_prefixes,
            traj_labels,
            alts,
        )
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
        print(f"Plotted from existing HDF5 {out_path}")
        return

    gneiss_t0 = None
    gneiss_maglat_lookup = None
    if args.mission == "GNEISS":
        gneiss_t0 = rocket_launch_datetime("397")
        gneiss_maglat_lookup = load_gneiss_maglat_lookup()
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
    calibration_cache = {}
    while t <= end_dt:
        step_idx += 1
        time_arg = format_time_arg(t)
        print_progress(step_idx, total_steps, time_arg)
        imgs_calibrated = {}
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
                calibrated, _background = calibrate_image_cached(
                    site,
                    im_raw,
                    skymaps_by_alt[alts[0]][site]["mask"],
                    args.color,
                    site_frame_info["frame_time"],
                    calibration_cache,
                    edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX,
                )
                if calibrated is not None:
                    imgs_calibrated[site] = calibrated
                frame_info[site] = site_frame_info
            except Exception as exc:
                print(f"{time_arg} {site}: frame load/calibration failed: {exc}")

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
                    sample = (
                        best_rocket_brightness(lat, lon, skymaps_by_alt[alt], imgs_calibrated)
                        if lat is not None and lon is not None
                        else None
                    )
                    outside_footprint = bool(sample and sample.get("outside_footprint", False))
                    if sample and not outside_footprint:
                        plot_value = sample["brightness"]
                    else:
                        plot_value = None
                else:
                    sample = None
                    plot_value = None
                alt_prefix = alt_column_prefix(csv_key, alt)
                row[f"{alt_prefix}_brightness"] = f"{sample['brightness']:.3f}" if sample and not outside_footprint else ""
                series_key = f"{csv_key}_{format_alt_token(alt)}"
                if series_key in plot_series:
                    plot_series[series_key].append(plot_value)
        rows.append(row)
        t += step_td

    write_brightness_hdf5(out_path, args, rows, traj_configs, csv_prefixes, alts)
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
