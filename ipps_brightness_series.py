#!/usr/bin/env python3
"""
Build an IPP-brightness-vs-time dataset for mission trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket geodetic position at that time.
3) Computes the ionospheric pierce point for each receiver and rocket.
4) Samples ASI brightness at each IPP using the same nearest-valid-pixels logic
   used by map_asi_archive.py.
5) Writes per-receiver IPP time series to a structured HDF5 file.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
from core.calc_ipp import calc_ipp
from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.series_utils import (
    count_steps,
    format_time_arg,
    load_tiff_frame_with_metadata,
    make_plot_output_path,
    print_progress,
)
from core.skymaps import load_skymaps
from core.time_utils import parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from core.missions import default_date, default_sites, default_time_range, mission_output_dir, trajectory_config_tuples, validate_color_and_sites
from core.receivers import filter_receivers_for_mission, load_receivers
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates
from core.traj_utils import build_traj_lookup, lookup_traj_geodetic_position, mapped_apex_height


def receiver_suffix(receivers, all_receivers):
    selected = [receiver["acronym"] for receiver in receivers]
    available = sorted(receiver["acronym"] for receiver in all_receivers)
    if sorted(selected) == available:
        return ""
    return "_receivers_" + "_".join(selected)


def series_prefix(mission):
    return "ipps_brightness_series" if str(mission).upper() == "GNEISS" else f"{str(mission).upper()}_ipps_brightness_series"


def format_height_token(height_km):
    return f"{float(height_km):g}".replace(".", "p")


def make_output_path(
    out_arg,
    mission,
    date,
    start,
    end,
    step,
    receivers,
    all_receivers,
    color,
    auroral_layer_height_km,
):
    if out_arg:
        path = Path(out_arg)
        return path if path.suffix.lower() in {".h5", ".hdf5"} else path.with_suffix(".h5")
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    default_height_km = float(mapped_apex_height(color))
    height_suffix = ""
    if not np.isclose(float(auroral_layer_height_km), default_height_km):
        height_suffix = f"_alt{format_height_token(auroral_layer_height_km)}km"
    return Path(
        f"{series_prefix(mission)}_{date}_{start_tok}_{end_tok}_step{step_tok}"
        f"{height_suffix}{receiver_suffix(receivers, all_receivers)}.h5"
    )


def filter_receivers(receivers, requested_acronyms):
    if not requested_acronyms:
        return receivers
    requested = [acronym.upper() for acronym in requested_acronyms]
    receiver_map = {receiver["acronym"].upper(): receiver for receiver in receivers}
    missing = [acronym for acronym in requested if acronym not in receiver_map]
    if missing:
        raise ValueError(f"Unknown receiver acronym(s): {', '.join(missing)}")
    return [receiver_map[acronym] for acronym in requested]


def rocket_is_above_ipp(rocket_geo, ipp_height_km):
    """Return True only when a valid rocket position is above the IPP layer."""
    return (
        rocket_geo is not None
        and len(rocket_geo) >= 3
        and rocket_geo[0] is not None
        and rocket_geo[1] is not None
        and rocket_geo[2] is not None
        and np.isfinite(rocket_geo[2])
        and rocket_geo[2] > ipp_height_km
    )


def compute_receiver_ipp_samples(receivers, rocket_geo, skymaps, imgs_calibrated, ipp_height_km):
    samples = []
    if not receivers or not rocket_is_above_ipp(rocket_geo, ipp_height_km):
        for receiver in receivers:
            samples.append(
                {
                    "acronym": receiver["acronym"],
                    "ipp_lat": None,
                    "ipp_lon": None,
                    "site": "",
                    "percentile": None,
                    "brightness": None,
                }
            )
        return samples

    rocket_position = [rocket_geo[0], rocket_geo[1], rocket_geo[2] * 1000.0]
    for receiver in receivers:
        ipp_lat, ipp_lon = calc_ipp(
            [receiver["lat"], receiver["lon"], receiver.get("alt_m", 0.0)],
            rocket_position,
            rockcoords="geo",
            height=ipp_height_km,
        )
        brightness_sample = best_rocket_brightness(
            float(ipp_lat),
            float(ipp_lon),
            skymaps,
            imgs_calibrated,
        )
        samples.append(
            {
                "acronym": receiver["acronym"],
                "ipp_lat": float(ipp_lat),
                "ipp_lon": float(ipp_lon),
                "site": brightness_sample["site"] if brightness_sample else "",
                "percentile": brightness_sample["percentile"] if brightness_sample else None,
                "brightness": brightness_sample["brightness"] if brightness_sample else None,
            }
        )
    return samples


def _row_float(rows, field):
    return np.asarray(
        [float(row[field]) if row.get(field, "") != "" else np.nan for row in rows],
        dtype=np.float64,
    )


def write_hdf5(
    path,
    rows,
    receivers,
    rocket_labels,
    mission,
    color,
    ipp_height_km,
    calibration_json,
    selected_sites,
    step_s,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    layer_height_km = float(ipp_height_km)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "ipp_brightness_series"
        h5.attrs["schema_version"] = "1.0"
        h5.attrs["mission"] = mission
        h5.attrs["color"] = color
        h5.attrs["brightness_units"] = "Rayleighs"
        # These names describe the same physical layer and must stay identical.
        h5.attrs["auroral_layer_height_km"] = layer_height_km
        h5.attrs["ipp_height_km"] = layer_height_km
        h5.attrs["step_s"] = float(step_s)
        h5.attrs["selected_sites_json"] = json.dumps(sorted(selected_sites))
        h5.attrs["receivers_json"] = json.dumps(
            [receiver["acronym"] for receiver in receivers]
        )
        h5.attrs["rockets_json"] = json.dumps(rocket_labels)
        h5.attrs["calibration_json"] = calibration_json
        h5.create_dataset(
            "time_iso",
            data=np.asarray([row["time"] for row in rows], dtype=object),
            dtype=string_dtype,
        )

        for rocket_label in rocket_labels:
            rocket_group = h5.require_group(f"rockets/{rocket_label}/receivers")
            for receiver in receivers:
                acronym = receiver["acronym"]
                prefix = f"{rocket_label}_{acronym}_ipp"
                group = rocket_group.create_group(acronym)
                for dataset_name, field_suffix, units in (
                    ("latitude_deg", "lat", "degrees_north"),
                    ("longitude_deg", "lon", "degrees_east"),
                    ("brightness", "brightness", "Rayleighs"),
                    ("percentile", "percentile", "percent"),
                ):
                    dataset = group.create_dataset(
                        dataset_name,
                        data=_row_float(rows, f"{prefix}_{field_suffix}"),
                        **compression,
                    )
                    dataset.attrs["units"] = units
                group.create_dataset(
                    "site",
                    data=np.asarray(
                        [row.get(f"{prefix}_site", "") for row in rows],
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )


def load_hdf5_plot_rows(path, receivers, rocket_labels):
    rows = []
    with h5py.File(path, "r") as h5:
        ipp_height_km = float(h5.attrs["ipp_height_km"])
        auroral_layer_height_km = float(h5.attrs["auroral_layer_height_km"])
        if auroral_layer_height_km != ipp_height_km:
            raise ValueError(
                f"{path} has inconsistent layer heights: "
                f"auroral_layer_height_km={auroral_layer_height_km:g}, "
                f"ipp_height_km={ipp_height_km:g}"
            )
        times = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5["time_iso"][:]
        ]
        brightness = {}
        for rocket_label in rocket_labels:
            for receiver in receivers:
                acronym = receiver["acronym"]
                brightness[(rocket_label, acronym)] = np.asarray(
                    h5[f"rockets/{rocket_label}/receivers/{acronym}/brightness"],
                    dtype=float,
                )
    for index, time_iso in enumerate(times):
        row = {"time": time_iso}
        for (rocket_label, acronym), values in brightness.items():
            value = values[index]
            row[f"{rocket_label}_{acronym}_ipp_brightness"] = (
                f"{value:.3f}" if np.isfinite(value) else ""
            )
        rows.append(row)
    return rows, ipp_height_km


def plot_ipps_timeseries(times, rows, receivers, rocket_labels, output_path, title):
    if not times:
        raise ValueError("No rows with iso_time were collected")

    fig, axes = plt.subplots(len(rocket_labels), 1, figsize=(12, 8 * len(rocket_labels)), sharex=True)
    if len(rocket_labels) == 1:
        axes = [axes]
    rocket_configs = list(zip(rocket_labels, axes))
    cmap = plt.get_cmap("tab10")
    all_brightnesses = []

    for idx, receiver in enumerate(receivers):
        acronym = receiver["acronym"]
        color = cmap(idx % 10)
        for rocket_label, ax in rocket_configs:
            brightnesses = []
            for row in rows:
                value = row[f"{rocket_label}_{acronym}_ipp_brightness"]
                brightnesses.append(float(value) if value else None)
            all_brightnesses.extend(value for value in brightnesses if value is not None and np.isfinite(value) and value > 0)
            ax.plot(times, brightnesses, linewidth=1.0, color=color, label=acronym)
            ax.set_yscale("log")
            ax.set_ylabel(f"{rocket_label} brightness (Rayleighs)")
            ax.grid(True, alpha=0.3)

    if all_brightnesses:
        ymin = min(all_brightnesses)
        ymax = max(all_brightnesses)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncols=min(4, len(receivers)), fontsize=8, loc="upper left")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="Start time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--end", default=None, help="End time HHMMSS(.fraction); defaults to mission rocket window")
    ap.add_argument("--step", type=float, default=0.05, help="Step size in seconds")
    ap.add_argument("--sites", nargs="*", default=None, help="Sites to include")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default="GNEISS", help="Mission dataset to use")
    ap.add_argument("--receivers", nargs="*", default=None, help="Receiver acronyms to include in the HDF5 file and plot")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument(
        "--auroral-layer-height-km",
        type=float,
        default=None,
        help="Auroral layer and IPP height in km; defaults to 110 for green and 180 for red",
    )
    ap.add_argument("--output", default=None, help="Output HDF5 path")
    ap.add_argument("--no-plot", action="store_true", help="Write the HDF5 file only and skip the PNG plot")
    ap.add_argument(
        "--no-h5",
        dest="no_h5",
        action="store_true",
        help="Skip generation and plot from the existing HDF5 file instead",
    )
    args = ap.parse_args()
    args.mission = args.mission.upper()
    args.date = default_date(args.mission)
    default_start, default_end = default_time_range(args.mission, args.date)
    if args.start is None:
        args.start = default_start
    if args.end is None:
        args.end = default_end
    if args.sites is None:
        args.sites = default_sites(args.mission, include_pkr=True)
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
    if args.auroral_layer_height_km is None:
        args.auroral_layer_height_km = mapped_apex_height(args.color)
    if args.auroral_layer_height_km <= 0:
        ap.error("--auroral-layer-height-km must be > 0")

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    selected_sites = set(args.sites)
    all_receivers = load_receivers()
    try:
        mission_receivers = filter_receivers_for_mission(all_receivers, args.mission)
        receivers = filter_receivers(mission_receivers, args.receivers)
    except ValueError as exc:
        ap.error(str(exc))
    ipp_height_km = float(args.auroral_layer_height_km)
    skymaps = load_skymaps(
        selected_sites,
        color=args.color,
        mission=args.mission,
        map_alt_km=ipp_height_km,
    )
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
        key: build_traj_lookup(str(path), color=args.color)
        for key, _tag, _label, path in traj_configs
    }
    rocket_labels = [tag for _key, tag, _label, _path in traj_configs]

    out_path = make_output_path(
        args.output,
        args.mission,
        args.date,
        args.start,
        args.end,
        args.step,
        receivers,
        mission_receivers,
        args.color,
        ipp_height_km,
    )
    if not out_path.is_absolute():
        out_path = mission_output_dir(args.mission, color=args.color, date=args.date) / out_path

    if args.no_h5:
        if args.no_plot:
            ap.error("--no-h5 cannot be combined with --no-plot")
        if not out_path.exists():
            raise FileNotFoundError(f"Existing HDF5 file not found: {out_path}")
        rows, file_ipp_height_km = load_hdf5_plot_rows(
            out_path, receivers, rocket_labels
        )
        plot_times = [dt.datetime.fromisoformat(row["time"]) for row in rows if row.get("time")]
        plot_output = make_plot_output_path(out_path, None)
        plot_title = (
            f"Calibrated IPP Brightness vs Time "
            f"({args.color}, {file_ipp_height_km:g} km IPP layer)"
        )
        plot_ipps_timeseries(plot_times, rows, receivers, rocket_labels, plot_output, plot_title)
        print(f"Plotted from existing HDF5 {out_path}")
        return

    rows = []
    plot_times = []
    step_td = dt.timedelta(seconds=args.step)
    total_steps = count_steps(start_dt, end_dt, args.step)
    step_idx = 0
    t = start_dt
    calibration_cache = {}
    calibration_json = json.dumps(calibration_metadata(args.sites, args.color), sort_keys=True)
    while t <= end_dt:
        step_idx += 1
        time_arg = format_time_arg(t)
        print_progress(step_idx, total_steps, time_arg)
        rocket_geos = {
            tag: lookup_traj_geodetic_position(traj_lookups[key], time_arg)
            for key, tag, _label, _path in traj_configs
        }

        imgs_calibrated = {}

        for site in ["ARV", "VEE", "BVR"]:
            if site not in selected_sites:
                continue
            try:
                im_raw, _site_frame_info = load_tiff_frame_with_metadata(
                    site,
                    tiff_metadata.get(site, []),
                    t,
                    frame_interval=frame_interval,
                )
                calibrated, _background = calibrate_image_cached(
                    site,
                    im_raw,
                    skymaps[site]["mask"],
                    args.color,
                    _site_frame_info["frame_time"],
                    calibration_cache,
                    edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX,
                )
                if calibrated is not None:
                    imgs_calibrated[site] = calibrated
            except Exception as exc:
                print(f"{time_arg} {site}: frame load/calibration failed: {exc}")

        rocket_samples = {
            rocket_label: compute_receiver_ipp_samples(
                receivers,
                geo,
                skymaps,
                imgs_calibrated,
                ipp_height_km,
            )
            for rocket_label, geo in rocket_geos.items()
        }

        row = {
            "time": t.isoformat(),
            "brightness_units": "Rayleighs",
            "calibration_json": calibration_json,
        }
        for rocket_label, samples in rocket_samples.items():
            for sample in samples:
                acronym = sample["acronym"]
                row[f"{rocket_label}_{acronym}_ipp_lat"] = f"{sample['ipp_lat']:.6f}" if sample["ipp_lat"] is not None else ""
                row[f"{rocket_label}_{acronym}_ipp_lon"] = f"{sample['ipp_lon']:.6f}" if sample["ipp_lon"] is not None else ""
                row[f"{rocket_label}_{acronym}_ipp_site"] = sample["site"]
                row[f"{rocket_label}_{acronym}_ipp_percentile"] = f"{sample['percentile']:.3f}" if sample["percentile"] is not None else ""
                row[f"{rocket_label}_{acronym}_ipp_brightness"] = f"{sample['brightness']:.3f}" if sample["brightness"] is not None else ""
        rows.append(row)
        plot_times.append(t)
        t += step_td

    write_hdf5(
        out_path,
        rows,
        receivers,
        rocket_labels,
        args.mission,
        args.color,
        ipp_height_km,
        calibration_json,
        args.sites,
        args.step,
    )
    print(f"Wrote {len(rows)} rows to {out_path}")
    if not args.no_plot:
        plot_output = make_plot_output_path(out_path, None)
        plot_title = (
            f"Calibrated IPP Brightness vs Time "
            f"({args.color}, {ipp_height_km:g} km IPP layer)"
        )
        plot_ipps_timeseries(plot_times, rows, receivers, rocket_labels, plot_output, plot_title)


if __name__ == "__main__":
    main()
