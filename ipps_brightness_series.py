#!/usr/bin/env python3
"""
Build an IPP-brightness-vs-time dataset for both GNEISS trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket geodetic position (left/right trajectory) at that time.
3) Computes the ionospheric pierce point for each receiver and rocket.
4) Samples ASI brightness at each IPP using the same nearest-valid-pixels logic
   used by map_asi_archive.py.
5) Writes one CSV row per timestamp with per-receiver IPP values.
"""

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from core.brightness import best_rocket_brightness
from core.calc_ipp import calc_ipp
from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.fetch_url import closest_amisr_png_url
from core.masks import build_overlap_masks
from core.remote_data import retrieve_image
from core.skymaps import load_skymaps
from core.time_utils import parse_date_and_time, parse_hhmmss_fractional, sanitize_time_for_filename
from core.paths import GNEISS_LEFT_TRAJECTORY_PATH, GNEISS_RIGHT_TRAJECTORY_PATH
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates
from core.traj_utils import build_traj_lookup, lookup_traj_geodetic_position, mapped_apex_height
from traj_brightness_series import count_steps, format_time_arg, load_receivers, load_tiff_frame_with_metadata, print_progress


def receiver_suffix(receivers, all_receivers):
    selected = [receiver["acronym"] for receiver in receivers]
    available = sorted(receiver["acronym"] for receiver in all_receivers)
    if sorted(selected) == available:
        return ""
    return "_receivers_" + "_".join(selected)


def make_output_path(out_arg, start, end, step, receivers, all_receivers):
    if out_arg:
        return Path(out_arg)
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    return Path(f"ipps_brightness_series_{start_tok}_{end_tok}_step{step_tok}{receiver_suffix(receivers, all_receivers)}.csv")


def make_plot_output_path(csv_path, output_arg):
    if output_arg:
        return Path(output_arg)
    return Path(csv_path).with_suffix(".png")


def parse_series_filename(csv_path, prefix):
    pattern = rf"^{re.escape(prefix)}_(\d{{6}}(?:p\d+)?)_(\d{{6}}(?:p\d+)?)_step(\d+(?:p\d+)?)(?:_receivers_.+)?\.csv$"
    match = re.match(pattern, csv_path.name)
    if not match:
        return None
    start_tok, end_tok, step_tok = match.groups()
    return {
        "start_tok": start_tok,
        "end_tok": end_tok,
        "start_time": start_tok.replace("p", "."),
        "end_time": end_tok.replace("p", "."),
        "step": float(step_tok.replace("p", ".")),
    }


def build_requested_iso_times(date, start, end, step):
    start_dt = parse_date_and_time(date, start)
    end_dt = parse_date_and_time(date, end)
    step_td = dt.timedelta(seconds=step)
    requested = []
    t = start_dt
    while t <= end_dt:
        requested.append(t.isoformat())
        t += step_td
    return requested


def csv_contains_requested_times(csv_path, requested_times):
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            available = {row["time"] for row in reader if row.get("time")}
    except Exception:
        return False
    return all(time_value in available for time_value in requested_times)


def find_reusable_csv(preferred_path, prefix, date, start, end, step, required_fields):
    requested_times = build_requested_iso_times(date, start, end, step)
    if preferred_path.exists() and csv_contains_requested_times(preferred_path, requested_times):
        try:
            load_rows_from_csv(preferred_path, required_fields)
            return preferred_path
        except Exception:
            pass

    request_start_dt = parse_date_and_time(date, start)
    request_end_dt = parse_date_and_time(date, end)
    candidates = []
    for candidate in preferred_path.parent.glob(f"{prefix}_*.csv"):
        parsed = parse_series_filename(candidate, prefix)
        if parsed is None:
            continue
        try:
            candidate_start_dt = parse_date_and_time(date, parsed["start_time"])
            candidate_end_dt = parse_date_and_time(date, parsed["end_time"])
        except ValueError:
            continue
        if candidate_start_dt > request_start_dt or candidate_end_dt < request_end_dt:
            continue
        if csv_contains_requested_times(candidate, requested_times):
            try:
                load_rows_from_csv(candidate, required_fields)
            except Exception:
                continue
            span_seconds = (candidate_end_dt - candidate_start_dt).total_seconds()
            candidates.append((span_seconds, parsed["step"], candidate))
    if not candidates:
        return preferred_path
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    return candidates[0][2]


def load_rows_from_csv(csv_path, required_fields):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"CSV file {csv_path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def filter_receivers(receivers, requested_acronyms):
    if not requested_acronyms:
        return receivers
    requested = [acronym.upper() for acronym in requested_acronyms]
    receiver_map = {receiver["acronym"].upper(): receiver for receiver in receivers}
    missing = [acronym for acronym in requested if acronym not in receiver_map]
    if missing:
        raise ValueError(f"Unknown receiver acronym(s): {', '.join(missing)}")
    return [receiver_map[acronym] for acronym in requested]


def compute_receiver_ipp_samples(receivers, rocket_geo, skymaps, imgs_raw, ipp_height_km):
    samples = []
    if (
        not receivers
        or rocket_geo[0] is None
        or rocket_geo[1] is None
        or rocket_geo[2] is None
        or rocket_geo[2] <= ipp_height_km
    ):
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
        brightness_sample = best_rocket_brightness(float(ipp_lat), float(ipp_lon), skymaps, imgs_raw)
        samples.append(
            {
                "acronym": receiver["acronym"],
                "ipp_lat": float(ipp_lat),
                "ipp_lon": float(ipp_lon),
                "site": brightness_sample["site"] if brightness_sample else "",
                "percentile": brightness_sample["percentile"] if brightness_sample else None,
                "brightness": brightness_sample["raw_brightness"] if brightness_sample else None,
            }
        )
    return samples


def build_fieldnames(receivers):
    fieldnames = ["time"]
    for rocket_label in ["397", "398"]:
        for receiver in receivers:
            acronym = receiver["acronym"]
            fieldnames.extend(
                [
                    f"{rocket_label}_{acronym}_ipp_lat",
                    f"{rocket_label}_{acronym}_ipp_lon",
                    f"{rocket_label}_{acronym}_ipp_site",
                    f"{rocket_label}_{acronym}_ipp_percentile",
                    f"{rocket_label}_{acronym}_ipp_brightness",
                ]
            )
    return fieldnames


def plot_ipps_timeseries(times, rows, receivers, output_path, title):
    if not times:
        raise ValueError("No rows with iso_time were collected")

    fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
    rocket_configs = [("397", axes[0]), ("398", axes[1])]
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
            ax.set_ylabel(f"{rocket_label} brightness")
            ax.grid(True, alpha=0.3)

    if all_brightnesses:
        ymin = min(all_brightnesses)
        ymax = max(all_brightnesses)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncols=min(4, len(receivers)), fontsize=8, loc="upper left")
    axes[0].set_title(title)
    axes[1].set_xlabel("Time")
    axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260210", help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=0.05, help="Step size in seconds")
    ap.add_argument("--sites", nargs="*", default=["ARV", "BVR", "VEE", "PKR"], help="Sites to include")
    ap.add_argument("--receivers", nargs="*", default=None, help="Receiver acronyms to include in the CSV and plot")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output CSV path")
    ap.add_argument("--plot-output", default=None, help="Optional output PNG path for the brightness plot")
    ap.add_argument("--plot-title", default=None, help="Optional plot title")
    ap.add_argument("--no-plot", action="store_true", help="Write the CSV only and skip the PNG plot")
    ap.add_argument("--no-csv", dest="no_csv", action="store_true", default=True, help="Skip CSV generation and plot from an existing CSV instead (default)")
    ap.add_argument("--csv", dest="no_csv", action="store_false", help="Generate a CSV instead of reusing an existing one")
    args = ap.parse_args()

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
    all_receivers = load_receivers()
    try:
        receivers = filter_receivers(all_receivers, args.receivers)
    except ValueError as exc:
        ap.error(str(exc))
    skymaps = load_skymaps(selected_sites, color=args.color)
    build_overlap_masks(skymaps)

    tiff_candidates = {}
    tiff_metadata = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site in selected_sites:
            tiff_candidates[site] = get_site_tiff_candidates(site, args.date, args.color)

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED
    for site in ["ARV", "VEE", "BVR"]:
        if site in tiff_candidates:
            tiff_metadata[site] = build_tiff_metadata(tiff_candidates[site], frame_interval)

    ipp_height_km = mapped_apex_height(args.color)
    left_traj = build_traj_lookup(str(GNEISS_LEFT_TRAJECTORY_PATH), color=args.color)
    right_traj = build_traj_lookup(str(GNEISS_RIGHT_TRAJECTORY_PATH), color=args.color)

    out_path = make_output_path(args.output, args.start, args.end, args.step, receivers, all_receivers)
    if not out_path.is_absolute():
        out_path = Path("..") / "mapped" / args.color / out_path

    fieldnames = build_fieldnames(receivers)

    if args.no_csv:
        if args.no_plot:
            ap.error("--no-csv cannot be combined with --no-plot")
        csv_path = find_reusable_csv(out_path, "ipps_brightness_series", args.date, args.start, args.end, args.step, ["time", *fieldnames[1:]])
        rows = load_rows_from_csv(csv_path, ["time", *fieldnames[1:]])
        requested_times = set(build_requested_iso_times(args.date, args.start, args.end, args.step))
        rows = [row for row in rows if row.get("time") in requested_times]
        plot_times = [dt.datetime.fromisoformat(row["time"]) for row in rows if row.get("time")]
        plot_output = make_plot_output_path(out_path, args.plot_output)
        plot_title = args.plot_title or f"IPP Brightness vs Time ({args.color})"
        plot_ipps_timeseries(plot_times, rows, receivers, plot_output, plot_title)
        print(f"Plotted from existing CSV {csv_path}")
        return

    rows = []
    plot_times = []
    step_td = dt.timedelta(seconds=args.step)
    total_steps = count_steps(start_dt, end_dt, args.step)
    step_idx = 0
    t = start_dt
    while t <= end_dt:
        step_idx += 1
        time_arg = format_time_arg(t)
        print_progress(step_idx, total_steps, time_arg)
        left_geo = lookup_traj_geodetic_position(left_traj, time_arg)
        right_geo = lookup_traj_geodetic_position(right_traj, time_arg)

        imgs_raw = {}

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
                imgs_raw[site] = im_raw
            except Exception as exc:
                print(f"{time_arg} {site}: frame load failed: {exc}")

        if "PKR" in selected_sites:
            try:
                pkr_lookup_time = t.strftime("%H%M%S")
                url_pkr = closest_amisr_png_url("PKR", args.date, pkr_lookup_time, color=args.color)
                imgs_raw["PKR"] = retrieve_image(url_pkr)
            except Exception as exc:
                print(f"{time_arg} PKR: frame load failed: {exc}")

        left_samples = compute_receiver_ipp_samples(receivers, left_geo, skymaps, imgs_raw, ipp_height_km)
        right_samples = compute_receiver_ipp_samples(receivers, right_geo, skymaps, imgs_raw, ipp_height_km)

        row = {"time": t.isoformat()}
        for rocket_label, samples in (("397", left_samples), ("398", right_samples)):
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if not args.no_plot:
        plot_output = make_plot_output_path(out_path, args.plot_output)
        plot_title = args.plot_title or f"IPP Brightness vs Time ({args.color})"
        plot_ipps_timeseries(plot_times, rows, receivers, plot_output, plot_title)


if __name__ == "__main__":
    main()
