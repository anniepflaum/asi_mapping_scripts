#!/usr/bin/env python3
"""
Build a brightness-vs-time dataset for both GNEISS trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket position (left/right trajectory) at that time.
3) Samples brightness at rocket position using the same logic as map_asi_archive.py
   (mean of 25 nearest valid pixels, with percentile metadata).
4) Writes one CSV row per timestamp with left/right values.
"""

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import tifffile

from core.brightness import best_rocket_brightness
from core.calc_ipp import calc_ipp
from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED
from core.masks import build_overlap_masks
from core.remote_data import retrieve_image
from core.skymaps import load_skymaps
from core.time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from core.fetch_url import closest_amisr_png_url
from core.paths import LEFT_TRAJECTORY_PATH, RECEIVERS_PATH, RIGHT_TRAJECTORY_PATH
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates
from core.traj_utils import build_traj_lookup, lookup_traj_geodetic_position, lookup_traj_position


def format_time_arg(t):
    hhmmss = t.strftime("%H%M%S")
    frac = f"{t.microsecond:06d}".rstrip("0")
    return f"{hhmmss}.{frac}" if frac else hhmmss


def make_output_path(out_arg, date, start, end, step):
    if out_arg:
        return Path(out_arg)
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    return Path(f"brightness_vs_time_{start_tok}_{end_tok}_step{step_tok}.csv")


def make_plot_output_path(csv_path, output_arg):
    if output_arg:
        return Path(output_arg)
    return Path(csv_path).with_suffix(".png")


def make_receiver_output_path(csv_path):
    return csv_path.with_name(f"{csv_path.stem}_receiver_ipps.csv")


def plot_brightness_timeseries(times, left, right, output_path, title, left_site_changes=None, right_site_changes=None):
    if not times:
        raise ValueError("No rows with iso_time were collected")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(times, left, color="tab:red", s=10, label="397 brightness")
    ax.scatter(times, right, color="tab:blue", s=10, label="398 brightness")
    for idx, change_time in enumerate(left_site_changes or []):
        ax.axvline(
            change_time,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label="397 frame site change" if idx == 0 else None,
        )
    for idx, change_time in enumerate(right_site_changes or []):
        ax.axvline(
            change_time,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label="398 frame site change" if idx == 0 else None,
        )
    ax.set_title(title)
    ax.set_xlabel("Time")
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


def parse_frame_datetime_from_url(url):
    match = re.search(r"(\d{8})_(\d{6})", url)
    if not match:
        raise ValueError(f"Could not parse frame timestamp from URL: {url}")
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def load_tiff_frame_with_metadata(site, tiff_metadata, target_dt, frame_interval):
    if not tiff_metadata:
        raise FileNotFoundError(f"No TIFF files found for {site}")

    candidates = []
    for meta in tiff_metadata:
        raw_idx = int(round((target_dt - meta["start_dt"]).total_seconds() / frame_interval))
        idx = min(max(raw_idx, 0), meta["n_frames"] - 1)
        frame_dt = meta["start_dt"] + dt.timedelta(seconds=idx * frame_interval)
        candidates.append(
            {
                "path": meta["path"],
                "idx": idx,
                "frame_dt": frame_dt,
                "delta_s": abs((frame_dt - target_dt).total_seconds()),
                "in_range": meta["start_dt"] <= target_dt <= meta["end_dt"],
            }
        )

    in_range_candidates = [c for c in candidates if c["in_range"]]
    best = min(in_range_candidates or candidates, key=lambda c: c["delta_s"])

    with tifffile.TiffFile(best["path"]) as tif:
        im = tif.pages[best["idx"]].asarray()
    if im.ndim == 3:
        im = im[:, :, 0]
    return im.astype("float32"), {"site": site, "frame_time": best["frame_dt"].isoformat()}


def get_site_change_times(rows, fieldname):
    change_times = []
    previous_value = None
    first_row = True
    for row in rows:
        current_value = row[fieldname]
        if first_row:
            previous_value = current_value
            first_row = False
            continue
        if current_value != previous_value:
            change_times.append(dt.datetime.fromisoformat(row["time"]))
        previous_value = current_value
    return change_times


def load_receivers(path=RECEIVERS_PATH):
    with open(path, "r", encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd)
        return [
            {
                "name": row["Name"].strip(),
                "acronym": row["acronym"].strip(),
                "lat": float(row["Lat"]),
                "lon": float(row["Lon"]),
                "alt_m": 0.0,
            }
            for row in reader
            if row.get("Lat") and row.get("Lon")
        ]


def sample_receiver_ipp_brightnesses(receivers, rocket_geo, skymaps, imgs_raw, ipp_height_km=110.0):
    brightnesses = []
    sites = []
    if rocket_geo[0] is None or rocket_geo[1] is None or rocket_geo[2] is None:
        return [None] * len(receivers), [""] * len(receivers)

    rocket_lat, rocket_lon, rocket_alt_km = rocket_geo
    rocket_position = [rocket_lat, rocket_lon, rocket_alt_km * 1000.0]
    for receiver in receivers:
        ipp_lat, ipp_lon = calc_ipp(
            [receiver["lat"], receiver["lon"], receiver["alt_m"]],
            rocket_position,
            rockcoords="geo",
            height=ipp_height_km,
        )
        sample = best_rocket_brightness(ipp_lat, ipp_lon, skymaps, imgs_raw)
        brightnesses.append(sample["raw_brightness"] if sample else None)
        sites.append(sample["site"] if sample else "")
    return brightnesses, sites

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260210", help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=0.05, help="Step size in seconds (default: 0.3)")
    ap.add_argument("--sites", nargs="*", default=["ARV", "BVR", "VEE", "PKR"], help="Sites to include")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output CSV path")
    ap.add_argument("--plot-output", default=None, help="Optional output PNG path for the brightness plot")
    ap.add_argument("--plot-title", default=None, help="Optional plot title")
    ap.add_argument("--no-plot", action="store_true", help="Write the CSV only and skip the PNG plot")
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
    receivers = load_receivers()
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

    left_traj = build_traj_lookup(str(LEFT_TRAJECTORY_PATH), color=args.color)
    right_traj = build_traj_lookup(str(RIGHT_TRAJECTORY_PATH), color=args.color)

    out_path = make_output_path(args.output, args.date, args.start, args.end, args.step)
    if not out_path.is_absolute():
        out_path = Path("..") / "mapped" / args.color / out_path

    fieldnames = [
        "time",
        "left_frame_site",
        "left_frame_time",
        "left_percentile",
        "left_brightness",
        "right_frame_site",
        "right_frame_time",
        "right_percentile",
        "right_brightness",
    ]
    receiver_fieldnames = ["time"]
    for rocket_label in ["397", "398"]:
        for receiver in receivers:
            acronym = receiver["acronym"]
            receiver_fieldnames.append(f"{rocket_label}_{acronym}_ipp_brightness")
            receiver_fieldnames.append(f"{rocket_label}_{acronym}_ipp_site")

    rows = []
    receiver_rows = []
    plot_times = []
    plot_left = []
    plot_right = []
    step_td = dt.timedelta(seconds=args.step)
    t = start_dt
    while t <= end_dt:
        time_arg = format_time_arg(t)
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

        lat_l, lon_l = lookup_traj_position(left_traj, time_arg)
        lat_r, lon_r = lookup_traj_position(right_traj, time_arg)
        left_geo = lookup_traj_geodetic_position(left_traj, time_arg)
        right_geo = lookup_traj_geodetic_position(right_traj, time_arg)

        left = best_rocket_brightness(lat_l, lon_l, skymaps, imgs_raw) if lat_l is not None and lon_l is not None else None
        right = best_rocket_brightness(lat_r, lon_r, skymaps, imgs_raw) if lat_r is not None and lon_r is not None else None
        left_receiver_brightnesses, left_receiver_sites = sample_receiver_ipp_brightnesses(receivers, left_geo, skymaps, imgs_raw)
        right_receiver_brightnesses, right_receiver_sites = sample_receiver_ipp_brightnesses(receivers, right_geo, skymaps, imgs_raw)

        plot_times.append(t)
        plot_left.append(left["raw_brightness"] if left else None)
        plot_right.append(right["raw_brightness"] if right else None)
        rows.append(
            {
                "time": t.isoformat(),
                "left_frame_site": frame_info[left["site"]]["site"] if left and left["site"] in frame_info else "",
                "left_frame_time": frame_info[left["site"]]["frame_time"] if left and left["site"] in frame_info else "",
                "left_percentile": f"{left['percentile']:.3f}" if left else "",
                "left_brightness": f"{left['raw_brightness']:.3f}" if left else "",
                "right_frame_site": frame_info[right["site"]]["site"] if right and right["site"] in frame_info else "",
                "right_frame_time": frame_info[right["site"]]["frame_time"] if right and right["site"] in frame_info else "",
                "right_percentile": f"{right['percentile']:.3f}" if right else "",
                "right_brightness": f"{right['raw_brightness']:.3f}" if right else "",
            }
        )
        receiver_row = {"time": t.isoformat()}
        for receiver, brightness, site_name in zip(receivers, left_receiver_brightnesses, left_receiver_sites):
            acronym = receiver["acronym"]
            receiver_row[f"397_{acronym}_ipp_brightness"] = f"{brightness:.3f}" if brightness is not None else ""
            receiver_row[f"397_{acronym}_ipp_site"] = site_name
        for receiver, brightness, site_name in zip(receivers, right_receiver_brightnesses, right_receiver_sites):
            acronym = receiver["acronym"]
            receiver_row[f"398_{acronym}_ipp_brightness"] = f"{brightness:.3f}" if brightness is not None else ""
            receiver_row[f"398_{acronym}_ipp_site"] = site_name
        receiver_rows.append(receiver_row)
        t += step_td

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    receiver_out_path = make_receiver_output_path(out_path)
    with receiver_out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=receiver_fieldnames)
        writer.writeheader()
        writer.writerows(receiver_rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Wrote {len(receiver_rows)} rows to {receiver_out_path}")
    if not args.no_plot:
        left_site_changes = get_site_change_times(rows, "left_frame_site")
        right_site_changes = get_site_change_times(rows, "right_frame_site")
        plot_output = make_plot_output_path(out_path, args.plot_output)
        plot_title = args.plot_title or f"Brightness vs Time ({args.color})"
        plot_brightness_timeseries(
            plot_times,
            plot_left,
            plot_right,
            plot_output,
            plot_title,
            left_site_changes=left_site_changes,
            right_site_changes=right_site_changes,
        )


if __name__ == "__main__":
    main()
