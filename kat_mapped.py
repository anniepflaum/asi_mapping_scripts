#!/usr/bin/env python3
"""Export KAT ASI mapped brightness images and trajectory data to one NetCDF file."""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from netCDF4 import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.constants import DEFAULT_GREEN_ALT_KM, FRAME_INTERVAL_SECONDS_GREEN
from core.masks import build_overlap_masks
from core.missions import mission_output_dir, trajectory_configs, trajectory_display_labels
from core.paths import WORKSPACE_DIR
from core.series_utils import count_steps, format_time_arg, load_tiff_frame_with_metadata, print_progress
from core.skymaps import load_skymaps
from core.time_utils import parse_date_and_time, sanitize_time_for_filename
from core.tiff_utils import build_tiff_metadata, get_site_tiff_candidates
from core.traj_utils import build_traj_lookup, lookup_traj_position


SITES = ("ARV", "VEE", "BVR")
DEFAULT_DATE = "20260210"
DEFAULT_START = "100000"
DEFAULT_END = "104500"
DEFAULT_STEP_S = 2.0
SITE_CODES = {site: idx + 1 for idx, site in enumerate(SITES)}
FILL_VALUE = np.float32(np.nan)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Build one native-grid GNEISS ASI map file from ../images/green/{site} TIFF "
            "chunks, including timestamps, mapped pixel coordinates, and trajectory data."
        )
    )
    ap.add_argument("--date", default=DEFAULT_DATE, help="Image date as YYYYMMDD")
    ap.add_argument("--start", default=DEFAULT_START, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", default=DEFAULT_END, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP_S, help="Output cadence in seconds")
    ap.add_argument("--color", choices=("green",), default="green", help="ASI color channel")
    ap.add_argument("--green-alt", type=float, default=DEFAULT_GREEN_ALT_KM, help="Mapped altitude in km")
    ap.add_argument("--input-dir", type=Path, default=None, help="Input image root; default: ../images/green with site subfolders")
    ap.add_argument("--output", type=Path, default=None, help="Output NetCDF path")
    ap.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help="Optional metadata map bounds. Default derives from valid mapped ASI pixels.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return ap.parse_args()


def build_times(date, start, end, step_s):
    if step_s <= 0:
        raise ValueError("--step must be > 0")
    start_dt = parse_date_and_time(date, start)
    end_dt = parse_date_and_time(date, end)
    if end_dt < start_dt:
        raise ValueError("--end must be at or after --start")

    times = []
    current = start_dt
    step = dt.timedelta(seconds=step_s)
    while current <= end_dt + dt.timedelta(microseconds=1):
        times.append(current)
        current += step
    return times


def default_output_path(args):
    start_token = sanitize_time_for_filename(args.start)
    end_token = sanitize_time_for_filename(args.end)
    step_token = f"{args.step:g}".replace(".", "p")
    name = f"GNEISS_KAT_merged_{args.color}_{args.date}_{start_token}_{end_token}_step{step_token}s.nc"
    return mission_output_dir("GNEISS", color=args.color, date=args.date) / name


def site_search_dirs(input_root, site):
    if input_root is None:
        return None
    site_dir = input_root / site
    if site_dir.exists():
        return [str(site_dir)]
    return [str(input_root)]


def load_site_tiffs(input_root, date, color):
    tiffs = {}
    for site in SITES:
        candidates = get_site_tiff_candidates(
            site,
            date,
            color,
            override_dirs=site_search_dirs(input_root, site),
            mission="GNEISS",
        )
        metadata = build_tiff_metadata(candidates, FRAME_INTERVAL_SECONDS_GREEN)
        if not metadata:
            search_root = input_root if input_root is not None else WORKSPACE_DIR / "images" / color / site
            raise FileNotFoundError(f"No TIFFs found for {site} in {search_root}")
        first_path = metadata[0]["path"]
        with tifffile.TiffFile(first_path) as tif:
            n_pages = len(tif.pages)
            shape = tif.pages[0].shape
            dtype = tif.pages[0].dtype
        tiffs[site] = {
            "paths": [meta["path"] for meta in metadata],
            "metadata": metadata,
            "n_pages": sum(int(meta["n_frames"]) for meta in metadata),
            "shape": shape,
            "dtype": str(dtype),
        }
    return tiffs


def valid_site_mask(skymap):
    valid = ~np.asarray(skymap["mask"], dtype=bool)
    valid &= np.isfinite(skymap["lat"]) & np.isfinite(skymap["lon"])
    for mask in skymap.get("extra_masks", {}).values():
        valid &= ~np.asarray(mask, dtype=bool)
    return valid


def derive_bounds(skymaps, valid_masks):
    lats = []
    lons = []
    for site in SITES:
        valid = valid_masks[site]
        lats.append(np.asarray(skymaps[site]["lat"])[valid])
        lons.append(np.asarray(skymaps[site]["lon"])[valid])
    lat_values = np.concatenate(lats)
    lon_values = np.concatenate(lons)
    return (
        float(np.nanmin(lon_values)),
        float(np.nanmax(lon_values)),
        float(np.nanmin(lat_values)),
        float(np.nanmax(lat_values)),
    )


def masked_site_frame(tiffs, site, valid_masks, sample_dt):
    frame, _frame_info = load_tiff_frame_with_metadata(
        site,
        tiffs[site]["metadata"],
        sample_dt,
        frame_interval=FRAME_INTERVAL_SECONDS_GREEN,
    )
    masked = np.full(frame.shape, FILL_VALUE, dtype=np.float32)
    valid = valid_masks[site] & np.isfinite(frame)
    masked[valid] = frame[valid]
    return masked


def write_string_coord(ds, name, dim, values):
    var = ds.createVariable(name, str, (dim,))
    var[:] = np.asarray(values, dtype=object)
    return var


def trajectory_data(times, color, green_alt, date):
    configs = trajectory_configs("GNEISS", date=date)
    labels = trajectory_display_labels("GNEISS", date=date)
    lookups = {cfg.key: build_traj_lookup(str(cfg.path), color=color, green_alt=green_alt) for cfg in configs}
    n_rockets = len(configs)
    n_times = len(times)
    n_points = max(len(lookups[cfg.key]["lats"]) for cfg in configs)

    path_lat = np.full((n_rockets, n_points), np.nan, dtype=np.float32)
    path_lon = np.full((n_rockets, n_points), np.nan, dtype=np.float32)
    path_time = np.full((n_rockets, n_points), np.nan, dtype=np.float32)
    sample_lat = np.full((n_times, n_rockets), np.nan, dtype=np.float32)
    sample_lon = np.full((n_times, n_rockets), np.nan, dtype=np.float32)

    for rocket_idx, cfg in enumerate(configs):
        lookup = lookups[cfg.key]
        n = len(lookup["lats"])
        path_lat[rocket_idx, :n] = np.asarray(lookup["lats"], dtype=np.float32)
        path_lon[rocket_idx, :n] = np.asarray(lookup["lons"], dtype=np.float32)
        path_time[rocket_idx, :n] = np.asarray(lookup["times"], dtype=np.float32)

    for time_idx, sample_dt in enumerate(times):
        time_arg = format_time_arg(sample_dt)
        for rocket_idx, cfg in enumerate(configs):
            mapped = lookup_traj_position(lookups[cfg.key], time_arg)
            if mapped[0] is not None:
                sample_lat[time_idx, rocket_idx] = mapped[0]
                sample_lon[time_idx, rocket_idx] = mapped[1]

    return {
        "rocket_tags": [cfg.tag for cfg in configs],
        "rocket_labels": [labels.get(cfg.key, cfg.label) for cfg in configs],
        "path_lat": path_lat,
        "path_lon": path_lon,
        "path_time": path_time,
        "sample_lat": sample_lat,
        "sample_lon": sample_lon,
        "source_paths": {cfg.tag: str(cfg.path) for cfg in configs},
    }


def write_netcdf(path, args, times, skymaps, valid_masks, tiffs, traj, bounds):
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {path} (use --overwrite to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    y_size, x_size = tiffs[SITES[0]]["shape"]

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("site", len(SITES))
        ds.createDimension("y", y_size)
        ds.createDimension("x", x_size)
        ds.createDimension("rocket", len(traj["rocket_tags"]))
        ds.createDimension("trajectory_point", traj["path_lat"].shape[1])

        time_seconds = np.asarray([(time - times[0]).total_seconds() for time in times], dtype=np.float64)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var[:] = time_seconds
        time_var.units = f"seconds since {times[0].isoformat()}"
        time_var.calendar = "proleptic_gregorian"
        time_var.long_name = "requested output timestamp"

        write_string_coord(ds, "time_iso", "time", [time.isoformat() for time in times])
        write_string_coord(ds, "site", "site", SITES)
        write_string_coord(ds, "rocket", "rocket", traj["rocket_tags"])
        write_string_coord(ds, "rocket_label", "rocket", traj["rocket_labels"])

        lat_var = ds.createVariable("lat", "f4", ("site", "y", "x"), fill_value=FILL_VALUE, zlib=True, complevel=4, shuffle=True)
        lon_var = ds.createVariable("lon", "f4", ("site", "y", "x"), fill_value=FILL_VALUE, zlib=True, complevel=4, shuffle=True)
        valid_var = ds.createVariable("valid_mask", "i1", ("site", "y", "x"), zlib=True, complevel=4, shuffle=True)
        site_lat_var = ds.createVariable("site_lat", "f4", ("site",))
        site_lon_var = ds.createVariable("site_lon", "f4", ("site",))

        for site_idx, site in enumerate(SITES):
            lat = np.asarray(skymaps[site]["lat"], dtype=np.float32)
            lon = np.asarray(skymaps[site]["lon"], dtype=np.float32)
            valid = valid_masks[site]
            lat_out = np.full(lat.shape, FILL_VALUE, dtype=np.float32)
            lon_out = np.full(lon.shape, FILL_VALUE, dtype=np.float32)
            lat_out[valid] = lat[valid]
            lon_out[valid] = lon[valid]
            lat_var[site_idx, :, :] = lat_out
            lon_var[site_idx, :, :] = lon_out
            valid_var[site_idx, :, :] = valid.astype(np.int8)
            site_lat_var[site_idx] = np.float32(skymaps[site]["site_lat"])
            site_lon_var[site_idx] = np.float32(skymaps[site]["site_lon"])

        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        lat_var.long_name = "mapped ASI pixel latitude"
        lon_var.long_name = "mapped ASI pixel longitude"
        valid_var.long_name = "pixels retained for the main merged map after base and overlap masks"
        site_lat_var.units = "degrees_north"
        site_lon_var.units = "degrees_east"

        compression = {"zlib": True, "complevel": 4, "shuffle": True}
        brightness = ds.createVariable("brightness", "f4", ("time", "site", "y", "x"), fill_value=FILL_VALUE, **compression)
        brightness.units = "raw_counts"
        brightness.long_name = "mapped ASI brightness on each native camera grid"
        brightness.merge_method = "same as map_asi_archive.py main map: native site grids with base mask and overlap masks applied; no regular-grid resampling"

        for name, data, dims, units, long_name in (
            ("trajectory_path_lat", traj["path_lat"], ("rocket", "trajectory_point"), "degrees_north", "full mapped trajectory latitude"),
            ("trajectory_path_lon", traj["path_lon"], ("rocket", "trajectory_point"), "degrees_east", "full mapped trajectory longitude"),
            ("trajectory_path_time", traj["path_time"], ("rocket", "trajectory_point"), "s", "trajectory flight time since launch"),
            ("trajectory_sample_lat", traj["sample_lat"], ("time", "rocket"), "degrees_north", "mapped trajectory latitude at each ASI timestamp"),
            ("trajectory_sample_lon", traj["sample_lon"], ("time", "rocket"), "degrees_east", "mapped trajectory longitude at each ASI timestamp"),
        ):
            var = ds.createVariable(name, "f4", dims, fill_value=FILL_VALUE, **compression)
            var[:] = data
            var.units = units
            var.long_name = long_name

        total_steps = count_steps(times[0], times[-1], args.step)
        for frame_idx, sample_dt in enumerate(times):
            print_progress(frame_idx, total_steps, format_time_arg(sample_dt))
            for site_idx, site in enumerate(SITES):
                brightness[frame_idx, site_idx, :, :] = masked_site_frame(tiffs, site, valid_masks, sample_dt)
        print_progress(len(times), total_steps, format_time_arg(times[-1]))

        ds.title = "GNEISS KAT mapped ASI brightness images and trajectories"
        ds.mission = "GNEISS"
        ds.color = args.color
        ds.map_alt_km = float(args.green_alt)
        ds.requested_start_time = parse_date_and_time(args.date, args.start).isoformat()
        ds.requested_end_time = parse_date_and_time(args.date, args.end).isoformat()
        ds.start_time = times[0].isoformat()
        ds.end_time = times[-1].isoformat()
        ds.cadence_seconds = float(args.step)
        ds.bounds = json.dumps({"lon_min": bounds[0], "lon_max": bounds[1], "lat_min": bounds[2], "lat_max": bounds[3]})
        ds.sites = ",".join(SITES)
        ds.site_codes = json.dumps(SITE_CODES)
        ds.native_image_shape = json.dumps({"y": y_size, "x": x_size})
        ds.spatial_layout = "native_site_pixel_grids"
        ds.source_tiffs = json.dumps({site: [str(path) for path in info["paths"]] for site, info in tiffs.items()}, sort_keys=True)
        ds.source_tiff_frame_counts = json.dumps({site: int(info["n_pages"]) for site, info in tiffs.items()}, sort_keys=True)
        ds.trajectory_sources = json.dumps(traj["source_paths"], sort_keys=True)
        ds.history = "created by kat_mapped.py"


def main():
    args = parse_args()
    input_root = args.input_dir or WORKSPACE_DIR / "images" / args.color
    output_path = args.output or default_output_path(args)
    times = build_times(args.date, args.start, args.end, args.step)
    tiffs = load_site_tiffs(input_root, args.date, args.color)

    skymaps = load_skymaps(set(SITES), color=args.color, mission="GNEISS", green_alt=args.green_alt)
    build_overlap_masks(skymaps, map_alt_km=args.green_alt)
    valid_masks = {site: valid_site_mask(skymaps[site]) for site in SITES}
    bounds = tuple(args.bounds) if args.bounds is not None else derive_bounds(skymaps, valid_masks)
    traj = trajectory_data(times, args.color, args.green_alt, args.date)
    y_size, x_size = tiffs[SITES[0]]["shape"]

    print(f"Writing KAT ASI mapped image NetCDF to {output_path}")
    print(f"Native grids: {len(SITES)} sites x {y_size} y x {x_size} x, {len(times)} timestamps")
    write_netcdf(output_path, args, times, skymaps, valid_masks, tiffs, traj, bounds)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
