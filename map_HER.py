#!/usr/bin/env python3
"""Map one HERA/HER ASI frame using supplied azimuth/elevation FITS maps."""

import argparse
import datetime as dt
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tifffile
from apexpy import Apex
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.constants import DEFAULT_GREEN_ALT_KM, FRAME_INTERVAL_SECONDS_GREEN
from core.generate_skymap import azel2geo
from core.missions import mission_output_dir
from core.plotting import plot_map
from core.skymaps import load_skymaps
from core.tiff_utils import get_site_tiff_candidates, load_best_frame_from_tiffs
from core.time_utils import parse_date_and_time, sanitize_time_for_filename


DEFAULT_TIFF = Path("/Users/anniepflaum/Downloads/HER260210_10060306_16bit_X09.tif")
DEFAULT_AZ = Path("/Users/anniepflaum/Downloads/VEE_HERA_20260210_Az.FIT")
DEFAULT_EL = Path("/Users/anniepflaum/Downloads/VEE_HERA_20260210_El.FIT")
DEFAULT_DATE = "20260210"
DEFAULT_TIME = "102800"
HER_TIMESTAMP_OFFSET_SECONDS = 52.0
DEFAULT_SITE_LAT = 67.050003
DEFAULT_SITE_LON = -146.399994
DEFAULT_BOUNDS = (-149.5, -144.5, 66.0, 68.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Map one HER/HERA TIFF frame at a requested time.")
    parser.add_argument("--time", default=DEFAULT_TIME, help="Frame time as HHMMSS or HHMMSS.s")
    return parser.parse_args()


def acquisition_start_from_name(path, date):
    match = re.search(r"HER(\d{6})_(\d{6})(\d{2})", Path(path).name)
    if not match:
        raise ValueError(f"Could not parse HER acquisition start time from {path.name}")
    yymmdd, hhmmss, centiseconds = match.groups()
    date_from_name = f"20{yymmdd}"
    if date_from_name != date:
        print(f"Warning: TIFF date {date_from_name} differs from requested --date {date}")
    return dt.datetime.strptime(f"{date_from_name}{hhmmss}", "%Y%m%d%H%M%S") + dt.timedelta(
        seconds=int(centiseconds) / 100.0
    )


def chunk_number_from_name(path):
    match = re.search(r"_X(\d+)\.tif+$", Path(path).name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 1


def tiff_cadence_seconds(tiff):
    page = tiff.pages[0]
    for tag_name in ("AndorKineticCycleTime", "AndorAcquisitionCycleTime"):
        tag = page.tags.get(tag_name)
        if tag is not None:
            return float(tag.value)
    raise ValueError("Could not find Andor cadence tag in TIFF")


def frame_index_for_time(tiff_path, date, target_dt):
    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        cadence_s = tiff_cadence_seconds(tif)
    acquisition_start = acquisition_start_from_name(tiff_path, date)
    chunk_number = chunk_number_from_name(tiff_path)
    chunk_start = acquisition_start + dt.timedelta(seconds=(chunk_number - 1) * n_pages * cadence_s)
    raw_idx = int(round((target_dt - chunk_start).total_seconds() / cadence_s))
    idx = min(max(raw_idx, 0), n_pages - 1)
    frame_dt = chunk_start + dt.timedelta(seconds=idx * cadence_s)
    if raw_idx < 0 or raw_idx >= n_pages:
        print(
            "Warning: requested time is outside this TIFF chunk; "
            f"using nearest boundary frame {idx + 1}/{n_pages}"
        )
    return idx, frame_dt, chunk_start, cadence_s, n_pages


def read_frame(tiff_path, frame_idx):
    with tifffile.TiffFile(tiff_path) as tif:
        frame = tif.pages[frame_idx].asarray()
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame.astype(np.float32)


def load_hera_skymap(az_path, el_path, site_lat, site_lon, map_alt):
    az = fits.getdata(az_path).astype(float)
    el = fits.getdata(el_path).astype(float)
    if az.shape != el.shape:
        raise ValueError(f"Az/El shape mismatch: {az.shape} vs {el.shape}")
    mask = (~np.isfinite(az)) | (~np.isfinite(el)) | (el < 15.0)
    lat, lon = azel2geo(site_lat, site_lon, az, el, alt=map_alt)
    return {
        "site_lat": site_lat,
        "site_lon": site_lon,
        "azmt": az,
        "elev": el,
        "mask": mask,
        "lat": lat,
        "lon": lon,
        "map_alt_km": map_alt,
    }


def load_vee_frame(date, target_dt):
    candidates = get_site_tiff_candidates("VEE", date, "green", mission="GNEISS")
    frame, _display = load_best_frame_from_tiffs(
        "VEE",
        candidates,
        target_dt,
        frame_interval=FRAME_INTERVAL_SECONDS_GREEN,
        color="green",
    )
    return frame


def output_path(time_arg):
    token = sanitize_time_for_filename(time_arg)
    return mission_output_dir("GNEISS", color="green", date=DEFAULT_DATE) / f"HER_on_VEE_mapped_{DEFAULT_DATE}_{token}.png"


def corrected_her_time(file_dt):
    return file_dt + dt.timedelta(seconds=HER_TIMESTAMP_OFFSET_SECONDS)


def her_file_time_from_corrected(corrected_dt):
    return corrected_dt - dt.timedelta(seconds=HER_TIMESTAMP_OFFSET_SECONDS)


def main():
    args = parse_args()
    args.date = DEFAULT_DATE
    args.color = "green"
    args.mission = "GNEISS"
    target_dt = parse_date_and_time(args.date, args.time)
    her_file_dt = her_file_time_from_corrected(target_dt)
    frame_idx, frame_file_dt, chunk_start, cadence_s, n_pages = frame_index_for_time(DEFAULT_TIFF, args.date, her_file_dt)
    frame_corrected_dt = corrected_her_time(frame_file_dt)
    her_frame = read_frame(DEFAULT_TIFF, frame_idx)
    vee_frame = load_vee_frame(args.date, target_dt)
    skymaps = load_skymaps({"VEE"}, color="green", mission="GNEISS", green_alt=DEFAULT_GREEN_ALT_KM)
    skymaps["HER"] = load_hera_skymap(DEFAULT_AZ, DEFAULT_EL, DEFAULT_SITE_LAT, DEFAULT_SITE_LON, DEFAULT_GREEN_ALT_KM)
    for site_skymap in skymaps.values():
        site_skymap["extra_masks"] = {}

    print(f"TIFF chunk start: {chunk_start.isoformat()}")
    print(f"Cadence: {cadence_s:.9f} s, pages: {n_pages}")
    print(
        f"Selected HER frame {frame_idx + 1}/{n_pages} at corrected {frame_corrected_dt.strftime('%H:%M:%S.%f')[:-3]} "
        f"(raw {frame_file_dt.strftime('%H:%M:%S.%f')[:-3]}, target corrected {target_dt.strftime('%H:%M:%S.%f')[:-3]})"
    )

    plot_map(
        skymaps,
        {"VEE": vee_frame, "HER": her_frame},
        {},
        output_path=output_path(args.time),
        map_time=args.time,
        map_date=args.date,
        bounds=DEFAULT_BOUNDS,
        color="green",
        imgs_raw={"VEE": vee_frame, "HER": her_frame},
        norm_limits=None,
        colorbar_scale="log",
        colorbar_color="monochromatic",
        apex=Apex(),
        plot_receivers=False,
        plot_ipps=False,
        pretty=False,
        plot_geodetic_traj=False,
        shared_norm=True,
        mission="GNEISS",
        green_alt=DEFAULT_GREEN_ALT_KM,
    )


if __name__ == "__main__":
    main()
