#!/usr/bin/env python3
"""
map_asi_archive.py

Script for mapping and visualizing all-sky imager (ASI) data from multiple ground sites (ARV, VEE, BVR, PKR).
Processes local multi-page TIFFs for ARV, VEE, BVR, and fetches PKR images from the web.
Selects frames by timestamp, normalizes intensities, overlays rocket trajectories, and saves unified output.


Usage:
    python map_asi_archive.py --time HHMMSS.s --sites ARV BVR VEE PKR

Arguments:
    --rocket         Rocket ID used to select the ASI image date
    --time           Time for the ASI images (format: HHMMSS or HHMMSS.s)
    --sites          List of sites to process (default: all sites)
    --pretty         Use pretty Cartopy plotting (default: fast plotting)
    --vmax           Upper percentile used as vmax for ASI normalization (default: 99)
"""

###############################################################
# --- Standard imports and dependencies ---
###############################################################
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import argparse
from apexpy import Apex
import datetime as dt
import time
import matplotlib

matplotlib.use("Agg")

from core.calibration import BACKGROUND_EDGE_BUFFER_PX, calibrate_green_image, green_calibration_factor
from core.masks import build_overlap_masks
from core.constants import (
    FRAME_INTERVAL_SECONDS_GREEN,
    FRAME_INTERVAL_SECONDS_RED,
    NORMALIZATION_LOWER_PERCENTILE,
    NORMALIZATION_UPPER_PERCENTILE,
)
from core.plotting import plot_map
from core.remote_data import load_pkr_image, retrieve_pfisr
from core.skymaps import load_skymaps
from core.missions import default_sites, mission_output_dir, resolve_mission_and_date, validate_color_and_sites
from core.time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from core.tiff_utils import get_site_tiff_candidates, load_best_frame_from_tiffs
from core.traj_utils import mapped_apex_height

# Suppress runtime and user warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

apex = Apex()


def calibrate_image_for_map(site, im_raw, skymaps, color):
    if str(color).lower() != "green":
        return im_raw, None
    calibrated, bg = calibrate_green_image(site, im_raw, skymaps[site]["mask"], edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX)
    if calibrated is None:
        print(f"{site}: no green calibration factor configured; using raw counts.")
        return im_raw, None
    print(
        f"{site}: calibrated green image using bg={bg['center']:.2f} counts, "
        f"sigma={bg['sigma']:.2f}, factor={green_calibration_factor(site):g} R s/count"
    )
    return calibrated, bg


def main():
    """
    Main entry point: parses command-line arguments, loads skymaps, processes images for each site,
    normalizes and selects frames, overlays PFISR and rocket trajectories, and saves the mapped output.
    """
    ticall = time.time()
    # --- Parse command-line arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretty", action='store_true')  # Use pretty Cartopy plotting
    ap.add_argument("--rocket", choices=["397", "398", "380", "381"], default=None, help="Rocket ID used to select the ASI image date")
    ap.add_argument("--time", required=True, type=str, default=dt.datetime.now(dt.UTC).strftime("%H%M%S"), help="Time for the ASI images (format: HHMMSS or HHMMSS.s)")
    ap.add_argument("--sites", nargs='*', default=None, help="List of sites to process (default: mission-specific sites)")
    ap.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default=None, help="Mission dataset to use for trajectories and mission-specific site assets")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel for TIFF lookup and frame timing")
    ap.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help="Optional map bounds override: lon_min lon_max lat_min lat_max",
    )
    ap.add_argument("--colorbar-scale", choices=["linear", "log"], default="log", help="Colorbar scaling for ASI intensity")
    ap.add_argument("--colorbar-color", choices=["viridis", "monochromatic"], default="monochromatic", help="Colorbar colormap")
    ap.add_argument("--render", choices=["auto", "pcolor", "points", "regrid"], default="auto", help="Image rendering mode. auto uses regrid for red and pcolor otherwise")
    ap.add_argument("--vmax", type=float, default=NORMALIZATION_UPPER_PERCENTILE, help="Upper percentile used as vmax for ASI normalization")
    ap.add_argument(
        "--no-shared-norm",
        dest="no_shared_norm",
        action="store_true",
        default=False,
        help="Disable cross-site shared brightness normalization",
    )
    ap.add_argument("--plot-receivers", action="store_true", help="Plot receiver locations from receivers.csv on the map")
    ap.add_argument("--plot-ipps", action="store_true", help="Plot receiver ionospheric pierce points on the map")
    ap.add_argument("--plot-geodetic-traj", dest="plot_geodetic_traj", action="store_true", help="Overlay the rocket trajectories in geodetic coordinates as blue traces")
    ap.add_argument("--plot-ezie", action="store_true", help="Overlay EZIE MEM trajectories mapped along magnetic field lines to 110 km")
    args = ap.parse_args()
    shared_norm = not args.no_shared_norm
    try:
        args.mission, date = resolve_mission_and_date(args.mission, args.rocket)
    except ValueError as exc:
        ap.error(str(exc))
    args.date = date
    if args.sites is None:
        args.sites = default_sites(args.mission)
    try:
        parse_hhmmss_fractional(args.time)
    except ValueError as exc:
        ap.error(f"--time {exc}")
    if args.bounds is not None:
        lon_min, lon_max, lat_min, lat_max = args.bounds
        if lon_min >= lon_max or lat_min >= lat_max:
            ap.error("--bounds must satisfy LON_MIN < LON_MAX and LAT_MIN < LAT_MAX")
    if not (NORMALIZATION_LOWER_PERCENTILE < args.vmax <= 100):
        ap.error(f"--vmax must be > {NORMALIZATION_LOWER_PERCENTILE:g} and <= 100")
    # --- Load geographic mapping for each ASI site ---
    selected_sites = set([s.upper() for s in args.sites])
    validate_color_and_sites(ap, args.mission, args.color, selected_sites, giraff_message_site="VEE TIFFs and PKR PNGs")
    skymaps = load_skymaps(selected_sites, color=args.color, mission=args.mission)

    # --- Calculate masks for overlapping images between sites ---
    build_overlap_masks(skymaps)

    imgs = dict()  # Stores normalized display images for each site
    imgs_raw = dict()  # Stores raw image values for brightness sampling
    time_str = args.time
    try:
        target_dt = parse_date_and_time(date, time_str)
    except ValueError as exc:
        ap.error(str(exc))
    time_token = sanitize_time_for_filename(time_str)

    # --- Retrieve PFISR data for overlay ---
    pfisr = {}
    try:
        pfisr = retrieve_pfisr(apex=apex, map_alt_km=mapped_apex_height(args.color))
    except Exception as e:
        if "resolvedvelocities module is not installed" not in str(e):
            print(f"Could not retrieve PFISR data: {e}")

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED

    # --- Process TIFF-backed sites: search multiple tiles and select closest frame ---
    for site in ['ARV', 'VEE', 'BVR']:
        if site not in selected_sites:
            continue
        try:
            tiff_candidates = get_site_tiff_candidates(site, date, args.color, mission=args.mission)
            im_raw, _im_display = load_best_frame_from_tiffs(
                site,
                tiff_candidates,
                target_dt,
                frame_interval=frame_interval,
                color=args.color,
            )
            im_calibrated, _bg = calibrate_image_for_map(site, im_raw, skymaps, args.color)
            imgs_raw[site] = im_calibrated
            imgs[site] = im_calibrated
        except Exception as e:
            print(f"Could not load {site} TIFF image: {e}")
    # --- Process PKR site: prefer local PNGs and fall back to archive URL ---
    if 'PKR' in selected_sites:
        try:
            pkr_lookup_time = target_dt.strftime("%H%M%S")
            pkr_img, _pkr_source, _pkr_frame_dt = load_pkr_image(date, pkr_lookup_time, color=args.color)
            imgs_raw['PKR'] = pkr_img
            imgs['PKR'] = pkr_img
        except Exception as e:
            print(f"Could not load PKR image: {e}")

    # --- Compose output path for mapped image, include plotting mode ---
    sites_str = '_'.join(sorted(selected_sites))
    color = args.color
    output_path = mission_output_dir(args.mission, color=args.color, date=date) / f"calibrated_{color}_{sites_str}_{date}_{time_token}.png"

    # --- Run downstream plotting even if no TIFFs were found ---
    plot_map(
        skymaps,
        imgs,
        pfisr,
        output_path=output_path,
        map_time=args.time,
        map_date=date,
        bounds=args.bounds,
        color=args.color,
        imgs_raw=imgs_raw,
        norm_limits=None,
        colorbar_scale=args.colorbar_scale,
        colorbar_color=args.colorbar_color,
        shared_norm=shared_norm,
        apex=apex,
        plot_receivers=args.plot_receivers,
        plot_ipps=args.plot_ipps,
        pretty=args.pretty,
        plot_geodetic_traj=args.plot_geodetic_traj,
        plot_ezie=args.plot_ezie,
        mission=args.mission,
        upper_percentile=args.vmax,
        render_mode=args.render,
        colorbar_label="Rayleighs" if args.color == "green" else None,
    )

    tocall = time.time()
    print(f"Total run time: {tocall - ticall:.2f} s")

if __name__ == "__main__":
    main()
