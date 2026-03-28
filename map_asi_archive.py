#!/usr/bin/env python3
"""
map_asi_archive.py

Script for mapping and visualizing all-sky imager (ASI) data from multiple ground sites (ARV, VEE, BVR, PKR).
Processes local multi-page TIFFs for ARV, VEE, BVR, and fetches PKR images from the web.
Selects frames by timestamp, normalizes intensities, overlays rocket trajectories, and saves unified output.

Usage:
    python map_asi_archive_3Hz.py --time HHMMSS.s --sites ARV BVR VEE PKR

Arguments:
    --date           Date for the ASI images (format: YYYYMMDD)
    --time           Time for the ASI images (format: HHMMSS or HHMMSS.s)
    --sites          List of sites to process (default: all sites)
    --pretty         Use pretty Cartopy plotting (default: fast plotting)
"""

###############################################################
# --- Standard imports and dependencies ---
###############################################################
import argparse
from apexpy import Apex
import datetime as dt
import time
from core.masks import build_overlap_masks
from core.fetch_url import closest_amisr_png_url
from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED, REFERENCE_NORMALIZATION_TIME
from core.plot_norm import compute_reference_norm_limits
from core.plotting import plot_fast, plot_pretty
from core.remote_data import retrieve_image, retrieve_pfisr
from core.skymaps import load_skymaps
from core.time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from core.tiff_utils import get_site_tiff_candidates, load_best_frame_from_tiffs

# Suppress runtime and user warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

apex = Apex()


def main():
    """
    Main entry point: parses command-line arguments, loads skymaps, processes images for each site,
    normalizes and selects frames, overlays PFISR and rocket trajectories, and saves the mapped output.
    """
    ticall = time.time()
    # --- Parse command-line arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretty", action='store_true')  # Use pretty Cartopy plotting
    ap.add_argument("--date", required=False, type=str, default="20260210", help="Date for the ASI images (format: YYYYMMDD)")
    ap.add_argument("--time", required=True, type=str, default=dt.datetime.now(dt.UTC).strftime("%H%M%S"), help="Time for the ASI images (format: HHMMSS or HHMMSS.s)")
    ap.add_argument("--sites", nargs='*', default=['ARV', 'PKR', 'VEE', 'BVR'], help="List of sites to process (default: all sites)")
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
    ap.add_argument("--plot-receivers", action="store_true", help="Plot receiver locations from receivers.csv on the map")
    args = ap.parse_args()
    try:
        parse_hhmmss_fractional(args.time)
    except ValueError as exc:
        ap.error(f"--time {exc}")
    if args.bounds is not None:
        lon_min, lon_max, lat_min, lat_max = args.bounds
        if lon_min >= lon_max or lat_min >= lat_max:
            ap.error("--bounds must satisfy LON_MIN < LON_MAX and LAT_MIN < LAT_MAX")

    # --- Load geographic mapping for each ASI site ---
    selected_sites = set([s.upper() for s in args.sites])
    skymaps = load_skymaps(selected_sites, color=args.color)

    # --- Calculate masks for overlapping images between sites ---
    build_overlap_masks(skymaps)

    imgs = dict()  # Stores normalized display images for each site
    imgs_raw = dict()  # Stores raw image values for brightness sampling
    date = args.date
    time_str = args.time
    try:
        target_dt = parse_date_and_time(date, time_str)
    except ValueError as exc:
        ap.error(str(exc))
    time_token = sanitize_time_for_filename(time_str)

    # --- Retrieve PFISR data for overlay ---
    pfisr = {}
    try:
        pfisr = retrieve_pfisr(apex=apex)
    except Exception as e:
        print(f"Could not retrieve PFISR data: {e}")

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED

    # --- Process TIFF-backed sites: search multiple tiles and select closest frame ---
    fixed_norm_limits = compute_reference_norm_limits(
        skymaps,
        selected_sites,
        date,
        REFERENCE_NORMALIZATION_TIME,
        args.color,
        frame_interval,
        colorbar_scale=args.colorbar_scale,
    )
    for site in ['ARV', 'VEE', 'BVR']:
        if site not in selected_sites:
            continue
        try:
            tiff_candidates = get_site_tiff_candidates(site, date, args.color)
            im_raw, _im_display = load_best_frame_from_tiffs(
                site,
                tiff_candidates,
                target_dt,
                frame_interval=frame_interval,
                color=args.color,
            )
            imgs_raw[site] = im_raw
            imgs[site] = im_raw
        except Exception as e:
            print(f"Could not load {site} TIFF image: {e}")
    # --- Process PKR site: fetch image from web and store ---
    if 'PKR' in selected_sites:
        try:
            pkr_lookup_time = target_dt.strftime("%H%M%S")
            url_pkr = closest_amisr_png_url('PKR', date, pkr_lookup_time, color=args.color)
            pkr_img = retrieve_image(url_pkr)
            imgs_raw['PKR'] = pkr_img
            imgs['PKR'] = pkr_img
        except Exception as e:
            print(f"Could not fetch PKR image: {e}")

    # --- Compose output path for mapped image, include plotting mode ---
    sites_str = '_'.join(sorted(selected_sites))
    color = args.color
    output_path = f"../mapped/{color}/GNEISS_launch_{color}_{sites_str}_{date}_{time_token}.png"

    # --- Run downstream plotting for all processed sites ---
    if imgs:
        if args.pretty:
            plot_pretty(
                skymaps,
                imgs,
                pfisr,
                output_path=output_path,
                map_time=args.time,
                bounds=args.bounds,
                color=args.color,
                colorbar_scale=args.colorbar_scale,
                colorbar_color=args.colorbar_color,
                apex=apex,
                plot_receivers=args.plot_receivers,
            )
        else:
            plot_fast(
                skymaps,
                imgs,
                pfisr,
                output_path=output_path,
                map_time=args.time,
                bounds=args.bounds,
                color=args.color,
                imgs_raw=imgs_raw,
                norm_limits=fixed_norm_limits,
                colorbar_scale=args.colorbar_scale,
                colorbar_color=args.colorbar_color,
                plot_receivers=args.plot_receivers,
            )

    tocall = time.time()
    print(f"Total run time: {tocall - ticall:.2f} s")

if __name__ == "__main__":
    main()
