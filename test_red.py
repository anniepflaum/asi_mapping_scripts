#!/usr/bin/env python3
"""Plot individual red-channel ASI maps with zenith markers."""

import argparse
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

from apexpy import Apex
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import (
    DEFAULT_RED_ALT_KM,
    FRAME_INTERVAL_SECONDS_RED,
    NORMALIZATION_UPPER_PERCENTILE,
)
from core.masks import build_overlap_masks
from core.paths import COAST_LAT_PATH, COAST_LON_PATH
from core.plotting import (
    choose_image_cmap,
    draw_image_regrid,
    prepare_image_layers,
)
from core.skymaps import load_skymaps
from core.missions import mission_output_dir
from core.time_utils import parse_date_and_time, sanitize_time_for_filename
from core.tiff_utils import get_site_tiff_candidates, load_best_frame_from_tiffs
from map_asi_archive import calibrate_image_for_map, calibrated_reference_norm_limits


SITES = ("ARV", "VEE", "BVR")
DEFAULT_DATE = "20260210"
DEFAULT_BOUNDS = (-170.0, -135.0, 57.5, 72.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", required=True, help="Frame time as HHMMSS(.fraction)")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Image date as YYYYMMDD")
    parser.add_argument("--mission", choices=["GNEISS", "GIRAFF"], default="GNEISS")
    parser.add_argument("--red-wavelength", choices=["6300"], default="6300")
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=DEFAULT_BOUNDS,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    parser.add_argument("--vmax", type=float, default=NORMALIZATION_UPPER_PERCENTILE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def magnetic_zenith(apex, site_lat, site_lon, height_km):
    """Map the station along its magnetic field line to the emission layer."""
    lat, lon, _error = apex.map_to_height(
        np.asarray([site_lat], dtype=float),
        np.asarray([site_lon], dtype=float),
        np.asarray([0.0], dtype=float),
        float(height_km),
    )
    return float(lat[0]), float(lon[0])


def load_images(args, skymaps, target_dt):
    images = {}
    for site in SITES:
        candidates = get_site_tiff_candidates(
            site,
            args.date,
            "red",
            mission=args.mission,
            red_wavelength=args.red_wavelength,
        )
        raw, _display = load_best_frame_from_tiffs(
            site,
            candidates,
            target_dt,
            frame_interval=FRAME_INTERVAL_SECONDS_RED,
            color="red",
        )
        images[site], _background = calibrate_image_for_map(
            site, raw, skymaps, "red"
        )
    return images


def default_output(args):
    token = sanitize_time_for_filename(args.time)
    return (
        mission_output_dir(args.mission, color="red", date=args.date)
        / f"test_red_ARV_VEE_BVR_{args.date}_{token}.png"
    )


def main():
    args = parse_args()
    lon_min, lon_max, lat_min, lat_max = args.bounds
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("--bounds must satisfy LON_MIN < LON_MAX and LAT_MIN < LAT_MAX")
    if not 0 < args.vmax <= 100:
        raise ValueError("--vmax must be > 0 and <= 100")

    target_dt = parse_date_and_time(args.date, args.time)
    skymaps = load_skymaps(SITES, color="red", mission=args.mission)
    build_overlap_masks(skymaps)
    images = load_images(args, skymaps, target_dt)
    norm_limits = calibrated_reference_norm_limits(
        skymaps,
        set(SITES),
        args.date,
        args.time,
        "red",
        FRAME_INTERVAL_SECONDS_RED,
        colorbar_scale="log",
        mission=args.mission,
        red_wavelength=args.red_wavelength,
        upper_percentile=args.vmax,
    )
    side_images, _main_images, _site_limits, _site_norms, vmin, vmax, norm = (
        prepare_image_layers(
            skymaps,
            images,
            "log",
            norm_limits=norm_limits,
            shared_norm=True,
            upper_percentile=args.vmax,
        )
    )

    coast_lon = np.loadtxt(COAST_LON_PATH)
    coast_lat = np.loadtxt(COAST_LAT_PATH)
    cmap = choose_image_cmap("monochromatic", "red")
    apex = Apex(date=target_dt)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
    image_handle = None

    for ax, site in zip(axes, SITES):
        skymap = skymaps[site]
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect(2.2)
        image_handle = draw_image_regrid(
            ax,
            skymap["lon"],
            skymap["lat"],
            side_images[site],
            cmap,
            norm,
            vmin,
            vmax,
            ax.transData,
        )
        ax.plot(coast_lon, coast_lat, color="black", linewidth=1.0, zorder=10)

        zenith_lat = float(skymap["site_lat"])
        zenith_lon = float(skymap["site_lon"])
        mag_lat, mag_lon = magnetic_zenith(
            apex, zenith_lat, zenith_lon, DEFAULT_RED_ALT_KM
        )
        ax.scatter(
            zenith_lon,
            zenith_lat,
            marker="*",
            s=180,
            color="cyan",
            edgecolor="black",
            linewidth=0.8,
            label="Geographic zenith",
            zorder=20,
        )
        ax.scatter(
            mag_lon,
            mag_lat,
            marker="X",
            s=120,
            color="yellow",
            edgecolor="black",
            linewidth=0.8,
            label="Magnetic zenith",
            zorder=20,
        )
        ax.set_title(site, fontsize=16, fontweight="bold")
        ax.grid(True, color="0.35", alpha=0.55, linewidth=0.8, zorder=12)
        ax.set_xlabel("Longitude (degrees)")

    axes[0].set_ylabel("Latitude (degrees)")
    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.suptitle(
        f"Red ASIs at {DEFAULT_RED_ALT_KM:g} km — {target_dt:%Y-%m-%d %H:%M:%S}",
        fontsize=18,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.09, top=0.88, wspace=0.12)
    if image_handle is not None:
        colorbar_ax = fig.add_axes([0.925, 0.15, 0.015, 0.68])
        colorbar = fig.colorbar(image_handle, cax=colorbar_ax)
        colorbar.set_label("Rayleighs")

    output = args.output or default_output(args)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
