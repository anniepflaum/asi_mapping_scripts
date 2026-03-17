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
import os
import numpy as np
import numpy.ma as ma
import h5py
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from apexpy import Apex
import magcoordmap as mcm  # Leslie Lamarche's custom module for magnetic grid lines
from PIL import Image
from io import BytesIO
import datetime as dt
import time
import check_intersect as ci
import generate_skymap as skymap
import tifffile
import json
import re
import csv
from glob import glob
from inspect import currentframe
from fetch_url import closest_amisr_png_url
try:
    from resolvedvelocities.ResolveVectorsLat import ResolveVectorsLat
except ImportError:
    ResolveVectorsLat = None

# Suppress runtime and user warnings (optional, comment out if you want to see warnings)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

apex = Apex()

FRAME_INTERVAL_SECONDS_GREEN = 0.3
FRAME_INTERVAL_SECONDS_RED = 0.9
REFERENCE_NORMALIZATION_TIME = "102400.0"
TIME_WITH_OPTIONAL_FRACTION_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(?:\.(\d{1,6}))?$")


def parse_hhmmss_fractional(time_str):
    """
    Parse a time string in HHMMSS or HHMMSS.<fraction> format.
    Returns (hour, minute, second, microsecond, frac_str_or_none).
    """
    m = TIME_WITH_OPTIONAL_FRACTION_RE.fullmatch(str(time_str).strip())
    if not m:
        raise ValueError("time must be HHMMSS or HHMMSS.s (up to 6 fractional digits)")
    hour = int(m.group(1))
    minute = int(m.group(2))
    second = int(m.group(3))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError("time components out of range (HH: 00-23, MM/SS: 00-59)")
    frac = m.group(4)
    microsecond = int(frac.ljust(6, "0")) if frac else 0
    return hour, minute, second, microsecond, frac


def parse_date_and_time(date_str, time_str):
    """Parse YYYYMMDD and HHMMSS(.fraction) into a datetime."""
    if not re.fullmatch(r"\d{8}", str(date_str).strip()):
        raise ValueError("date must be YYYYMMDD")
    base_date = dt.datetime.strptime(date_str, "%Y%m%d")
    hour, minute, second, microsecond, _ = parse_hhmmss_fractional(time_str)
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)


def hhmmss_fractional_to_seconds(time_str):
    """Convert HHMMSS(.fraction) to seconds since midnight as float."""
    hour, minute, second, microsecond, _ = parse_hhmmss_fractional(time_str)
    return hour * 3600.0 + minute * 60.0 + second + microsecond / 1e6


def format_time_label(time_str):
    """Format HHMMSS(.fraction) as HH:MM:SS(.fraction) for plot labels."""
    hour, minute, second, _, frac = parse_hhmmss_fractional(time_str)
    base = f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{base}.{frac}" if frac else base


def sanitize_time_for_filename(time_str):
    """Return a filename-safe time token preserving fractional seconds."""
    return str(time_str).replace(".", "p")


def parse_tiff_start_datetime(tiff_path):
    """Parse TIFF start datetime from filename pattern *_YYYYMMDD_HHMMSS.tiff."""
    fname = os.path.basename(tiff_path)
    m = re.search(r'_(\d{8})_(\d{6})\.tiff$', fname)
    if not m:
        raise ValueError(f"Could not parse start time from TIFF filename: {fname}")
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def get_site_tiff_candidates(site, date_str, color, override_dirs=None):
    """
    Return candidate TIFF paths for a site.
    Priority:
    1) TIFFs discovered from explicit CLI folder(s) (`override_dirs`) if provided.
    2) Auto-discovered TIFFs for the requested date in ../raw_tiffs/<COLOR>/<SITE>/.
    """
    if isinstance(override_dirs, str):
        override_dirs = [override_dirs]
    dirs_to_search = list(override_dirs) if override_dirs else [f"../raw_tiffs/{color}/{site}"]
    # Be tolerant of BRV/BVR naming drift in folder and filename conventions.
    if site == "BVR":
        alt_dir = f"../raw_tiffs/{color}/BRV"
        if alt_dir not in dirs_to_search:
            dirs_to_search.append(alt_dir)
        site_prefixes = ["BVR", "BRV"]
    else:
        site_prefixes = [site]

    matched_paths = []
    searched_patterns = []
    for folder in dirs_to_search:
        for prefix in site_prefixes:
            patterns = [
                os.path.join(folder, f"{prefix}_558_{date_str}_*.tiff"),
                os.path.join(folder, f"*_{date_str}_*.tiff"),
            ]
            for pattern in patterns:
                searched_patterns.append(pattern)
                matched_paths.extend(glob(pattern))

    unique_paths = sorted(set(matched_paths))
    if not unique_paths:
        print(f"{site}: no TIFFs matched for date {date_str}.")
        print(f"{site}: searched patterns: {', '.join(searched_patterns)}")
    return unique_paths


def load_best_frame_from_tiffs(site, tiff_paths, target_dt, frame_interval=FRAME_INTERVAL_SECONDS_GREEN, color="green", verbose=True):
    """
    Search all candidate TIFF tiles and load the frame closest to target_dt.
    Prefer TIFFs whose coverage includes target_dt; if none do, use nearest boundary frame.
    Returns (raw_image, normalized_image), both float32 arrays.
    """
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found for {site}")

    candidates = []
    errors = []
    for path in tiff_paths:
        try:
            start_dt = parse_tiff_start_datetime(path)
            with tifffile.TiffFile(path) as tif:
                n_frames = len(tif.pages)
            end_dt = start_dt + dt.timedelta(seconds=(n_frames - 1) * frame_interval)
            raw_idx = int(round((target_dt - start_dt).total_seconds() / frame_interval))
            idx = min(max(raw_idx, 0), n_frames - 1)
            frame_dt = start_dt + dt.timedelta(seconds=idx * frame_interval)
            delta_s = abs((frame_dt - target_dt).total_seconds())
            in_range = start_dt <= target_dt <= end_dt
            candidates.append({
                'path': path,
                'idx': idx,
                'n_frames': n_frames,
                'delta_s': delta_s,
                'frame_dt': frame_dt,
                'in_range': in_range,
            })
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if not candidates:
        raise RuntimeError(f"All TIFF candidates failed for {site}: {'; '.join(errors)}")

    in_range_candidates = [c for c in candidates if c['in_range']]
    if in_range_candidates:
        best = min(in_range_candidates, key=lambda c: c['delta_s'])
    else:
        best = min(candidates, key=lambda c: c['delta_s'])
        if verbose:
            print(
                f"{site}: requested time outside all tile ranges; "
                f"using nearest boundary frame."
            )

    if verbose:
        print(
            f"{site}: {os.path.basename(best['path'])} frame "
            f"{best['idx'] + 1}/{best['n_frames']} "
            f"(delta {best['delta_s']:.2f}s)"
        )

    with tifffile.TiffFile(best['path']) as tif:
        im = tif.pages[best['idx']].asarray()
    if im.ndim == 3:
        im = im[:, :, 0]
    im_raw = im.astype(np.float32)

    # Display normalization is applied globally across sites in plot_fast().
    im_boost = im_raw.copy()
    return im_raw, im_boost.astype(np.float32)


def sample_raw_brightness_at_latlon(site, lat0, lon0, skymaps, imgs_raw):
    """
    Sample raw image brightness at (lat0, lon0) using the mean of the 25
    closest valid pixels for a site.
    Returns dict with brightness, percentile, and nearest-pixel metadata, or None if unavailable.
    """
    if site not in skymaps or site not in imgs_raw:
        return None
    lat_grid = skymaps[site]['lat']
    lon_grid = skymaps[site]['lon']
    raw_img = imgs_raw[site]
    valid = (~skymaps[site]['mask']) & np.isfinite(lat_grid) & np.isfinite(lon_grid) & np.isfinite(raw_img)
    for m in skymaps[site].get('extra_masks', {}).values():
        valid &= ~m
    if not np.any(valid):
        return None
    d2 = (lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2
    valid_flat = valid.ravel()
    d2_flat = d2.ravel()
    valid_idx = np.flatnonzero(valid_flat)
    if valid_idx.size == 0:
        return None
    k = min(25, valid_idx.size)
    nearest_order = np.argpartition(d2_flat[valid_idx], k - 1)[:k]
    nearest_idx = valid_idx[nearest_order]
    nearest_d2 = d2_flat[nearest_idx]
    raw_flat = raw_img.ravel()
    raw_val = float(np.mean(raw_flat[nearest_idx]))
    valid_vals = raw_img[valid]
    # Percentile rank within valid mapped pixels for this site.
    percentile = 100.0 * float(np.mean(valid_vals <= raw_val))
    closest_idx = int(nearest_idx[np.argmin(nearest_d2)])
    rr, cc = np.unravel_index(closest_idx, d2.shape)
    return {
        'site': site,
        'raw_brightness': raw_val,
        'percentile': percentile,
        'distance_deg': float(np.mean(np.sqrt(nearest_d2))),
        'n_pixels': int(k),
        'row': int(rr),
        'col': int(cc),
    }


def best_rocket_brightness(lat0, lon0, skymaps, imgs_raw):
    """
    Return nearest-pixel raw brightness info across all available sites for a rocket location.
    """
    samples = []
    for site in imgs_raw.keys():
        s = sample_raw_brightness_at_latlon(site, lat0, lon0, skymaps, imgs_raw)
        if s is not None:
            samples.append(s)
    if not samples:
        return None
    return min(samples, key=lambda x: x['distance_deg'])


def compute_reference_norm_limits(skymaps, selected_sites, date, ref_time_str, color, frame_interval, tiff_overrides):
    """
    Compute shared normalization limits from a fixed reference time using
    post-mask main-map pixels across currently selected sites.
    """
    ref_dt = parse_date_and_time(date, ref_time_str)
    ref_imgs = {}
    for site in ['ARV', 'VEE', 'BVR']:
        if site not in selected_sites:
            continue
        try:
            tiff_candidates = get_site_tiff_candidates(site, date, color, tiff_overrides[site])
            im_raw, _im_display = load_best_frame_from_tiffs(
                site,
                tiff_candidates,
                ref_dt,
                frame_interval=frame_interval,
                color=color,
                verbose=False,
            )
            ref_imgs[site] = im_raw
        except Exception as exc:
            print(f"{site}: reference normalization frame unavailable at {ref_time_str}: {exc}")
    if 'PKR' in selected_sites:
        try:
            pkr_lookup_time = ref_dt.strftime("%H%M%S")
            url_pkr = closest_amisr_png_url('PKR', date, pkr_lookup_time, color=color)
            ref_imgs['PKR'] = retrieve_image(url_pkr)
        except Exception as exc:
            print(f"PKR: reference normalization frame unavailable at {ref_time_str}: {exc}")

    norm_pool = []
    for site, img in ref_imgs.items():
        side_img = img.copy()
        side_img[skymaps[site]['mask']] = np.nan
        main_img = side_img.copy()
        for m in skymaps[site]['extra_masks'].values():
            main_img[m] = np.nan
        vals = main_img[np.isfinite(main_img)]
        if vals.size > 0:
            norm_pool.append(vals)

    if not norm_pool:
        print(f"Reference normalization at {ref_time_str} found no valid pixels; falling back to per-frame normalization.")
        return None, None

    all_vals = np.concatenate(norm_pool)
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
    print(f"Normalization fixed to {ref_time_str}: vmin={vmin:.2f}, vmax={vmax:.2f}")
    return vmin, vmax

def load_skymaps(selected_sites=None, color="green"):
    """
    Loads latitude and longitude mapping arrays for each site using skymap module.
    Returns a dictionary of skymaps for each site, including azimuth/elevation and masks.
    """
    skymaps = dict()
    if selected_sites is None:
        selected_sites = {'ARV', 'PKR', 'VEE', 'BVR'}
    if 'ARV' in selected_sites:
        lat, lon, az, el, mask = skymap.load_ARV()
        skymaps['ARV'] = {'site_lat': lat, 'site_lon': lon, 'azmt': az, 'elev': el, 'mask': mask}
    if 'VEE' in selected_sites:
        lat, lon, az, el, mask = skymap.load_VEE()
        skymaps['VEE'] = {'site_lat': lat, 'site_lon': lon, 'azmt': az, 'elev': el, 'mask': mask}
    if 'BVR' in selected_sites:
        lat, lon, az, el, mask = skymap.load_BVR()
        skymaps['BVR'] = {'site_lat': lat, 'site_lon': lon, 'azmt': az, 'elev': el, 'mask': mask}
    if 'PKR' in selected_sites:
        lat, lon, az, el, mask = skymap.load_PKR()
        skymaps['PKR'] = {'site_lat': lat, 'site_lon': lon, 'azmt': az, 'elev': el, 'mask': mask}
    map_alt_km = 200.0 if str(color).lower() == "red" else 110.0
    for sm in skymaps.values():
        lat, lon = skymap.azel2geo(sm['site_lat'], sm['site_lon'], sm['azmt'], sm['elev'], alt=map_alt_km)
        sm['lat'] = lat
        sm['lon'] = lon
    return skymaps


def scale_uv(lon, lat, u, v):
    """
    Adjusts vector scaling/rotation for cartopy quiver plots to account for latitude distortion.
    """
    us = u / np.cos(lat * np.pi / 180.)
    vs = v
    sf = np.sqrt(u ** 2 + v ** 2) / np.sqrt(us ** 2 + vs ** 2)
    return us * sf, vs * sf


def retrieve_image(url):
    """
    Downloads a single-channel image from a URL and returns it as a float32 numpy array.
    Used for PKR site images.
    """
    print(f"PKR: {url} ...")
    resp = requests.get(url, verify=False)  # verify=False disables SSL cert check (safe for public data)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        # If image is RGB, take only one channel (shouldn't be needed, but for safety)
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_traj(filename, map_time=None):
    """
    Loads rocket trajectory from an NSROC attitude-solution CSV.
    Maps lat/lon to 110 km altitude using Apex.
    Returns full trajectory, minute marks, apogee location, and optionally the trajectory point corresponding to map_time.
    map_time: string in HHMMSS(.fraction) format (requested map time)
    """
    filename_lower = filename.lower()
    launch_start = None
    if not filename_lower.endswith(".csv"):
        raise ValueError(f"Trajectory input must be an NSROC attitude-solution CSV: {filename}")

    t0_re = re.compile(r"^T0:\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s+UTC$")
    with open(filename, "r", encoding="utf-8") as fd:
        lines = fd.readlines()

    for line in lines:
        stripped = line.strip()
        match = t0_re.match(stripped)
        if match:
            launch_start = f"{match.group(4)}{match.group(5)}{match.group(6)}"
            if match.group(7):
                launch_start = f"{launch_start}.{match.group(7)}"
            print(f"launch_start: {launch_start}")
            break

    header_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Time,") and "Latgd" in stripped and "Long" in stripped and "Alt" in stripped:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not find trajectory header in {filename}")

    reader = csv.reader(lines[header_idx:])
    header = next(reader, None)
    next(reader, None)  # units row
    if header is None:
        raise ValueError(f"Could not read trajectory header in {filename}")
    columns = [column.strip() for column in header]

    rows = []
    for values in reader:
        if not values or not any(value.strip() for value in values):
            continue
        row = {column: value.strip() for column, value in zip(columns, values)}
        if not row.get("Time"):
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No trajectory samples found in {filename}")

    times = np.array([float(row["Time"]) for row in rows], dtype=float)
    lats = np.array([float(row["Latgd"]) for row in rows], dtype=float)
    lons = np.array([float(row["Long"]) for row in rows], dtype=float)
    alts = np.array([float(row["Alt"]) for row in rows], dtype=float) / 1000.0

    lats, lons, _ = apex.map_to_height(lats, lons, alts, 110.)
    idx = np.argwhere(np.isclose(times % 60, 0.0, atol=0.05))
    timem = times[idx].squeeze()
    latsm = lats[idx].squeeze()
    lonsm = lons[idx].squeeze()
    aidx = np.argmax(alts)
    lata = lats[aidx]
    lona = lons[aidx]

    traj_time_idx = None
    traj_lat_at_map = None
    traj_lon_at_map = None
    if map_time is not None and launch_start is not None:
        # Convert map_time and launch_start to seconds since midnight
        map_sec = hhmmss_fractional_to_seconds(map_time)
        launch_sec = hhmmss_fractional_to_seconds(str(launch_start).zfill(6))
        rel_sec = map_sec - launch_sec
        # Find closest time in trajectory
        if rel_sec >= 0 and rel_sec <= times[-1]:
            traj_time_idx = np.argmin(np.abs(times - rel_sec))
            traj_lat_at_map = lats[traj_time_idx]
            traj_lon_at_map = lons[traj_time_idx]

    return lats, lons, latsm, lonsm, lata, lona, traj_lat_at_map, traj_lon_at_map


def retrieve_pfisr():
    """
    Downloads and processes PFISR data.
    Returns electron density, velocity, and location arrays for plotting.
    """
    if ResolveVectorsLat is None:
        raise ImportError("resolvedvelocities module is not installed")
    url = "https://amisr.com/realtime/plots/fitted/single/dtc3/current.h5"
    #print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open('pfisr_latest.h5', 'wb') as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)
    pfisr_file = 'pfisr_latest.h5'
    with h5py.File(pfisr_file, 'r') as h5:
        ne = h5['FittedParams/Ne'][:]
        dne = h5['FittedParams/dNe'][:]
        glat = h5['Geomag/Latitude'][:]
        glon = h5['Geomag/Longitude'][:]
        galt = h5['Geomag/Altitude'][:]
    ne[dne > ne] = np.nan
    vvels = ResolveVectorsLat('vvels_config.ini')
    vvels.transform()
    vvels.bin_data_mlat()
    vvels.compute_vector_velocity()
    vvels.compute_electric_field()
    vvels.compute_geodetic_output()
    glat, glon, _ = apex.map_to_height(glat, glon, galt / 1000., 110.)
    aidx = np.argmin(np.abs(vvels.outalt - 110.))
    vv = vvels.Velocity_gd[0, aidx, :, :]
    vm = vvels.Vgd_mag[0, aidx, :]
    ve = vvels.Vgd_mag_err[0, aidx, :]
    vlat = vvels.bin_glat[aidx, :]
    vlon = vvels.bin_glon[aidx, :]
    vv[ve > vm, :] = [np.nan, np.nan, np.nan]
    vm[ve > vm] = np.nan
    pfisr_data = {'ne': ne, 'glat': glat, 'glon': glon, 'vel': vv, 'mag': vm, 'vlat': vlat, 'vlon': vlon}
    return pfisr_data



###############################################################
# --- PLOTTING FUNCTIONS ---
###############################################################

def plot_fast(skymaps, imgs, pfisr, output_path=None, map_time=None, bounds=None, color="green", imgs_raw=None, norm_limits=None):
    """
    Fast plotting mode: overlays ASI images, PFISR data, and rocket trajectories on a simple map.
    Used for quick visualization without Cartopy.
    """
    # Load coastline data
    coastlons = np.loadtxt('coastlon.txt')
    coastlats = np.loadtxt('coastlat.txt')
    # Create figure and main axis
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(coastlons, coastlats, color='black')
    lon_min, lon_max, lat_min, lat_max = bounds if bounds is not None else (-170, -135, 57.5, 72)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_xlim(xmin=lon_min, xmax=lon_max)
    ax.set_aspect(2.2)
    ax.grid()
    # Create sidebar axes for each site
    ax1 = dict()
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1])
        ax1[site].plot(coastlons, coastlats, color='black')
        ax1[site].set_ylim(ymin=lat_min, ymax=lat_max)
        ax1[site].set_xlim(xmin=lon_min, xmax=lon_max)
        ax1[site].set_aspect(2.2)
        ax1[site].grid()
        ax1[site].set_title(site)
    # Build masked images first, then normalize once across all sites.
    side_images = {}
    main_images = {}
    norm_pool = []
    for site, img in imgs.items():
        side_img = img.copy()
        side_img[skymaps[site]['mask']] = np.nan
        main_img = side_img.copy()
        for m in skymaps[site]['extra_masks'].values():
            main_img[m] = np.nan
        side_images[site] = side_img
        main_images[site] = main_img
        vals = main_img[np.isfinite(main_img)]
        if vals.size > 0:
            norm_pool.append(vals)

    if norm_limits is not None and norm_limits[0] is not None and norm_limits[1] is not None:
        global_vmin, global_vmax = norm_limits
    else:
        global_vmin = None
        global_vmax = None
        if norm_pool:
            all_vals = np.concatenate(norm_pool)
            global_vmin = float(np.percentile(all_vals, 1))
            global_vmax = float(np.percentile(all_vals, 99))
            if not np.isfinite(global_vmin) or not np.isfinite(global_vmax) or global_vmax <= global_vmin:
                global_vmin = float(np.nanmin(all_vals))
                global_vmax = float(np.nanmax(all_vals))

    # Plot each site's mapped image with one shared normalization.
    for site in imgs.keys():
        main_img = main_images[site]
        side_img = side_images[site]
        if global_vmin is None or global_vmax is None:
            im_handle = ax.pcolor(skymaps[site]['lon'], skymaps[site]['lat'], main_img)
            ax1[site].pcolor(skymaps[site]['lon'], skymaps[site]['lat'], side_img)
        else:
            im_handle = ax.pcolor(
                skymaps[site]['lon'],
                skymaps[site]['lat'],
                main_img,
                vmin=global_vmin,
                vmax=global_vmax,
            )
            ax1[site].pcolor(
                skymaps[site]['lon'],
                skymaps[site]['lat'],
                side_img,
                vmin=global_vmin,
                vmax=global_vmax,
            )
    # Plot rocket trajectories and minute marks
    lat1, lon1, latm1, lonm1, lata1, lona1, lat_map1, lon_map1 = load_traj('36.397_AttitudeSolution.csv', map_time=map_time)
    lat2, lon2, latm2, lonm2, lata2, lona2, lat_map2, lon_map2 = load_traj('36.398_AttitudeSolution.csv', map_time=map_time)
    ax.plot(lon1, lat1, color='red', label='GNEISS trajectory', zorder=7)
    ax.scatter(lonm1, latm1, color='red', s=15, zorder=7)
    ax.plot(lon2, lat2, color='red', zorder=7)
    ax.scatter(lonm2, latm2, color='red', s=15, zorder=7)
    # Mark position at map time if available
    if lat_map1 is not None and lon_map1 is not None:
        ax.scatter(lon_map1, lat_map1, color='orange', s=50, marker='o', zorder=8, label='Position at map time')
    if lat_map2 is not None and lon_map2 is not None:
        ax.scatter(lon_map2, lat_map2, color='orange', s=50, marker='o', zorder=8)
    # Raw auroral brightness at rocket positions (nearest valid mapped pixel)
    bright1 = None
    bright2 = None
    if imgs_raw is not None:
        if lat_map1 is not None and lon_map1 is not None:
            bright1 = best_rocket_brightness(lat_map1, lon_map1, skymaps, imgs_raw)
        if lat_map2 is not None and lon_map2 is not None:
            bright2 = best_rocket_brightness(lat_map2, lon_map2, skymaps, imgs_raw)
        if bright1 is not None:
            print(
                f"Rocket Left brightness percentile: {bright1['site']} P={bright1['percentile']:.1f} "
                f"(nearest {bright1['distance_deg']:.3f} deg)"
            )
        if bright2 is not None:
            print(
                f"Rocket Right brightness percentile: {bright2['site']} P={bright2['percentile']:.1f} "
                f"(nearest {bright2['distance_deg']:.3f} deg)"
            )
    # Add plot text for date/time
    frame = currentframe()
    args = frame.f_back.f_locals.get('args', None)
    if args is not None:
        date_str = args.date
        time_str = args.time
        label_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}"
    else:
        label_str = ""
    txt = ax.text(0.99, 0.01, label_str,
                 transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.set_title(f"Mapped ASIs and GNEISS trajectory ({color} channel)")
    ax.legend(loc='upper right')
    #ax.quiverkey(qp, 0.1, 0.9, 500., '500 m/s', transform=ax.transAxes)
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation='vertical')
    cbar.set_label('Green Channel Intensity')
    # Place rocket brightness markers on the colorbar at percentile positions.
    marker_specs = []
    if bright1 is not None:
        marker_specs.append(("L", bright1, "orange"))
    if bright2 is not None:
        marker_specs.append(("R", bright2, "gold"))
    for tag, b, color_marker in marker_specs:
        y = np.clip(b['percentile'] / 100.0, 0.0, 1.0)
        # Draw in colorbar-axes coordinates so marker is guaranteed on top of the bar.
        cbar.ax.plot(
            [0.0, 1.0],
            [y, y],
            transform=cbar.ax.transAxes,
            color='black',
            linewidth=4.0,
            zorder=1000,
            solid_capstyle='butt',
            clip_on=False,
        )
        cbar.ax.plot(
            [0.0, 1.0],
            [y, y],
            transform=cbar.ax.transAxes,
            color='white',
            linewidth=2.2,
            zorder=1001,
            solid_capstyle='butt',
            clip_on=False,
        )
        cbar.ax.text(
            -0.05,
            y,
            f"{tag}: {b['site']} P{b['percentile']:.1f}",
            transform=cbar.ax.transAxes,
            color='black',
            fontsize=8,
            va='center',
            ha='right',
            clip_on=False,
        )
    '''
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(pfisr_handle, cax=cax, orientation='vertical')
    cbar.set_label(r'Electron Density (m$^{-3}$)')
    '''
    plt.tight_layout()
    # Save figure
    if output_path is None:
        if args is not None:
            output_path = f"../mapped/GNEISS_launch_science_fast_{date_str}_{sanitize_time_for_filename(time_str)}.png"
        else:
            output_path = f"../mapped/GNEISS_launch_science_fast_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
    # plt.show()


def plot_pretty(skymaps, imgs, pfisr, output_path=None, bounds=None, color="green"):
    """
    Pretty plotting mode: overlays ASI images, PFISR data, and rocket trajectories on a Cartopy map.
    Used for publication-quality visualization.
    """
    # Set up Cartopy projection
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    # Main map axis
    ax = fig.add_subplot(gs[:, 0], projection=proj)
    lon_min, lon_max, lat_min, lat_max = bounds if bounds is not None else (-170, -140, 57, 72)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    ax.gridlines()
    # Add magnetic grid lines
    mcm.maggridlines(ax, apex=apex, apex_height=110.)
    # Sidebar axes for each site
    ax1 = dict()
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1], projection=proj)
        ax1[site].coastlines()
        ax1[site].gridlines()
        mcm.maggridlines(ax1[site], apex=apex, apex_height=110.)
        ax1[site].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax1[site].set_title(site)
    # Plot each site's mapped image
    for site, img in imgs.items():
        img[skymaps[site]['mask']] = np.nan
        im = img.copy()
        lat = skymaps[site]['lat'].copy()
        lon = skymaps[site]['lon'].copy()
        for m in skymaps[site]['extra_masks'].values():
            im[m] = np.nan
        img_flat = img[~skymaps[site]['mask']].flatten()
        lon_flat = skymaps[site]['lon'][~skymaps[site]['mask']].flatten()
        lat_flat = skymaps[site]['lat'][~skymaps[site]['mask']].flatten()
        im_handle = ax1[site].tripcolor(lon_flat, lat_flat, img_flat, zorder=3, transform=ccrs.PlateCarree())
        imf = im[np.isfinite(im)].flatten()
        latf = lat[np.isfinite(im)].flatten()
        lonf = lon[np.isfinite(im)].flatten()
        ax.tripcolor(lonf, latf, imf, transform=ccrs.PlateCarree())
    # Plot PFISR data
    '''
    print('PFISR')
    pfisr_handle = ax.scatter(pfisr['glon'], pfisr['glat'], c=pfisr['ne'], zorder=6, cmap='jet', transform=ccrs.Geodetic())
    u, v = scale_uv(pfisr['vlon'], pfisr['vlat'], pfisr['vel'][:, 0], pfisr['vel'][:, 1], vmin=0, vmax=4e11)
    qp = ax.quiver(pfisr['vlon'], pfisr['vlat'], u, v, zorder=7, scale=5000, width=0.005, transform=ccrs.PlateCarree())
    '''
    # Plot rocket trajectories and minute marks
    print('Trajectory')
    lat1, lon1, latm1, lonm1, lata1, lona1, lat_map1, lon_map1 = load_traj('36.397_AttitudeSolution.csv')
    lat2, lon2, latm2, lonm2, lata2, lona2, lat_map2, lon_map2 = load_traj('36.398_AttitudeSolution.csv')
    ax.plot(lon1, lat1, color='red', label='GNEISS trajectory', transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lonm1, latm1, color='red', s=15, transform=ccrs.PlateCarree(), zorder=7)
    ax.plot(lon2, lat2, color='red', transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lonm2, latm2, color='red', s=15, transform=ccrs.PlateCarree(), zorder=7)
    # Mark position at map time if available
    if lat_map1 is not None and lon_map1 is not None:
        ax.scatter(lon_map1, lat_map1, color='orange', s=50, marker='o', zorder=8, label='Position at map time')
    if lat_map2 is not None and lon_map2 is not None:
        ax.scatter(lon_map2, lat_map2, color='orange', s=50, marker='o', zorder=8)
    # Add plot text for date/time
    txt = ax.text(0.99, 0.01, dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
                 transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.set_title(f"Mapped ASIs and GNEISS trajectory ({color} channel)")
    ax.legend(loc='upper right')
    #ax.quiverkey(qp, 0.1, 0.9, 500., '500 m/s', transform=ax.transAxes)
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation='vertical')
    cbar.set_label('Green Channel Intensity')
    '''
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(pfisr_handle, cax=cax, orientation='vertical')
    cbar.set_label(r'Electron Density (m$^{-3}$)')
    '''
    plt.tight_layout()
    # Save figure
    if output_path is None:
        frame = currentframe()
        args = frame.f_back.f_locals.get('args', None)
        if args is not None:
            date_str = args.date
            time_str = args.time
            output_path = f"../launch_science_pretty/GNEISS_launch_science_pretty_{date_str}_{sanitize_time_for_filename(time_str)}.png"
        else:
            output_path = f"../launch_science_pretty/GNEISS_launch_science_pretty_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
    # plt.show()


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
    ap.add_argument("--arv-tiffs", nargs='*', default=None, help="Optional ARV folder(s) containing TIFF tiles")
    ap.add_argument("--vee-tiffs", nargs='*', default=None, help="Optional VEE folder(s) containing TIFF tiles")
    ap.add_argument("--bvr-tiffs", nargs='*', default=None, help="Optional BVR folder(s) containing TIFF tiles")
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
    sites = list(skymaps.keys())
    for s0 in sites:
        red_sites = sites.copy()
        red_sites.remove(s0)
        skymaps[s0]['extra_masks'] = dict()
        for s1 in red_sites:
            m0, m1 = ci.calculate_masks(skymaps[s0]['site_lat'], skymaps[s0]['site_lon'], skymaps[s0]['azmt'], skymaps[s0]['elev'], skymaps[s1]['site_lat'], skymaps[s1]['site_lon'], skymaps[s1]['azmt'], skymaps[s1]['elev'])
            skymaps[s0]['extra_masks'][s1] = m0

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
        pfisr = retrieve_pfisr()
    except Exception as e:
        print(f"Could not retrieve PFISR data: {e}")

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED

    # --- Process TIFF-backed sites: search multiple tiles and select closest frame ---
    tiff_overrides = {
        'ARV': args.arv_tiffs,
        'VEE': args.vee_tiffs,
        'BVR': args.bvr_tiffs,
    }
    fixed_norm_limits = compute_reference_norm_limits(
        skymaps,
        selected_sites,
        date,
        REFERENCE_NORMALIZATION_TIME,
        args.color,
        frame_interval,
        tiff_overrides,
    )
    for site in ['ARV', 'VEE', 'BVR']:
        if site not in selected_sites:
            continue
        try:
            tiff_candidates = get_site_tiff_candidates(site, date, args.color, tiff_overrides[site])
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
    mode_str = 'pretty' if args.pretty else 'fast'
    color = args.color
    output_path = f"../mapped/{color}/GNEISS_launch_{color}_{sites_str}_{date}_{time_token}.png"

    # --- Run downstream plotting for all processed sites ---
    if imgs:
        if args.pretty:
            plot_pretty(skymaps, imgs, pfisr, output_path=output_path, bounds=args.bounds, color=args.color)
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
            )

    tocall = time.time()
    print(f"Total run time: {tocall - ticall:.2f} s")

if __name__ == "__main__":
    main()
