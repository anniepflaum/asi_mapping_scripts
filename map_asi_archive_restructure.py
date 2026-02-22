#!/usr/bin/env python3
"""
map_asi_PKR_realtime.py

This script downloads the latest all-sky imager (ASI) green channel image from the Poker Flat Research Range (PKR) in Alaska,
maps it to geographic latitude/longitude coordinates using a provided skymap, and overlays rocket trajectories.
The output is a PNG image showing the mapped green channel intensity and rocket paths.

Usage:
    python map_asi_PKR_realtime.py --skymap skymap.mat --alt-km 110 --nx 512 --ny 512

Arguments:
    --skymap         Path to the .mat file containing the geographic mapping for the ASI
    --alt-km         Altitude (in km) for the mapping grid
    --nx, --ny       Output grid size in longitude and latitude
    --padding-deg    Optional: extra padding (in degrees) around the mapped region
    --lon-convention Optional: longitude format
"""

# --- Standard imports ---
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
from fetch_url import closest_amisr_png_url
from resolvedvelocities.ResolveVectorsLat import ResolveVectorsLat

# Suppress runtime and user warnings (optional, comment out if you want to see warnings)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

apex = Apex()

def load_skymaps(selected_sites=None):
    """
    Load the latitude and longitude mapping arrays from the skymap.mat file for a given altitude.
    Returns a dictionary of skymaps for each site.
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
    for sm in skymaps.values():
        lat, lon = skymap.azel2geo(sm['site_lat'], sm['site_lon'], sm['azmt'], sm['elev'], alt=110.)
        sm['lat'] = lat
        sm['lon'] = lon
    return skymaps


def scale_uv(lon, lat, u, v):
    """
    Fixes vector scaling/rotation for cartopy quiver plots.
    """
    us = u / np.cos(lat * np.pi / 180.)
    vs = v
    sf = np.sqrt(u ** 2 + v ** 2) / np.sqrt(us ** 2 + vs ** 2)
    return us * sf, vs * sf


def retrieve_image(url):
    """
    Download and return a single-channel image from a URL as a float32 numpy array.
    """
    print(f"Downloading {url} ...")
    resp = requests.get(url, verify=False)  # verify=False disables SSL cert check (safe for public data)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        # If image is RGB, take only one channel (shouldn't be needed, but for safety)
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_traj(filename):
    """
    Load latitude and longitude columns from a trajectory text file.
    Returns mapped lats/lons at 110 km, and apogee location.
    """
    times, lats, lons, alts = np.loadtxt(filename, skiprows=1, unpack=True)
    lats, lons, _ = apex.map_to_height(lats, lons, alts, 110.)
    idx = np.argwhere(times % 60 == 0)
    timem = times[idx].squeeze()
    latsm = lats[idx].squeeze()
    lonsm = lons[idx].squeeze()
    aidx = np.argmax(alts)
    lata = lats[aidx]
    lona = lons[aidx]
    return lats, lons, latsm, lonsm, lata, lona


def retrieve_pfisr():
    """
    Download and process PFISR data, returning a dictionary of relevant arrays.
    """
    url = "https://amisr.com/realtime/plots/fitted/single/dtc3/current.h5"
    print(f"Downloading {url} ...")
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


# --- PLOTTING FUNCTIONS ---

def plot_fast(skymaps, imgs, pfisr):
    """
    Fast plotting mode: overlays images, PFISR, and trajectories on a simple map.
    """
    coastlons = np.loadtxt('coastlon.txt')
    coastlats = np.loadtxt('coastlat.txt')
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(coastlons, coastlats, color='black')
    ax.set_ylim(ymin=57.5, ymax=72)
    ax.set_xlim(xmin=-170, xmax=-135)
    ax.set_aspect(2.2)
    ax.grid()
    # Sidebar plots for each site
    ax1 = dict()
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1])
        ax1[site].plot(coastlons, coastlats, color='black')
        ax1[site].set_ylim(ymin=57.5, ymax=72)
        ax1[site].set_xlim(xmin=-170, xmax=-135)
        ax1[site].set_aspect(2.2)
        ax1[site].grid()
        ax1[site].set_title(site)
    for site, img in imgs.items():
        print(site)
        img[skymaps[site]['mask']] = np.nan
        im = img.copy()
        for m in skymaps[site]['extra_masks'].values():
            im[m] = np.nan
        im_handle = ax.pcolor(skymaps[site]['lon'], skymaps[site]['lat'], im)
        ax1[site].pcolor(skymaps[site]['lon'], skymaps[site]['lat'], img)
    '''
    print('PFISR')
    pfisr_handle = ax.scatter(pfisr['glon'], pfisr['glat'], c=pfisr['ne'], zorder=6, cmap='jet', vmin=0, vmax=4e11)
    u, v = scale_uv(pfisr['vlon'], pfisr['vlat'], pfisr['vel'][:, 0], pfisr['vel'][:, 1])
    qp = ax.quiver(pfisr['vlon'], pfisr['vlat'], u, v, zorder=7, scale=5000, width=0.005)
    '''
    print('Trajectories')
    lat1, lon1, latm1, lonm1, lata1, lona1 = load_traj('Traj_Left.txt')
    lat2, lon2, latm2, lonm2, lata2, lona2 = load_traj('Traj_Right.txt')
    ax.plot(lon1, lat1, color='red', label='GNEISS trajectory', zorder=7)
    ax.scatter(lonm1, latm1, color='red', s=15, zorder=7)
    ax.scatter(lona1, lata1, color='lavenderblush', label='Apogee', marker='x', zorder=7)
    ax.plot(lon2, lat2, color='red', zorder=7)
    ax.scatter(lonm2, latm2, color='red', s=15, zorder=7)
    ax.scatter(lona2, lata2, color='lavenderblush', marker='x', zorder=7)
    # Use the date/time from the arguments for the plot text
    from inspect import currentframe
    frame = currentframe()
    args = frame.f_back.f_locals.get('args', None)
    if args is not None:
        date_str = args.date
        time_str = args.time
        label_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
    else:
        label_str = ""
    txt = ax.text(0.99, 0.01, label_str,
                 transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.set_title("GNEISS Ground Sites (magnetic footpointing to 110 km)")
    ax.legend(loc='upper right')
    #ax.quiverkey(qp, 0.1, 0.9, 500., '500 m/s', transform=ax.transAxes)
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation='vertical')
    cbar.set_label('Green Channel Intensity')
    cax = fig.add_subplot(gs[:, 2])
    #cbar = fig.colorbar(pfisr_handle, cax=cax, orientation='vertical')
    cbar.set_label(r'Electron Density (m$^{-3}$)')
    plt.tight_layout()
    # Use the date/time from the arguments for the output filename
    if args is not None:
        output_path = f"../mapped/GNEISS_launch_science_fast_{date_str}_{time_str}.png"
    else:
        output_path = f"../mapped/GNEISS_launch_science_fast_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
    # plt.show()


def plot_pretty(skymaps, imgs, pfisr):
    """
    Pretty plotting mode: overlays images, PFISR, and trajectories on a Cartopy map.
    """
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0], projection=proj)
    ax.set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    ax.gridlines()
    mcm.maggridlines(ax, apex=apex, apex_height=110.)
    ax1 = dict()
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1], projection=proj)
        ax1[site].coastlines()
        ax1[site].gridlines()
        mcm.maggridlines(ax1[site], apex=apex, apex_height=110.)
        ax1[site].set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
        ax1[site].set_title(site)
    for site, img in imgs.items():
        print(site)
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
    print('PFISR')
    pfisr_handle = ax.scatter(pfisr['glon'], pfisr['glat'], c=pfisr['ne'], zorder=6, cmap='jet', transform=ccrs.Geodetic())
    u, v = scale_uv(pfisr['vlon'], pfisr['vlat'], pfisr['vel'][:, 0], pfisr['vel'][:, 1], vmin=0, vmax=4e11)
    qp = ax.quiver(pfisr['vlon'], pfisr['vlat'], u, v, zorder=7, scale=5000, width=0.005, transform=ccrs.PlateCarree())
    print('Trajectory')
    lat1, lon1, latm1, lonm1, lata1, lona1 = load_traj('Traj_Left.txt')
    lat2, lon2, latm2, lonm2, lata2, lona2 = load_traj('Traj_Right.txt')
    ax.plot(lon1, lat1, color='red', label='GNEISS trajectory', transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lonm1, latm1, color='red', s=15, transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lona1, lata1, color='lavenderblush', marker='x', label='Apogee', transform=ccrs.PlateCarree(), zorder=8)
    ax.plot(lon2, lat2, color='red', transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lonm2, latm2, color='red', s=15, transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lona2, lata2, color='lavenderblush', marker='x', transform=ccrs.PlateCarree(), zorder=8)
    ax.set_title("GNEISS Ground Sites (magnetic footpointing to 110 km)")
    txt = ax.text(0.99, 0.01, dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
                 transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.legend(loc='upper right')
    ax.quiverkey(qp, 0.1, 0.9, 500., '500 m/s', transform=ax.transAxes)
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation='vertical')
    cbar.set_label('Green Channel Intensity')
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(pfisr_handle, cax=cax, orientation='vertical')
    cbar.set_label(r'Electron Density (m$^{-3}$)')
    plt.tight_layout()
    # Use the date/time from the arguments for the output filename
    from inspect import currentframe
    frame = currentframe()
    args = frame.f_back.f_locals.get('args', None)
    if args is not None:
        date_str = args.date
        time_str = args.time
        output_path = f"../launch_science_pretty/GNEISS_launch_science_pretty_{date_str}_{time_str}.png"
    else:
        output_path = f"../launch_science_pretty/GNEISS_launch_science_pretty_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
    # plt.show()


def main():
    """
    Main entry point: parses arguments, loads data, and runs plotting.
    """
    ticall = time.time()
    # --- Parse command-line arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretty", action='store_true')
    ap.add_argument("--date", required=True, type=str, default=dt.datetime.now(dt.UTC).strftime("%Y%m%d"), help="Date for the ASI images (format: YYYYMMDD)")
    ap.add_argument("--time", required=True, type=str, default=dt.datetime.now(dt.UTC).strftime("%H%M%S"), help="Time for the ASI images (format: HHMMSS)")
    ap.add_argument("--sites", nargs='*', default=['ARV', 'PKR', 'VEE', 'BVR'], help="List of sites to process (default: all sites)")
    args = ap.parse_args()
    # --- Load the geographic mapping for the ASI image ---
    selected_sites = set([s.upper() for s in args.sites])
    skymaps = load_skymaps(selected_sites)
    # --- Calculate mask for overlaping images ---
    sites = list(skymaps.keys())
    for s0 in sites:
        red_sites = sites.copy()
        red_sites.remove(s0)
        skymaps[s0]['extra_masks'] = dict()
        for s1 in red_sites:
            m0, m1 = ci.calculate_masks(skymaps[s0]['site_lat'], skymaps[s0]['site_lon'], skymaps[s0]['azmt'], skymaps[s0]['elev'], skymaps[s1]['site_lat'], skymaps[s1]['site_lon'], skymaps[s1]['azmt'], skymaps[s1]['elev'])
            skymaps[s0]['extra_masks'][s1] = m0
    imgs = dict()
    date = args.date
    time_str = args.time

    #ARV
    if 'ARV' in selected_sites:
        try:
            tiff_path = "../raw_tiffs/ARV/ARV_558_20260210_102102.tiff"
            im = Image.open(tiff_path)
            im = np.asarray(im)
            if im.ndim == 3:
                im = im[:, :, 0]
            imgs['ARV'] = im.astype(np.float32)
        except Exception as e:
            print(f"Could not load ARV TIFF image: {e}")
    if 'VEE' in selected_sites:
        try:
            tiff_path = "../raw_tiffs/VEE/VEE_558_20260210_102203.tiff"
            im = Image.open(tiff_path)
            im = np.asarray(im)
            if im.ndim == 3:
                im = im[:, :, 0]
            imgs['VEE'] = im.astype(np.float32)
        except Exception as e:
            print(f"Could not load VEE TIFF image: {e}")
    if 'BVR' in selected_sites:
        try:
            tiff_path = "../raw_tiffs/BVR/BVR_558_20260210_102100.tiff"
            im = Image.open(tiff_path)
            im = np.asarray(im)
            if im.ndim == 3:
                im = im[:, :, 0]
            imgs['BVR'] = im.astype(np.float32)
        except Exception as e:
            print(f"Could not load BVR TIFF image: {e}")
    if 'PKR' in selected_sites:
        try:
            url_pkr = closest_amisr_png_url('PKR', date, time_str)
            imgs['PKR'] = retrieve_image(url_pkr)
        except Exception as e:
            print(f"Could not fetch PKR image: {e}")
    pfisr = retrieve_pfisr()
    if args.pretty:
        plot_pretty(skymaps, imgs, pfisr)
    else:
        plot_fast(skymaps, imgs, pfisr)
    tocall = time.time()
    print(f"Total run time: {tocall - ticall:.2f} s")

if __name__ == "__main__":
    main()
