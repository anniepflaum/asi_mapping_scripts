#!/usr/bin/env python3
"""
map_asi_PKR_realtime.py

This script downloads the latest all-sky imager (ASI) green channel image from the Poker Flat Research Range (PKR) in Alaska,
maps it to geographic latitude/longitude coordinates using a provided skymap, and overlays rocket trajectories.
The output is a PNG image showing the mapped green channel intensity and rocket paths.

Usage:
    python map_asi_PKR_realtime.py --skymap skymap.mat --alt-km 110 --nx 512 --ny 512

Arguments:
    --skymap        Path to the .mat file containing the geographic mapping for the ASI
    --alt-km        Altitude (in km) for the mapping grid
    --nx, --ny      Output grid size in longitude and latitude
    --padding-deg   Optional: extra padding (in degrees) around the mapped region
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import magcoordmap as mcm  # Leslie Lamarche's custom module for magnetic grid lines
from PIL import Image
from io import BytesIO
import datetime as dt

import check_intersect as ci
import generate_skymap as skymap



## Load the latitude and longitude mapping arrays from the skymap.mat file for a given altitude
def load_skymaps():
    # Load skymaps for all cameras
    skymaps = dict()
    lat, lon, az, el, mask = skymap.load_PKR()
    skymaps['PKR'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    lat, lon, az, el, mask = skymap.load_VEE()
    skymaps['VEE'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    lat, lon, az, el, mask = skymap.load_BVR()
    skymaps['BVR'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    for sm in skymaps.values():
        lat, lon = skymap.azel2geo(sm['site_lat'], sm['site_lon'], sm['azmt'], sm['elev'], alt=110.)
        sm['lat'] = lat
        sm['lon'] = lon

    return skymaps


def retrieve_image(url):

    print(f"Downloading {url} ...")
    resp = requests.get(url, verify=False)  # verify=False disables SSL cert check (safe for public data)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        # If image is RGB, take only one channel (shouldn't be needed, but for safety)
        img = img[:, :, 0]
    return img.astype(np.float32)


def main():
    # --- Parse command-line arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--skymap", default="/Users/anniepflaum/ASI_mapping/skymap.mat")
    ap.add_argument("--alt-km", type=int, default=110)
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=512)
    ap.add_argument("--padding-deg", type=float, default=0.0)
    ap.add_argument("--lon-convention", choices=["native", "-180_180", "0_360"], default="native")
    args = ap.parse_args()


#    # --- Load the geographic mapping for the ASI image ---
    skymaps = load_skymaps()



    # --- Calculate mask for overlaping images
    # need to generalize
    #mp, mv = ci.calculate_masks(skymaps['PKR']['site_lat'], skymaps['PKR']['site_lon'], skymaps['PKR']['azmt'], skymaps['PKR']['elev'], skymaps['VEE']['site_lat'], skymaps['VEE']['site_lon'], skymaps['VEE']['azmt'], skymaps['VEE']['elev'])
    #skymaps['PKR']['mask'] = np.logical_or(skymaps['PKR']['mask'], mp)
    #skymaps['VEE']['mask'] = np.logical_or(skymaps['VEE']['mask'], mv)

    sites = list(skymaps.keys())
    for s0 in sites:
        red_sites = sites.copy()
        red_sites.remove(s0)
        print(s0)
        print(red_sites)
        skymaps[s0]['extra_masks'] = dict()
        for s1 in red_sites:
            m0, m1 = ci.calculate_masks(skymaps[s0]['site_lat'], skymaps[s0]['site_lon'], skymaps[s0]['azmt'], skymaps[s0]['elev'], skymaps[s1]['site_lat'], skymaps[s1]['site_lon'], skymaps[s1]['azmt'], skymaps[s1]['elev'])

            skymaps[s0]['extra_masks'][s1] = m0


    imgs = dict()

    # --- Download the latest PKR green channel image (already single-channel) ---
    url = "https://optics.gi.alaska.edu/realtime/latest/pkr_latest_green.jpg"
    imgs['PKR'] = retrieve_image(url)

    # --- Download the latest VEE green channel image (already single-channel) ---
    url = 'https://optics.gi.alaska.edu/realtime/latest/vee_558_latest.jpg'
    im = retrieve_image(url)
    imgs['VEE'] = np.flipud(im)


    # --- Download the latest BVR green channel image (already single-channel) ---
    url = 'https://optics.gi.alaska.edu/realtime/latest/bvr_558_latest.jpg'
    #imgs['BVR'] = retrieve_image(url)
    im = retrieve_image(url)
    imgs['BVR'] = np.flipud(im)





    # --- Set up the Cartopy map for plotting ---
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4,2, width_ratios=[4,1])
    #ax = plt.axes(projection=proj)
    ax = fig.add_subplot(gs[:,0], projection=proj)
    ax.set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    #extent = (float(np.nanmin(Lon)), float(np.nanmax(Lon)), float(np.nanmin(Lat)), float(np.nanmax(Lat)))
    ax.set_title("PKR ASI latest green channel mapped to geographic lat/lon")
    mgl = mcm.maggridlines(ax)

    # --- Setup sidebar plots ---
    ax1 = dict()
    for i, k in enumerate(imgs.keys()):
        ax1[k] = fig.add_subplot(gs[i,1], projection=proj)
        ax1[k].coastlines()
        ax1[k].gridlines()
        ax1[k].set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())


    # --- Overlay rocket trajectories from text files (if available) ---
    def load_traj(filename):
        # Load latitude and longitude columns from a trajectory text file
        times, lats, lons, alts = np.loadtxt(filename, skiprows=1, unpack=True)
        # Rocket trajectories every minute
        idx = np.argwhere(times % 60 == 0)
        timem = times[idx].squeeze()
        latsm = lats[idx].squeeze()     # lat every minute
        lonsm = lons[idx].squeeze()     # lon every minute
        aidx = np.argmax(alts)
        lata = lats[aidx]       # lat of appogee
        lona = lons[aidx]       # lon of appogee

        return lats, lons, latsm, lonsm, lata, lona

    try:
        lat1, lon1, latm1, lonm1, lata1, lona1 = load_traj('Traj_Left.txt')
        lat2, lon2, latm2, lonm2, lata2, lona2 = load_traj('Traj_Right.txt')
        ax.plot(lon1, lat1, color='red', label='GNEISS trajectory', transform=ccrs.PlateCarree(), zorder=7)
        ax.scatter(lonm1, latm1, color='red', s=15, transform=ccrs.PlateCarree(), zorder=7)
        ax.scatter(lona1, lata1, color='magenta', marker='x', transform=ccrs.PlateCarree(), zorder=8)
        ax.plot(lon2, lat2, color='red', transform=ccrs.PlateCarree(), zorder=7)
        ax.scatter(lonm2, latm2, color='red', s=15, transform=ccrs.PlateCarree(), zorder=7)
        ax.scatter(lona2, lata2, color='magenta', marker='x', transform=ccrs.PlateCarree(), zorder=8)
        ax.legend(loc='upper right')
    except Exception as e:
        print(f"Could not plot rocket trajectories: {e}")


    for site, img in imgs.items():
        print(site)

        #if site == 'BVR':
        #    print(f'skipping {site}')
        #    continue

        # apply mask
        img[skymaps[site]['mask']] = np.nan

        im = img.copy()
        lat = skymaps[site]['lat'].copy()
        lon = skymaps[site]['lon'].copy()
        for m in skymaps[site]['extra_masks'].values():
            im[m] = np.nan

        ############################
        # THIS WORKS BUT SLOW
        img_flat = img[~skymaps[site]['mask']].flatten()
        lon_flat = skymaps[site]['lon'][~skymaps[site]['mask']].flatten()
        lat_flat = skymaps[site]['lat'][~skymaps[site]['mask']].flatten()

        im_handle = ax1[site].tripcolor(lon_flat, lat_flat, img_flat, vmin=0, vmax=60, zorder=3, transform=ccrs.PlateCarree())

        imf = im[np.isfinite(im)].flatten()
        latf = lat[np.isfinite(im)].flatten()
        lonf = lon[np.isfinite(im)].flatten()

        #ax1[site].tripcolor(lon_flat, lat_flat, img_flat, transform=ccrs.PlateCarree())
        ax.tripcolor(lonf, latf, imf, vmin=0, vmax=60, transform=ccrs.PlateCarree())
        ax1[site].set_title(site)
        #############################


        #im_handle = ax.pcolor(skymaps[site]['lon'], skymaps[site]['lat'], img, transform=ccrs.PlateCarree())

        #ax1[site].pcolor(skymaps[site]['lon'], skymaps[site]['lat'], img, transform=ccrs.PlateCarree())
        #ax1[site].set_title(site)


    # --- Download the PFISR data ---
    url = "https://amisr.com/realtime/plots/fitted/single/dtc3/current.h5"
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)  # verify=False disables SSL cert check (safe for public data)
    resp.raise_for_status()
    with open('pfisr_latest.h5', 'wb') as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)
   
    # --- Plot the PFISR data ---
    pfisr_file = 'pfisr_latest.h5'
    with h5py.File(pfisr_file, 'r') as h5:
        ne = h5['FittedParams/Ne'][:]
        glat = h5['Geomag/Latitude'][:]
        glon = h5['Geomag/Longitude'][:]

    ax.scatter(glon, glat, c=ne, zorder=6, transform=ccrs.Geodetic())


    txt = ax.text(0.99, 0.01, dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # --- Add colorbar and save the figure ---
    cbar = plt.colorbar(im_handle, ax=ax, orientation='vertical', pad=0.02, fraction=0.05)
    cbar.set_label('Green channel intensity (normalized)')
    plt.tight_layout()
    output_path = "PKR_realtime.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
