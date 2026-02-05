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
import h5py
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import magcoordmap as mcm  # Leslie Lamarche's custom module for magnetic grid lines
from PIL import Image
from io import BytesIO

import check_intersect as ci


# Create a circular mask for the ASI field of view (optionally with a margin)
def circle_mask(h, w, margin_px=0):
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = min(h, w) / 2.0 - margin_px
    return (xx - cx)**2 + (yy - cy)**2 <= r**2


# Load the latitude and longitude mapping arrays from the skymap.mat file for a given altitude
def load_skymap_latlon(skymap_path: str, alt_km: int) -> tuple[np.ndarray, np.ndarray]:
    key = f"{alt_km}km"
    with h5py.File(skymap_path, "r") as f:
        g = f["magnetic_footpointing"]
        gg = g[key]
        lat = np.array(gg["lat"], dtype=float)
        lon = np.array(gg["lon"], dtype=float)
    return lat, lon


# Normalize longitude values to a specified convention
def normalize_lon(lon: np.ndarray, convention: str) -> np.ndarray:
    lon = lon.copy()
    if convention == "0_360":
        return np.mod(lon, 360.0)
    if convention == "-180_180":
        return (np.mod(lon + 180.0, 360.0) - 180.0)
    finite = np.isfinite(lon)
    if not np.any(finite):
        return lon
    lon0 = lon[finite]
    span0 = np.nanmax(lon0) - np.nanmin(lon0)
    lon_wrapped = (np.mod(lon + 180.0, 360.0) - 180.0)
    lon1 = lon_wrapped[finite]
    span1 = np.nanmax(lon1) - np.nanmin(lon1)
    if span1 < span0:
        return lon_wrapped
    return lon


# Create a regular output grid in lat/lon for remapping the ASI data
def make_target_grid(lat: np.ndarray, lon: np.ndarray, nx: int, ny: int, padding_deg: float = 0.0):
    good = np.isfinite(lat) & np.isfinite(lon)
    lat_min, lat_max = np.nanmin(lat[good]), np.nanmax(lat[good])
    lon_min, lon_max = np.nanmin(lon[good]), np.nanmax(lon[good])
    lat_vec = np.linspace(lat_min - padding_deg, lat_max + padding_deg, ny)
    lon_vec = np.linspace(lon_min - padding_deg, lon_max + padding_deg, nx)
    Lon, Lat = np.meshgrid(lon_vec, lat_vec)
    return Lon, Lat, lon_vec, lat_vec


# Interpolate scattered ASI data (in image coordinates) onto the regular lat/lon grid
def remap_scattered_to_grid(values_2d: np.ndarray,
                            lat_2d: np.ndarray,
                            lon_2d: np.ndarray,
                            Lon: np.ndarray,
                            Lat: np.ndarray,
                            fill_nearest: bool = True) -> np.ndarray:
    pts = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
    vals = values_2d.ravel()
    good = np.isfinite(pts).all(axis=1) & np.isfinite(vals)
    pts = pts[good]
    vals = vals[good]
    lin = LinearNDInterpolator(pts, vals, fill_value=np.nan)
    out = lin(Lon, Lat)
    if fill_nearest:
        nanmask = np.isnan(out)
        if np.any(nanmask):
            nn = NearestNDInterpolator(pts, vals)
            out[nanmask] = nn(Lon[nanmask], Lat[nanmask])
    return out.astype(np.float32)

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
#    lat, lon = load_skymap_latlon(args.skymap, args.alt_km)
#    lon = normalize_lon(lon, args.lon_convention)
    filename = '/Users/e30737/Desktop/Research/SoP_DI/ASIspecinvert_test/starcal/PKR_pixel_coords.h5'
    site_lon = -147.43
    site_lat = 65.1192
    with h5py.File(filename, 'r') as h5:
        lat = h5['Latitude'][:]
        lon = h5['Longitude'][:]
        azmt = h5['Azimuth'][:]
        elev = h5['Elevation'][:]
        mask = h5['Mask'][:]

    filename = '/Users/e30737/Desktop/Research/SoP_DI/ASIspecinvert_test/starcal/VEE_pixel_coords.h5'
    site_lonv = -146.407
    site_latv = 67.013
    with h5py.File(filename, 'r') as h5:
        latv = h5['Latitude'][:]
        lonv = h5['Longitude'][:]
        azmtv = h5['Azimuth'][:]
        elevv = h5['Elevation'][:]
        maskv = h5['Mask'][:]

    m, mv = ci.calculate_masks(site_lat, site_lon, np.rad2deg(azmt), np.rad2deg(elev), site_latv, site_lonv, np.rad2deg(azmtv), np.rad2deg(elevv))

    mask = np.logical_or(mask, m)
    maskv = np.logical_or(maskv, mv)

    # --- Download the latest PKR green channel image (already single-channel) ---
    url = "https://optics.gi.alaska.edu/realtime/latest/pkr_latest_green.jpg"
    print(f"Downloading {url} ...")
    resp = requests.get(url, verify=False)  # verify=False disables SSL cert check (safe for public data)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        # If image is RGB, take only one channel (shouldn't be needed, but for safety)
        img = img[:, :, 0]
    img = img.astype(np.float32)
    H, W = img.shape

    # --- Download the latest VEE green channel image (already single-channel) ---
    url = "https://optics.gi.alaska.edu/amisr_archive/VEE/GASI_5577/png/20260203/VEE_558_20260203_092814.png"
    #url = 'https://optics.gi.alaska.edu/realtime/latest/vee_latest.jpg'
    print(f"Downloading {url} ...")
    resp = requests.get(url, verify=False)  # verify=False disables SSL cert check (safe for public data)
    resp.raise_for_status()
    imgv = Image.open(BytesIO(resp.content))
    imgv = np.asarray(imgv)
    if imgv.ndim == 3:
        # If image is RGB, take only one channel (shouldn't be needed, but for safety)
        imgv = imgv[:, :, 0]
    imgv = imgv.astype(np.float32)
    Hv, Wv = imgv.shape


#    # --- Mask out pixels outside the ASI field of view and invalid mapping points ---
#    fov = circle_mask(H, W, margin_px=50) & np.isfinite(lat) & np.isfinite(lon)
#    lat_use = np.where(fov, lat, np.nan)
#    lon_use = np.where(fov, lon, np.nan)
#
#    # --- Create the output lat/lon grid ---
#    Lon, Lat, lon_vec, lat_vec = make_target_grid(lat_use, lon_use, args.nx, args.ny, args.padding_deg)
#    fill_nearest = False
#
#    # --- Determine which output grid points are inside the valid ASI footprint ---
#    footprint = remap_scattered_to_grid(fov.astype(np.float32), lat_use, lon_use, Lon, Lat, fill_nearest=False)
#    inside = np.isfinite(footprint) & (footprint > 0.5)
#    footprint = inside.astype(np.float32)
#    Lon, Lat = Lon.astype(np.float32), Lat.astype(np.float32)


    # --- Set up the Cartopy map for plotting ---
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    #extent = (float(np.nanmin(Lon)), float(np.nanmax(Lon)), float(np.nanmin(Lat)), float(np.nanmax(Lat)))
    ax.set_title("PKR ASI latest green channel mapped to geographic lat/lon")
    mgl = mcm.maggridlines(ax)


    # --- Overlay rocket trajectories from text files (if available) ---
    def load_traj(filename):
        # Load latitude and longitude columns from a trajectory text file
        data = np.loadtxt(filename, skiprows=1, usecols=(1,2))
        lats, lons = data[:,0], data[:,1]
        return lats, lons

    try:
        #lat1, lon1 = load_traj('/Users/anniepflaum/ASI_mapping/Traj_LeftGneissDec25.txt')
        #lat2, lon2 = load_traj('/Users/anniepflaum/ASI_mapping/Traj_RightGneissDec25.txt')
        lat1, lon1 = load_traj('../Traj_Left.txt')
        lat2, lon2 = load_traj('../Traj_Right.txt')
        ax.scatter(lon1, lat1, s=0.5, color='red', label='GNEISS trajectory', transform=ccrs.PlateCarree(), zorder=3)
        ax.scatter(lon2, lat2, s=0.5, color='red', transform=ccrs.PlateCarree(), zorder=3)
        ax.legend(loc='upper right')
    except Exception as e:
        print(f"Could not plot rocket trajectories: {e}")


    ## --- Remap the green channel image to the geographic grid ---
    #mapped = remap_scattered_to_grid(img, lat_use, lon_use, Lon, Lat, fill_nearest=fill_nearest)
    #mapped[~inside] = np.nan  # Mask out-of-footprint pixels
    #norm_green = np.clip(mapped / 255.0, 0, 1)  # Normalize to [0, 1] for display
    #mask = np.isfinite(mapped)
    #im_handle = ax.imshow(norm_green, origin="lower", extent=extent, transform=ccrs.PlateCarree(), zorder=1, alpha=mask.astype(float), cmap='viridis')
    img[mask] = np.nan
    im_handle = ax.pcolormesh(lon, lat, img, transform=ccrs.PlateCarree())

    imgv[maskv] = np.nan
    im_handle = ax.pcolormesh(lonv, latv, imgv, transform=ccrs.PlateCarree())

    # --- Add colorbar and save the figure ---
    cbar = plt.colorbar(im_handle, ax=ax, orientation='vertical', pad=0.02, fraction=0.05)
    cbar.set_label('Green channel intensity (normalized)')
    plt.tight_layout()
    output_path = "PKR_realtime.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")

if __name__ == "__main__":
    main()
