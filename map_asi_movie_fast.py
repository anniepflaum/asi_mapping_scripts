#!/usr/bin/env python3
"""
map_asi_movie_fast.py

Create a movie from a sequence of ASI images mapped to geographic coordinates using only the green channel.
Uses precomputed interpolation objects for efficiency.
"""
import os
import numpy as np
import h5py
import imageio.v3 as iio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import ArtistAnimation
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import magcoordmap as mcm
import cv2
import pytesseract
from PIL import Image

def circle_mask(h, w, margin_px=0):
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = min(h, w) / 2.0 - margin_px
    return (xx - cx)**2 + (yy - cy)**2 <= r**2

def load_skymap_latlon(skymap_path: str, alt_km: int):
    key = f"{alt_km}km"
    with h5py.File(skymap_path, "r") as f:
        g = f["magnetic_footpointing"]
        gg = g[key]
        lat = np.array(gg["lat"], dtype=float)
        lon = np.array(gg["lon"], dtype=float)
    return lat, lon

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

def make_target_grid(lat: np.ndarray, lon: np.ndarray, nx: int, ny: int, padding_deg: float = 0.0):
    good = np.isfinite(lat) & np.isfinite(lon)
    lat_min, lat_max = np.nanmin(lat[good]), np.nanmax(lat[good])
    lon_min, lon_max = np.nanmin(lon[good]), np.nanmax(lon[good])
    lat_vec = np.linspace(lat_min - padding_deg, lat_max + padding_deg, ny)
    lon_vec = np.linspace(lon_min - padding_deg, lon_max + padding_deg, nx)
    Lon, Lat = np.meshgrid(lon_vec, lat_vec)
    return Lon, Lat, lon_vec, lat_vec

def remap_scattered_to_grid(values_2d, lat_2d, lon_2d, Lon, Lat, fill_nearest=True):
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

def extract_time_from_frame(frame):
    h, w = frame.shape[:2]
    left_frac   = 0.00
    right_frac  = 0.60
    top_frac    = 0.93
    bottom_frac = 1.00
    x0 = int(left_frac * w)
    x1 = int(right_frac * w)
    y0 = int(top_frac * h)
    y1 = int(bottom_frac * h)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    th = cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:UTC "
    text = pytesseract.image_to_string(Image.fromarray(th), config=config)
    return text.strip()

def extract_date_from_frame(frame):
    h, w = frame.shape[:2]
    left_frac   = 0.70
    right_frac  = 1.00
    top_frac    = 0.00
    bottom_frac = 0.10
    x0 = int(left_frac * w)
    x1 = int(right_frac * w)
    y0 = int(top_frac * h)
    y1 = int(bottom_frac * h)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    th = cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/- "
    text = pytesseract.image_to_string(Image.fromarray(th), config=config)
    return text.strip()

def main():

    # --- Prompt user for date and set frame directory ---
    date_str = input("Enter date for frames (YYYYMMDD): ").strip()
    if not (len(date_str) == 8 and date_str.isdigit()):
        raise SystemExit("Invalid date format. Please use YYYYMMDD.")
    FRAMES_DIR = f"/Users/anniepflaum/ASI_mapping/PKR_frames/{date_str}"
    SKYMAP = "/Users/anniepflaum/ASI_mapping/skymap.mat"
    ALT_KM = 110
    OUT_MP4 = f"/Users/anniepflaum/ASI_mapping/mapped/PKR_timelapse_artistanim_fast_{date_str}.mp4"
    FPS = 12
    NX, NY = 512, 512
    PADDING_DEG = 0.0
    LON_CONVENTION = "native"

    # --- Precompute static mapping grid ---
    lat, lon = load_skymap_latlon(SKYMAP, ALT_KM)
    lon = normalize_lon(lon, LON_CONVENTION)
    img_path = sorted([
        os.path.join(FRAMES_DIR, f)
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])[0]
    img = iio.imread(img_path)
    img = np.asarray(img).astype(np.float32)
    H, W, _ = img.shape
    fov = circle_mask(H, W, margin_px=50) & np.isfinite(lat) & np.isfinite(lon)
    lat_use = np.where(fov, lat, np.nan)
    lon_use = np.where(fov, lon, np.nan)
    Lon, Lat, lon_vec, lat_vec = make_target_grid(lat_use, lon_use, NX, NY, PADDING_DEG)
    fill_nearest = False
    footprint = remap_scattered_to_grid(fov.astype(np.float32), lat_use, lon_use, Lon, Lat, fill_nearest=False)
    inside = np.isfinite(footprint) & (footprint > 0.5)
    Lon, Lat = Lon.astype(np.float32), Lat.astype(np.float32)
    extent = (float(np.nanmin(Lon)), float(np.nanmax(Lon)), float(np.nanmin(Lat)), float(np.nanmax(Lat)))

    # --- Precompute nearest-neighbor mapping from output grid to input image ---
    # For each (y, x) in output grid, find nearest (i, j) in input image where mask is valid
    from scipy.spatial import cKDTree
    img_indices = np.array(np.nonzero(fov)).T  # (N, 2) array of (i, j) valid input pixels
    img_lats = lat[img_indices[:, 0], img_indices[:, 1]]
    img_lons = lon[img_indices[:, 0], img_indices[:, 1]]
    img_coords = np.column_stack([img_lons, img_lats])
    tree = cKDTree(img_coords)
    out_coords = np.column_stack([Lon.ravel(), Lat.ravel()])
    dists, nn_idx = tree.query(out_coords)
    nn_i = img_indices[nn_idx, 0].reshape(Lon.shape)
    nn_j = img_indices[nn_idx, 1].reshape(Lon.shape)

    # --- Set up Cartopy plot (static, for all frames) ---
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=proj)
    ax.set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    ax.set_title("ASI mapped to geographic lat/lon (green intensity)")
    mgl = mcm.maggridlines(ax)

        # --- Rocket Trajectory Overlay ---
    def load_traj(filename):
        data = np.loadtxt(filename, skiprows=1, usecols=(1,2))
        lats, lons = data[:,0], data[:,1]
        return lats, lons

    try:
        lat1, lon1 = load_traj('/Users/anniepflaum/ASI_mapping/Traj_LeftGneissDec25.txt')
        lat2, lon2 = load_traj('/Users/anniepflaum/ASI_mapping/Traj_RightGneissDec25.txt')
        ax.scatter(lon1, lat1, s=0.5, color='red', label='GNEISS trajectory', transform=ccrs.PlateCarree(), zorder=3)
        ax.scatter(lon2, lat2, s=0.5, color='red', transform=ccrs.PlateCarree(), zorder=3)
        ax.legend(loc='upper right')
    except Exception as e:
        print(f"Could not plot rocket trajectories: {e}")

    # --- List all frames ---
    frame_files = sorted([
        os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])
    print(f"Found {len(frame_files)} frames.")

    # --- Extract date from first frame ---
    img_uint8 = img.astype(np.uint8)
    date_str = extract_date_from_frame(img_uint8)

    # --- Prepare artists for animation using precomputed nearest-neighbor mapping ---
    artists = []
    print("Processing frames for animation (precomputed nearest-neighbor)...")
    for frame_path in tqdm(frame_files, desc="Mapping frames", unit="frame"):
        img = iio.imread(frame_path)
        img = np.asarray(img).astype(np.float32)
        green = img[:, :, 1]
        mapped = green[nn_i, nn_j]
        mapped[~inside] = np.nan
        norm_green = np.clip(mapped / 255.0, 0, 1)
        mask = np.isfinite(mapped)
        # Extract time from current frame
        img_uint8 = img.astype(np.uint8)
        time_str = extract_time_from_frame(img_uint8)
        # Plot image and time/date annotation
        im = ax.imshow(norm_green, origin="lower", extent=extent, transform=ccrs.PlateCarree(), zorder=1, alpha=mask.astype(float), cmap='viridis', animated=True)
        txt = ax.text(0.99, 0.01, f"{date_str} {time_str}", transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'), animated=True)
        artists.append([im, txt])

    cbar = plt.colorbar(artists[0][0], ax=ax, orientation='vertical', pad=0.02, fraction=0.05)
    cbar.set_label('Green channel intensity (normalized)')
    plt.tight_layout()

    # --- Create animation ---
    anim = ArtistAnimation(fig, artists, interval=1000//FPS, blit=True)
    anim.save(OUT_MP4, writer='ffmpeg', fps=FPS)
    print(f"Saved timelapse movie to {OUT_MP4}")

if __name__ == "__main__":
    main()
