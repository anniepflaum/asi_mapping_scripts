from io import BytesIO
import datetime as dt
import re
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image
from apexpy import Apex
from core.constants import DEFAULT_GREEN_ALT_KM
from core.fetch_url import closest_amisr_png_url

try:
    from resolvedvelocities.ResolveVectorsLat import ResolveVectorsLat
except ImportError:
    ResolveVectorsLat = None


CORE_DIR = Path(__file__).resolve().parent
PKR_LOCAL_ROOT = Path("/Users/anniepflaum/lab317/asi_mapping/images")
PKR_FILENAME_RE = re.compile(r"^PFRR_(\d{8})_(\d{6})_(0558|0630)\.png$", re.IGNORECASE)


def retrieve_image(url, verbose=True):
    """
    Download a single-channel image from a URL and return it as float32.
    """
    if verbose:
        print(f"PKR: {url} ...")
    resp = requests.get(url, verify=False)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_image_file(path):
    img = Image.open(path)
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def pkr_png_datetime(path):
    match = PKR_FILENAME_RE.match(Path(path).name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def pkr_png_datetime_from_source(source):
    frame_dt = pkr_png_datetime(Path(str(source)))
    if frame_dt is not None:
        return frame_dt
    match = re.search(r"(\d{8})_(\d{6})", str(source))
    if not match:
        raise ValueError(f"Could not parse PKR frame timestamp from {source}")
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def closest_local_pkr_png(date, time, color="green", tolerance_seconds=15):
    color = str(color).lower().strip()
    suffix = "0558" if color == "green" else "0630"
    target = dt.datetime.strptime(re.sub(r"\D", "", date) + re.sub(r"\D", "", time), "%Y%m%d%H%M%S")
    base_dir = PKR_LOCAL_ROOT / color / "PKR"
    search_dirs = [
        base_dir / target.strftime("%Y%m%d"),
        base_dir / target.strftime("%Y") / target.strftime("%Y%m%d") / target.strftime("%H"),
        base_dir / target.strftime("%Y") / target.strftime("%Y%m%d"),
        base_dir,
    ]
    candidates = []
    seen = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob(f"PFRR_*_*_{suffix}.png"):
            if path in seen:
                continue
            seen.add(path)
            frame_dt = pkr_png_datetime(path)
            if frame_dt is None:
                continue
            delta = abs((frame_dt - target).total_seconds())
            if delta <= tolerance_seconds:
                candidates.append((delta, frame_dt, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[2].name))
    return candidates[0][2], candidates[0][1]


def load_pkr_image(date, time, color="green", verbose=True):
    """
    Load the closest PKR PNG, preferring local images files and falling back
    to the AMISR archive URL if no local file is available.
    """
    local = closest_local_pkr_png(date, time, color=color)
    if local is not None:
        path, frame_dt = local
        if verbose:
            print(f"PKR: local {path} ...")
        return load_image_file(path), str(path), frame_dt

    url = closest_amisr_png_url("PKR", date, time, color=color)
    return retrieve_image(url, verbose=verbose), url, pkr_png_datetime_from_source(url)


def retrieve_pfisr(apex=None, map_alt_km=DEFAULT_GREEN_ALT_KM):
    """
    Download and process PFISR data.
    Returns electron density, velocity, and location arrays for plotting.
    """
    if ResolveVectorsLat is None:
        raise ImportError("resolvedvelocities module is not installed")
    if apex is None:
        apex = Apex()
    url = "https://amisr.com/realtime/plots/fitted/single/dtc3/current.h5"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open("pfisr_latest.h5", "wb") as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)
    with h5py.File("pfisr_latest.h5", "r") as h5:
        ne = h5["FittedParams/Ne"][:]
        dne = h5["FittedParams/dNe"][:]
        glat = h5["Geomag/Latitude"][:]
        glon = h5["Geomag/Longitude"][:]
        galt = h5["Geomag/Altitude"][:]
    ne[dne > ne] = np.nan
    vvels = ResolveVectorsLat(str(CORE_DIR / "vvels_config.ini"))
    vvels.transform()
    vvels.bin_data_mlat()
    vvels.compute_vector_velocity()
    vvels.compute_electric_field()
    vvels.compute_geodetic_output()
    glat, glon, _ = apex.map_to_height(glat, glon, galt / 1000.0, map_alt_km)
    aidx = np.argmin(np.abs(vvels.outalt - map_alt_km))
    vv = vvels.Velocity_gd[0, aidx, :, :]
    vm = vvels.Vgd_mag[0, aidx, :]
    ve = vvels.Vgd_mag_err[0, aidx, :]
    vlat = vvels.bin_glat[aidx, :]
    vlon = vvels.bin_glon[aidx, :]
    vv[ve > vm, :] = [np.nan, np.nan, np.nan]
    vm[ve > vm] = np.nan
    return {"ne": ne, "glat": glat, "glon": glon, "vel": vv, "mag": vm, "vlat": vlat, "vlon": vlon}
