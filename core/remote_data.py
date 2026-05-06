from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image
from apexpy import Apex
from core.constants import DEFAULT_GREEN_ALT_KM

try:
    from resolvedvelocities.ResolveVectorsLat import ResolveVectorsLat
except ImportError:
    ResolveVectorsLat = None


CORE_DIR = Path(__file__).resolve().parent


def retrieve_image(url):
    """
    Download a single-channel image from a URL and return it as float32.
    """
    print(f"PKR: {url} ...")
    resp = requests.get(url, verify=False)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


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
