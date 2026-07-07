# generate_skymap.py

import numpy as np
from astropy.io import fits
from scipy.io import readsav
import pymap3d as pm
import h5py

from core.constants import DEFAULT_GREEN_ALT_KM
from core.paths import starmap_path


#def azel2geo(centlat, centlon, az, el, mapalt_km=DEFAULT_GREEN_ALT_KM):
#    # Radius of earth at poker flat latitude
#    # In the future, replace this with an ellipsoid formula
#    # to find earth radius at any latitude
#    r_e = 6360562
#    
#    # Preprocessing az, el: right now they have zeros outside the usable FOV, we replace them with NaNs.
#    # Everywhere with az=el=0 is replaced with a NaN. This would only mark one valid point as invalid, and
#    # since it has el=0, it isn't usable anyway.
#    badrange = np.where((az==0) & (el==0))
#    az[badrange] = np.nan
#    el[badrange] = np.nan
#
#    # Distance from the ground site to the pixel's light source
#    # This is just trigonometry!
#    r = np.sqrt( ( r_e*np.sin(np.pi*el/180) )**2 + 2*(1000*mapalt_km)*r_e + (1000*mapalt_km)**2 ) - r_e*np.sin(np.pi*el/180)
#
#    # Converting. Note that fullalt gives a sanity check on how good your projection was
#    lat,lon,fullalt = pm.aer2geodetic(az,el,r,centlat,centlon,1000*mapalt_km)
#    lon = normalize_lon(lon, convention='0_360')
#    lat[~np.isfinite(lat)] = 0.
#    lon[~np.isfinite(lon)] = 0.
#
#    return lat, lon

def azel2geo(site_lat, site_lon, az, el, alt=DEFAULT_GREEN_ALT_KM):

    # lat/lon array
    x, y, z = pm.geodetic2ecef(site_lat, site_lon, 0.)
    e, n, u = pm.aer2enu(az, el, 1.)
    vx, vy, vz = pm.enu2uvw(e, n, u, site_lat, site_lon)

    earth = pm.Ellipsoid.from_name('wgs84')
    a2 = (earth.semimajor_axis + alt*1000.)**2
    b2 = (earth.semimajor_axis + alt*1000.)**2
    c2 = (earth.semiminor_axis + alt*1000.)**2

    A = vx**2/a2 + vy**2/b2 + vz**2/c2
    B = x*vx/a2 + y*vy/b2 + z*vz/c2
    C = x**2/a2 + y**2/b2 + z**2/c2 -1

    alpha = (np.sqrt(B**2-A*C)-B)/A

    lat, lon, alt = pm.ecef2geodetic(x + alpha*vx, y + alpha*vy, z + alpha*vz)

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


# Poker Flat (PKR)
def load_PKR(color="green"):

    site_lon, site_lat = [-147.43,   65.1192]
    azdat = readsav(starmap_path(color, "PKR", "PKR_DASC_5577_20260210_RAW_FULL_Az.sav"), python_dict=True)
    eldat = readsav(starmap_path(color, "PKR", "PKR_DASC_5577_20260210_RAW_FULL_El.sav"), python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask




# Venetie (VEE)
def load_VEE(color="green", mission="GNEISS"):

    mission_key = str(mission).upper()
    site_lon, site_lat = [-146.407,  67.013]
    if mission_key == "GIRAFF":
        with h5py.File(starmap_path(color, "VEE", "GIRAFF", "sok_pixelcoords.h5"), 'r') as fd:
            latmap = fd['Latitude'][()].copy()
            lonmap = fd['Longitude'][()].copy()
            azmap = np.rad2deg(fd['Azimuth'][()].copy())
            elmap = np.rad2deg(fd['Elevation'][()].copy())
            mask = fd['Mask'][()].copy().astype(bool)
        return site_lat, site_lon, azmap, elmap, mask, latmap, lonmap

    if str(color).lower() == "red":
        az_path = starmap_path(color, "VEE", "VEE_GASI_630_20260210_080000_asistarcalibration_full_Az.sav")
        el_path = starmap_path(color, "VEE", "VEE_GASI_630_20260210_080000_asistarcalibration_full_El.sav")
    else:
        az_path = starmap_path(color, "VEE", "GNEISS", "VEE_GASI_20260210_050100_rot5_full_Az.sav")
        el_path = starmap_path(color, "VEE", "GNEISS", "VEE_GASI_20260210_050100_rot5_full_El.sav")
    azdat = readsav(az_path, python_dict=True)
    eldat = readsav(el_path, python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask



# Beaver
def load_BVR(color="green"):
    site_lon, site_lat = [-147.4,    66.36]

    if str(color).lower() == "red":
        az_path = starmap_path(color, "BVR", "BVR_GASI_630_20260210_051800_asistarcalibration_full_Az.sav")
        el_path = starmap_path(color, "BVR", "BVR_GASI_630_20260210_051800_asistarcalibration_full_El.sav")
    else:
        az_path = starmap_path(color, "BVR", "BVR_20260210_090000_750_rot5_Az.sav")
        el_path = starmap_path(color, "BVR", "BVR_20260210_090000_750_rot5_El.sav")
    azdat = readsav(az_path, python_dict=True)
    eldat = readsav(el_path, python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap < 15.

    return site_lat, site_lon, azmap, elmap, mask


# Arctic Village
def load_ARV(color="green"):
    site_lon, site_lat = [-145.533,  68.127]

    if color == "red":
        azmap = fits.getdata(starmap_path(color, "ARV", "ARV_GASI_630_20260209_Az.FIT")).copy()
        elmap = fits.getdata(starmap_path(color, "ARV", "ARV_GASI_630_20260209_El.FIT")).copy()
    else:
        azdat = readsav(starmap_path(color, "ARV", "ARV_GASI_20260209_063700_rot5_full_Az.sav"), python_dict=True)
        eldat = readsav(starmap_path(color, "ARV", "ARV_GASI_20260209_063700_rot5_full_El.sav"), python_dict=True)
        azmap = azdat[list(azdat.keys())[0]].copy()
        elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap < 15.

    return site_lat, site_lon, azmap, elmap, mask
