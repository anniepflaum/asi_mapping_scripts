# generate_skymap.py

import numpy as np
from astropy.io import fits
from scipy.io import readsav
import pymap3d as pm
import h5py
from skimage.transform import resize


#def azel2geo(centlat, centlon, az, el, mapalt_km=110.):
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

def azel2geo(site_lat, site_lon, az, el, alt=110.):

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
def load_PKR():

    site_lon, site_lat = [-147.43,   65.1192]
    azdat = readsav('../starmaps/PKR/PKR_DASC_5577_20260210_RAW_FULL_Az.sav', python_dict=True)
    eldat = readsav('../starmaps/PKR/PKR_DASC_5577_20260210_RAW_FULL_El.sav', python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask




# Venetie (VEE)
def load_VEE():

    site_lon, site_lat = [-146.407,  67.013]
    azdat = readsav('../starmaps/VEE/VEE_GASI_20260210_050100_rot5_full_Az.sav', python_dict=True)
    eldat = readsav('../starmaps/VEE/VEE_GASI_20260210_050100_rot5_full_El.sav', python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask



# Beaver
def load_BVR():
    site_lon, site_lat = [-147.4,    66.36]

    azdat = readsav('../starmaps/BVR/BVR_20260210_090000_750_rot5_Az.sav', python_dict=True)
    eldat = readsav('../starmaps/BVR/BVR_20260210_090000_750_rot5_El.sav', python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap < 15.

    return site_lat, site_lon, azmap, elmap, mask


# Arctic Village
def load_ARV():
    site_lon, site_lat = [-145.533,  68.127]
    

    azdat = readsav('../starmaps/ARV/ARV_GASI_20260209_063700_rot5_full_Az.sav', python_dict=True)
    eldat = readsav('../starmaps/ARV/ARV_GASI_20260209_063700_rot5_full_El.sav', python_dict=True)
    azmap = azdat[list(azdat.keys())[0]].copy()
    elmap = eldat[list(eldat.keys())[0]].copy()

    mask = elmap < 15.
  
    # Downsample to 750x750 to match TIFF images
    target_shape = (750, 750)
    azmap = resize(azmap, target_shape, order=1, preserve_range=True, anti_aliasing=True)
    elmap = resize(elmap, target_shape, order=1, preserve_range=True, anti_aliasing=True)
    mask = resize(mask.astype(float), target_shape, order=0, preserve_range=True) > 0.5

    return site_lat, site_lon, azmap, elmap, mask

