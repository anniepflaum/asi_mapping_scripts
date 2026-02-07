# generate_skymap.py

import numpy as np
from astropy.io import fits
from scipy.io import readsav
import pymap3d as pm
import h5py


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

    #print(az[az>0.], el[el>0.])

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
    
    azmap = np.rot90(fits.open('PKR_DASC_20220305_Az.FIT')[0].data,3).T
    elmap = np.rot90(fits.open('PKR_DASC_20220305_El.FIT')[0].data,3).T

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask




# Venetie (VEE)
def load_VEE():

    site_lon, site_lat = [-146.407,  67.013]
    dat = readsav('VEE_558_latest_az_el_512.sav', python_dict=True)
    azmap = dat['az_latest_512'].copy()
    elmap = dat['el_latest_512'].copy()

    # Super hacky fix to interpolation across the az=0 line
    # This is horrible code, do not repeat anywhere
    ul = [88, 109]
    lr = [247,288]
    i0,j0 = ul
    i1,j1 = lr

    ivec = np.arange(i1-i0) + i0
    m = (j1-j0)/(i1-i0)
    jvec = m*(ivec-i0) + j0
    jvec2 = jvec.astype(int)-4
    jvec3 = jvec.astype(int)+4

    for i in ivec:
        fix_area = azmap[jvec2[i-i0]:jvec3[i-i0],i]
        fix_area[(fix_area>10.) & (fix_area<350.)] = 0.
        azmap[jvec2[i-i0]:jvec3[i-i0],i] = fix_area

    #import matplotlib.pyplot as plt
    #plt.imshow(azmap)
    #plt.show()

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask



# Beaver
def load_BVR():
    site_lon, site_lat = [-147.4,    66.36]

    dat = readsav('BVR_558_latest_az_el_512.sav', python_dict=True)
    azmap = dat['az_latest_512'].copy()
    elmap = dat['el_latest_512'].copy()

    mask = elmap<15.

    return site_lat, site_lon, azmap, elmap, mask


# Arctic Village

site_lon, site_lat = [-145.533,  68.127]



