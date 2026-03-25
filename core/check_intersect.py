# check_intersect.py
import numpy as np
import h5py
import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

RE = 6371.
h1 = 110.
h2 = 110.

## Load data files (just needed for example here)
#filename1 = 'mango-mto-redline-level1-20251111.hdf5'
#with h5py.File(filename1, 'r') as h5:
#    sitelat1, sitelon1 = h5['SiteInfo/GeodeticCoordinates'][:]
#    az1 = h5['Coordinates/Azimuth'][:]
#    el1 = h5['Coordinates/Elevation'][:]
#    lat1 = h5['Coordinates/Latitude'][:]
#    lon1 = h5['Coordinates/Longitude'][:]
#    h1 = h5['ProcessingInfo/Altitude'][()]
#    m1 = h5['Mask'][:]
#
#
#filename2 = 'mango-eio-redline-level1-20251111.hdf5'
#with h5py.File(filename2, 'r') as h5:
#    sitelat2, sitelon2 = h5['SiteInfo/GeodeticCoordinates'][:]
#    az2 = h5['Coordinates/Azimuth'][:]
#    el2 = h5['Coordinates/Elevation'][:]
#    lat2 = h5['Coordinates/Latitude'][:]
#    lon2 = h5['Coordinates/Longitude'][:]
#    h2 = h5['ProcessingInfo/Altitude'][()]
#    m2 = h5['Mask'][:]


def calculate_masks(sitelat1, sitelon1, az1, el1, sitelat2, sitelon2, az2, el2):

    # Convert site latitude and longitude to radians
    l1 = np.deg2rad(sitelat1)
    p1 = np.deg2rad(sitelon1)
    l2 = np.deg2rad(sitelat2)
    p2 = np.deg2rad(sitelon2)
    # Differences
    dl = (l2 - l1)
    dp = (p2 - p1)
    
    # Calculate the bearing
    y = np.sin(dp) * np.cos(l2)
    x = np.cos(l1) * np.sin(l2) - np.sin(l1) * np.cos(l2) * np.cos(dp)
    brg1 = np.atan2(y, x)
    
    y = np.sin(-dp) * np.cos(l1)
    x = np.cos(l2) * np.sin(l1) - np.sin(l2) * np.cos(l1) * np.cos(-dp)
    brg2 = np.atan2(y, x)
    
    # Calculate the angular distance
    a = np.sin(dl/2)**2 + np.sin(dp/2)**2 * np.cos(l1) * np.cos(l2)
    psi = 2 * np.asin(np.sqrt(a))
    
    
    # Calculate the masks
    Gam1 = np.deg2rad(az1) - brg1
    Gam2 = np.deg2rad(az2) - brg2
    lam1 = np.deg2rad(el1)
    lam2 = np.deg2rad(el2)
    
    A1 = np.sqrt(1-(RE/(RE+h1))**2*np.cos(lam1)**2)/(RE/(RE+h1)*np.cos(lam1))
    B1 = (A1-np.tan(lam1))/(1+A1*np.tan(lam1))*np.cos(Gam1)
    mask1 = B1 > np.tan(psi/2)
    #mask1 = np.logical_or(m1, B1 > np.tan(psi/2))
    
    A2 = np.sqrt(1-(RE/(RE+h2))**2*np.cos(lam2)**2)/(RE/(RE+h2)*np.cos(lam2))
    B2 = (A2-np.tan(lam2))/(1+A2*np.tan(lam2))*np.cos(Gam2)
    mask2 = B2 > np.tan(psi/2)
    #mask2 = np.logical_or(m2, B2 > np.tan(psi/2))

    return mask1, mask2



## Plot (for sanity check)
#fig = plt.figure(figsize=(10,3))
#gs = gridspec.GridSpec(1,3)
#
#ax = fig.add_subplot(gs[0])
#c = ax.imshow(mask1)
#fig.colorbar(c)
#
#ax = fig.add_subplot(gs[1], projection=ccrs.LambertConformal())
#ax.add_feature(cfeature.BORDERS, linestyle='-')
#ax.add_feature(cfeature.STATES, linestyle=':')
#ax.gridlines()
#ax.pcolormesh(lon1, lat1, mask1, alpha=0.5, transform=ccrs.PlateCarree())
#ax.pcolormesh(lon2, lat2, mask2, alpha=0.5, transform=ccrs.PlateCarree())
#
#ax = fig.add_subplot(gs[2])
#c = ax.imshow(mask2)
#fig.colorbar(c)
#
#plt.show()
