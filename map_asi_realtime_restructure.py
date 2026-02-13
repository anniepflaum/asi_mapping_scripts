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


from resolvedvelocities.ResolveVectorsLat import ResolveVectorsLat


apex = Apex()

## Load the latitude and longitude mapping arrays from the skymap.mat file for a given altitude
def load_skymaps():
    # Load skymaps for all cameras
    skymaps = dict()

    lat, lon, az, el, mask = skymap.load_ARV()
    skymaps['ARV'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    lat, lon, az, el, mask = skymap.load_VEE()
    skymaps['VEE'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    lat, lon, az, el, mask = skymap.load_BVR()
    skymaps['BVR'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    lat, lon, az, el, mask = skymap.load_PKR()
    skymaps['PKR'] = {'site_lat':lat, 'site_lon':lon, 'azmt':az, 'elev':el, 'mask':mask}

    for sm in skymaps.values():
        lat, lon = skymap.azel2geo(sm['site_lat'], sm['site_lon'], sm['azmt'], sm['elev'], alt=110.)
        sm['lat'] = lat
        sm['lon'] = lon

    return skymaps



# This function is needed due to an issue with cartopy where vectors are not scale/rotated
#   correctly in some coordinate systems (see: https://github.com/SciTools/cartopy/issues/1179)
def scale_uv(lon, lat, u, v):
    us = u/np.cos(lat*np.pi/180.)
    vs = v
    sf = np.sqrt(u**2+v**2)/np.sqrt(us**2+vs**2)
    return us*sf, vs*sf



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


def load_imagery():

    imgs = dict()

    # --- Download the latest ARV green channel image (already single-channel) ---
    url = 'https://optics.gi.alaska.edu/realtime/latest/arv_558_latest.jpg'
    im = retrieve_image(url)
    imgs['ARV'] = np.flipud(im)


    # --- Download the latest VEE green channel image (already single-channel) ---
    url = 'https://optics.gi.alaska.edu/realtime/latest/vee_558_latest.jpg'
    im = retrieve_image(url)
    imgs['VEE'] = np.flipud(im)


    # --- Download the latest BVR green channel image (already single-channel) ---
    url = 'https://optics.gi.alaska.edu/realtime/latest/bvr_558_latest.jpg'
    im = retrieve_image(url)
    imgs['BVR'] = np.flipud(im)


    # --- Download the latest PKR green channel image (already single-channel) ---
    url = "https://optics.gi.alaska.edu/realtime/latest/pkr_latest_green.jpg"
    imgs['PKR'] = retrieve_image(url)

    return imgs





# --- Overlay rocket trajectories from text files (if available) ---
#def load_traj(filename, offset=0.):
#    
#    #    lat1, lon1, latm1, lonm1, lata1, lona1 = load_traj('Traj_Left.txt')
#    #    lat2, lon2, latm2, lonm2, lata2, lona2 = load_traj('Traj_Right.txt')
#
#    # Load latitude and longitude columns from a trajectory text file
#    times, lats, lons, alts = np.loadtxt(filename, skiprows=1, unpack=True)
#    times -= offset
#
##    # TRUE
##    true_lata, true_lona = [67.009, -147.317]
##    true_lata, true_lona = [66.971, -146.049]
#    
#    # Map to 110 km along field lines
#    lats, lons, _ = apex.map_to_height(lats, lons, alts, 110.)
#
#    # Rocket trajectories every minute
#    idx = np.argwhere(times % 60 == 0)
#    timem = times[idx].squeeze()
#    latsm = lats[idx].squeeze()     # lat every minute
#    lonsm = lons[idx].squeeze()     # lon every minute
#    aidx = np.argmax(alts)
#    lata = lats[aidx]       # lat of appogee
#    lona = lons[aidx]       # lon of appogee
#
#    return lats, lons, latsm, lonsm, lata, lona

def load_trajectory():

    traj = dict()
    timel, latl, lonl, altl = np.loadtxt('Traj_Left.txt', skiprows=1, unpack=True)
    # Map to 110 km along field lines
    latl, lonl, _ = apex.map_to_height(latl, lonl, altl, 110.)
    traj['Left'] = {'coords':[latl, lonl]}

    timer, latr, lonr, altr = np.loadtxt('Traj_Right.txt', skiprows=1, unpack=True)
    # Map to 110 km along field lines
    latr, lonr, _ = apex.map_to_height(latr, lonr, altr, 110.)
    #timer -= 30.    # Shift right trajectory time by 30 seconds to line up with left trajectory
    traj['Right'] = {'coords':[latr, lonr]}


    # Coords every minute
    idx = np.argwhere(timel % 100 == 0)
    #timem = times[idx].squeeze()
    #latml = latl[idx].squeeze()     # lat every minute
    #lonml = lonl[idx].squeeze()     # lon every minute
    traj['Left']['min'] = [latl[idx].squeeze(), lonl[idx].squeeze()]
    idx = np.argwhere((timer+30) % 100 == 0)
    traj['Right']['min'] = [latr[idx].squeeze(), lonr[idx].squeeze()]

    # Coords at apogee
    aidx = np.argmax(altl)
    traj['Left']['app'] = [latl[aidx],  lonl[aidx]]
#    lata = lats[aidx]       # lat of appogee
#    lona = lons[aidx]       # lon of appogee
    aidx = np.argmax(altr)
    traj['Right']['app'] = [latr[aidx],  lonr[aidx]]

    # Coords at left = 280; right = 250
    idx = np.argmin(np.abs(timel-280.))
    traj['Left']['mag'] = [latl[idx].squeeze(), lonl[idx].squeeze()]
    idx = np.argmin(np.abs(timer-250.))
    traj['Right']['mag'] = [latr[idx].squeeze(), lonr[idx].squeeze()]



    true_apogee = {'Left': [67.009, -147.317, 319.], 'Right':[66.971, -146.049, 327.]}
    for k, coords in true_apogee.items():
        #for tc in true_apogee:
        lat, lon, alt = coords
    
        # Map to 110 km along field lines
        true_lat, true_lon, _ = apex.map_to_height(lat, lon, alt, 110.)

        traj[k]['true'] = [true_lat, true_lon]

        #ax.scatter(true_lon, true_lat, color='lightcyan', transform=axtrans)

#    traj = dict()
#    lat, lon, latm, lonm, lata, lona = load_traj('Traj_Left.txt')
#    traj['Left'] = {'coords':[lat,lon], 'min':[latm,lonm], 'app':[lata,lona]}
#
#    lat, lon, latm, lonm, lata, lona = load_traj('Traj_Right.txt', offset=30.)
#    traj['Right'] = {'coords':[lat,lon], 'min':[latm,lonm], 'app':[lata,lona]}

    return traj



def retrieve_pfisr():
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
       dne = h5['FittedParams/dNe'][:]
       glat = h5['Geomag/Latitude'][:]
       glon = h5['Geomag/Longitude'][:]
       galt = h5['Geomag/Altitude'][:]

    ne[dne>ne] = np.nan

    # Calculate vvels
    vvels = ResolveVectorsLat('vvels_config.ini')
    vvels.transform()
    vvels.bin_data_mlat()
    vvels.compute_vector_velocity()
    vvels.compute_electric_field()
    vvels.compute_geodetic_output()

    # Map to 110 km along field lines
    glat, glon, _ = apex.map_to_height(glat, glon, galt/1000., 110.)

    aidx = np.argmin(np.abs(vvels.outalt-110.))
    vv = vvels.Velocity_gd[0,aidx,:,:]
    vm = vvels.Vgd_mag[0,aidx,:]
    ve = vvels.Vgd_mag_err[0,aidx,:]
    vlat = vvels.bin_glat[aidx,:]
    vlon = vvels.bin_glon[aidx,:]

    vv[ve>vm,:] = [np.nan, np.nan, np.nan]
    vm[ve>vm] = np.nan

    pfisr_data = {'ne':ne, 'glat':glat, 'glon':glon, 'vel':vv, 'mag':vm, 'vlat':vlat, 'vlon':vlon}

    return pfisr_data



####### PLOTTING############

def generate_map(skymaps, imgs, pfisr, traj, fast=False):


    # Plot setup
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4,4, width_ratios=[4,0.2,0.2,1])

    if fast:

        ax = fig.add_subplot(gs[:,0])
        coastlons = np.loadtxt('coastlon.txt')
        coastlats = np.loadtxt('coastlat.txt')
    
        ax.plot(coastlons,coastlats, color='black')#,s=15) 
        ax.set_ylim(ymin=57.5, ymax=72)
        ax.set_xlim(xmin=-170, xmax=-135)
        ax.set_aspect(2.2)
        ax.grid()
        axtrans = ax.transData
    else:
        proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
        ax = fig.add_subplot(gs[:,0], projection=proj)
        # fast setup
        ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
        ax.set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
        ax.gridlines()
        mgl = mcm.maggridlines(ax, apex=apex, apex_height=110.)
        axtrans = ccrs.PlateCarree()


    # --- Setup sidebar plots ---
    ax1 = dict()
    axtrans1 = dict()
    for i, site in enumerate(imgs.keys()):
        if fast:
            # FAST
            ax1[site] = fig.add_subplot(gs[i,-1])
            ax1[site].plot(coastlons,coastlats, color='black')#,s=15) 
            ax1[site].set_ylim(ymin=57.5, ymax=72)
            ax1[site].set_xlim(xmin=-170, xmax=-135)
            ax1[site].set_aspect(2.2)
            ax1[site].grid()
            ax1[site].set_title(site)
            axtrans1[site] = ax1[site].transData
        else:
            ax1[site] = fig.add_subplot(gs[i,-1], projection=proj)
            ax1[site].add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
            ax1[site].add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
            ax1[site].add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
            ax1[site].add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
            ax1[site].set_extent([-170, -140, 57, 72], crs=ccrs.PlateCarree())
            ax1[site].gridlines()
            mgl = mcm.maggridlines(ax, apex=apex, apex_height=110.)
            axtrans1[site] = ccrs.PlateCarree()



    for site, img in imgs.items():
        print(site)

        # apply mask
        img[skymaps[site]['mask']] = np.nan
        im = img.copy()
        for m in skymaps[site]['extra_masks'].values():
            im[m] = np.nan
        # Partial FoV for main plots
        im_handle = ax.pcolor(skymaps[site]['lon'], skymaps[site]['lat'], im, vmin=0, transform=axtrans)
        #im_handle = ax.pcolor(skymaps[site]['lon'], skymaps[site]['lat'], im, vmin=0)
        # Full FoV for sidebar plots
        ax1[site].pcolor(skymaps[site]['lon'], skymaps[site]['lat'], img, vmin=0, transform=axtrans1[site])

    # --- Overlay PFISR ---
    print('PFISR')
    pfisr_handle = ax.scatter(pfisr['glon'], pfisr['glat'], c=pfisr['ne'], zorder=6, cmap='jet', vmin=0, vmax=4e11, transform=axtrans)
    u, v = scale_uv(pfisr['vlon'], pfisr['vlat'], pfisr['vel'][:,0], pfisr['vel'][:,1])
    qp = ax.quiver(pfisr['vlon'], pfisr['vlat'], u, v, zorder=7, scale=5000, width=0.005, transform=axtrans)


    # --- Overlay rocket trajectories from text files ---
    print('Trajectories')
    
    #lat1, lon1, latm1, lonm1, lata1, lona1 = load_traj('Traj_Left.txt')
    #lat2, lon2, latm2, lonm2, lata2, lona2 = load_traj('Traj_Right.txt')

    #ax.plot(lon1, lat1, color='red', label='GNEISS trajectory',zorder=7, transform=axtrans)
    for t in traj.values():
        ax.plot(t['coords'][1], t['coords'][0], color='red', label='GNEISS trajectory',zorder=7, transform=axtrans)
        ax.scatter(t['min'][1], t['min'][0], color='red', s=15, zorder=7, transform=axtrans)
        ax.scatter(t['app'][1], t['app'][0], color='lavenderblush', label='Apogee', marker='x', zorder=7, transform=axtrans)
        ax.scatter(t['mag'][1], t['mag'][0], color='oldlace', label='Apogee', marker='+', zorder=7, transform=axtrans)

        ax.scatter(t['true'][1], t['true'][0], color='lightcyan', marker='*', transform=axtrans)
    #ax.plot(lon2, lat2, color='red', zorder=7, transform=axtrans)
    #ax.scatter(lonm2, latm2, color='red', s=15, zorder=7, transform=axtrans)
    #ax.scatter(lona2, lata2, color='lavenderblush', marker='x', zorder=7, transform=axtrans)


    # TRUE
    true_apogee = [[67.009, -147.317, 319.], [66.971, -146.049, 327.]]
    for tc in true_apogee:
        true_lata, true_lona, true_alta = tc
    
        # Map to 110 km along field lines
        true_lat, true_lon, _ = apex.map_to_height(true_lata, true_lona, true_alta, 110.)

        ax.scatter(true_lon, true_lat, color='lightcyan', transform=axtrans)





    # Final output plot
    txt = ax.text(0.99, 0.01, dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), transform=ax.transAxes, fontsize=12, color='w', ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.set_title("GNEISS Ground Sites (magnetic footpointing to 110 km)")
    ax.legend(loc='upper right')

    # --- Add colorbar and save the figure ---
    ax.quiverkey(qp, 0.1, 0.9, 500., '500 m/s', transform=ax.transAxes)
    cax = fig.add_subplot(gs[:,1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation='vertical')
    cbar.set_label('Green Channel Intensity')
    cax = fig.add_subplot(gs[:,2])
    cbar = fig.colorbar(pfisr_handle, cax=cax, orientation='vertical')
    cbar.set_label(r'Electron Density (m$^{-3}$)')
    plt.tight_layout()


    #output_path = f"../launch_science_fast/GNEISS_launch_science_fast_{dt.datetime.utcnow():%Y%m%dT%H%M%S}.png"
    output_path = f"GNEISS_launch_science.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
    #plt.show()







def main():

    ticall=time.time()

    # --- Parse command-line arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action='store_true')
    args = ap.parse_args()


    # --- Load the geographic mapping for the ASI image ---
    skymaps = load_skymaps()



    # --- Calculate mask for overlaping images
    sites = list(skymaps.keys())
    for s0 in sites:
        red_sites = sites.copy()
        red_sites.remove(s0)
        skymaps[s0]['extra_masks'] = dict()
        for s1 in red_sites:
            m0, m1 = ci.calculate_masks(skymaps[s0]['site_lat'], skymaps[s0]['site_lon'], skymaps[s0]['azmt'], skymaps[s0]['elev'], skymaps[s1]['site_lat'], skymaps[s1]['site_lon'], skymaps[s1]['azmt'], skymaps[s1]['elev'])

            skymaps[s0]['extra_masks'][s1] = m0


#    imgs = dict()
#
#    # --- Download the latest ARV green channel image (already single-channel) ---
#    url = 'https://optics.gi.alaska.edu/realtime/latest/arv_558_latest.jpg'
#    im = retrieve_image(url)
#    imgs['ARV'] = np.flipud(im)
#
#
#    # --- Download the latest VEE green channel image (already single-channel) ---
#    url = 'https://optics.gi.alaska.edu/realtime/latest/vee_558_latest.jpg'
#    im = retrieve_image(url)
#    imgs['VEE'] = np.flipud(im)
#
#
#    # --- Download the latest BVR green channel image (already single-channel) ---
#    url = 'https://optics.gi.alaska.edu/realtime/latest/bvr_558_latest.jpg'
#    im = retrieve_image(url)
#    imgs['BVR'] = np.flipud(im)
#
#
#    # --- Download the latest PKR green channel image (already single-channel) ---
#    url = "https://optics.gi.alaska.edu/realtime/latest/pkr_latest_green.jpg"
#    imgs['PKR'] = retrieve_image(url)



    imgs = load_imagery()


    pfisr = retrieve_pfisr()

    traj = load_trajectory()

    generate_map(skymaps, imgs, pfisr, traj, fast=args.fast)


    tocall=time.time()
    print(tocall-ticall,'s total run time')

if __name__ == "__main__":
    main()
