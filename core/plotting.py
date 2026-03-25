from inspect import currentframe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import magcoordmap as mcm
from core.brightness import best_rocket_brightness
from core.paths import COAST_LAT_PATH, COAST_LON_PATH, LEFT_TRAJECTORY_PATH, RIGHT_TRAJECTORY_PATH
from core.plot_norm import choose_image_cmap, compute_linear_image_limits, compute_log_image_limits
from core.time_utils import format_time_label, sanitize_time_for_filename
from core.traj_utils import format_time_since_launch, get_launch_start_from_traj_csv, load_traj


def scale_uv(lon, lat, u, v):
    """
    Adjust vector scaling/rotation for cartopy quiver plots to account for latitude distortion.
    """
    us = u / np.cos(lat * np.pi / 180.0)
    vs = v
    sf = np.sqrt(u ** 2 + v ** 2) / np.sqrt(us ** 2 + vs ** 2)
    return us * sf, vs * sf


def channel_label(color):
    return "Red Channel Intensity" if str(color).lower() == "red" else "Green Channel Intensity"


def plot_fast(skymaps, imgs, pfisr, output_path=None, map_time=None, bounds=None, color="green", imgs_raw=None, norm_limits=None, colorbar_scale="linear", colorbar_color="viridis"):
    coastlons = np.loadtxt(COAST_LON_PATH)
    coastlats = np.loadtxt(COAST_LAT_PATH)
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(coastlons, coastlats, color="black")
    lon_min, lon_max, lat_min, lat_max = bounds if bounds is not None else (-170, -135, 57.5, 72)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_xlim(xmin=lon_min, xmax=lon_max)
    ax.set_aspect(2.2)
    ax.grid()
    ax1 = {}
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1])
        ax1[site].plot(coastlons, coastlats, color="black")
        ax1[site].set_ylim(ymin=lat_min, ymax=lat_max)
        ax1[site].set_xlim(xmin=lon_min, xmax=lon_max)
        ax1[site].set_aspect(2.2)
        ax1[site].grid()
        ax1[site].set_title(site)

    side_images = {}
    main_images = {}
    norm_pool = []
    for site, img in imgs.items():
        side_img = img.copy()
        side_img[skymaps[site]["mask"]] = np.nan
        if colorbar_scale == "log":
            side_img[side_img <= 0] = np.nan
        main_img = side_img.copy()
        for mask in skymaps[site]["extra_masks"].values():
            main_img[mask] = np.nan
        side_images[site] = side_img
        main_images[site] = main_img
        vals = main_img[np.isfinite(main_img)]
        if vals.size > 0:
            norm_pool.append(vals)

    if norm_limits is not None and norm_limits[0] is not None and norm_limits[1] is not None:
        global_vmin, global_vmax = norm_limits
    elif colorbar_scale == "log":
        global_vmin, global_vmax = compute_log_image_limits(norm_pool)
    else:
        global_vmin, global_vmax = compute_linear_image_limits(norm_pool)

    image_norm = None
    if global_vmin is not None and global_vmax is not None and colorbar_scale == "log":
        image_norm = mpl.colors.LogNorm(vmin=global_vmin, vmax=global_vmax)

    image_cmap = choose_image_cmap(colorbar_color, color)
    for site in imgs.keys():
        main_img = main_images[site]
        side_img = side_images[site]
        if image_norm is None and global_vmin is None:
            im_handle = ax.pcolor(skymaps[site]["lon"], skymaps[site]["lat"], main_img, cmap=image_cmap)
            ax1[site].pcolor(skymaps[site]["lon"], skymaps[site]["lat"], side_img, cmap=image_cmap)
        elif image_norm is None:
            im_handle = ax.pcolor(skymaps[site]["lon"], skymaps[site]["lat"], main_img, cmap=image_cmap, vmin=global_vmin, vmax=global_vmax)
            ax1[site].pcolor(skymaps[site]["lon"], skymaps[site]["lat"], side_img, cmap=image_cmap, vmin=global_vmin, vmax=global_vmax)
        else:
            im_handle = ax.pcolor(skymaps[site]["lon"], skymaps[site]["lat"], main_img, cmap=image_cmap, norm=image_norm)
            ax1[site].pcolor(skymaps[site]["lon"], skymaps[site]["lat"], side_img, cmap=image_cmap, norm=image_norm)

    lat1, lon1, latm1, lonm1, _lata1, _lona1, lat_map1, lon_map1 = load_traj(str(LEFT_TRAJECTORY_PATH), map_time=map_time)
    lat2, lon2, latm2, lonm2, _lata2, _lona2, lat_map2, lon_map2 = load_traj(str(RIGHT_TRAJECTORY_PATH), map_time=map_time)
    ax.plot(lon1, lat1, color="red", label="GNEISS trajectory", zorder=7)
    ax.scatter(lonm1, latm1, color="red", s=15, zorder=7)
    ax.plot(lon2, lat2, color="red", zorder=7)
    ax.scatter(lonm2, latm2, color="red", s=15, zorder=7)
    if lat_map1 is not None and lon_map1 is not None:
        ax.scatter(lon_map1, lat_map1, color="orange", s=50, marker="o", zorder=8, label="Position at map time")
    if lat_map2 is not None and lon_map2 is not None:
        ax.scatter(lon_map2, lat_map2, color="orange", s=50, marker="o", zorder=8)

    bright1 = None
    bright2 = None
    if imgs_raw is not None:
        if lat_map1 is not None and lon_map1 is not None:
            bright1 = best_rocket_brightness(lat_map1, lon_map1, skymaps, imgs_raw)
        if lat_map2 is not None and lon_map2 is not None:
            bright2 = best_rocket_brightness(lat_map2, lon_map2, skymaps, imgs_raw)
        if bright1 is not None:
            print(f"Rocket Left brightness percentile: {bright1['site']} P={bright1['percentile']:.1f} (nearest {bright1['distance_deg']:.3f} deg)")
        if bright2 is not None:
            print(f"Rocket Right brightness percentile: {bright2['site']} P={bright2['percentile']:.1f} (nearest {bright2['distance_deg']:.3f} deg)")

    frame = currentframe()
    args = frame.f_back.f_locals.get("args", None)
    if args is not None:
        date_str = args.date
        time_str = args.time
        left_tplus = format_time_since_launch(time_str, get_launch_start_from_traj_csv(str(LEFT_TRAJECTORY_PATH)))
        right_tplus = format_time_since_launch(time_str, get_launch_start_from_traj_csv(str(RIGHT_TRAJECTORY_PATH)))
        label_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}\n36.397 {left_tplus} | 36.398 {right_tplus}"
    else:
        label_str = ""
    ax.text(
        0.99,
        0.01,
        label_str,
        transform=ax.transAxes,
        fontsize=12,
        color="w",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
    )
    ax.set_title(f"Mapped ASIs and GNEISS trajectory ({color} channel)")
    ax.legend(loc="upper right")
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation="vertical")
    cbar.set_label(channel_label(color))
    marker_specs = []
    if bright1 is not None:
        marker_specs.append(("L", bright1))
    if bright2 is not None:
        marker_specs.append(("R", bright2))
    for tag, bright in marker_specs:
        y = np.clip(bright["percentile"] / 100.0, 0.0, 1.0)
        cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="black", linewidth=4.0, zorder=1000, solid_capstyle="butt", clip_on=False)
        cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="white", linewidth=2.2, zorder=1001, solid_capstyle="butt", clip_on=False)
        cbar.ax.text(-0.05, y, f"{tag}: {bright['site']} P{bright['percentile']:.1f}", transform=cbar.ax.transAxes, color="black", fontsize=8, va="center", ha="right", clip_on=False)

    plt.tight_layout()
    if output_path is None:
        if args is not None:
            output_path = f"../mapped/GNEISS_launch_science_fast_{date_str}_{sanitize_time_for_filename(time_str)}.png"
        else:
            output_path = "../mapped/GNEISS_launch_science_fast.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")


def plot_pretty(skymaps, imgs, pfisr, output_path=None, bounds=None, color="green", colorbar_scale="linear", colorbar_color="viridis", apex=None):
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0], projection=proj)
    lon_min, lon_max, lat_min, lat_max = bounds if bounds is not None else (-170, -140, 57, 72)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    ax.gridlines()
    mcm.maggridlines(ax, apex=apex, apex_height=110.0)
    ax1 = {}
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1], projection=proj)
        ax1[site].coastlines()
        ax1[site].gridlines()
        mcm.maggridlines(ax1[site], apex=apex, apex_height=110.0)
        ax1[site].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax1[site].set_title(site)

    image_cmap = choose_image_cmap(colorbar_color, color)
    norm_pool = []
    for site, img in imgs.items():
        img[skymaps[site]["mask"]] = np.nan
        if colorbar_scale == "log":
            img[img <= 0] = np.nan
        mapped = img.copy()
        for mask in skymaps[site]["extra_masks"].values():
            mapped[mask] = np.nan
        vals = mapped[np.isfinite(mapped)]
        if vals.size > 0:
            norm_pool.append(vals)
    if colorbar_scale == "log":
        vmin, vmax = compute_log_image_limits(norm_pool)
        image_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else None
    else:
        vmin, vmax = compute_linear_image_limits(norm_pool)
        image_norm = None

    for site, img in imgs.items():
        img[skymaps[site]["mask"]] = np.nan
        if colorbar_scale == "log":
            img[img <= 0] = np.nan
        mapped = img.copy()
        lat = skymaps[site]["lat"].copy()
        lon = skymaps[site]["lon"].copy()
        for mask in skymaps[site]["extra_masks"].values():
            mapped[mask] = np.nan
        img_flat = img[~skymaps[site]["mask"]].flatten()
        lon_flat = skymaps[site]["lon"][~skymaps[site]["mask"]].flatten()
        lat_flat = skymaps[site]["lat"][~skymaps[site]["mask"]].flatten()
        imf = mapped[np.isfinite(mapped)].flatten()
        latf = lat[np.isfinite(mapped)].flatten()
        lonf = lon[np.isfinite(mapped)].flatten()
        if image_norm is None and vmin is not None and vmax is not None:
            im_handle = ax1[site].tripcolor(lon_flat, lat_flat, img_flat, zorder=3, transform=ccrs.PlateCarree(), cmap=image_cmap, vmin=vmin, vmax=vmax)
            ax.tripcolor(lonf, latf, imf, transform=ccrs.PlateCarree(), cmap=image_cmap, vmin=vmin, vmax=vmax)
        else:
            im_handle = ax1[site].tripcolor(lon_flat, lat_flat, img_flat, zorder=3, transform=ccrs.PlateCarree(), cmap=image_cmap, norm=image_norm)
            ax.tripcolor(lonf, latf, imf, transform=ccrs.PlateCarree(), cmap=image_cmap, norm=image_norm)

    print("Trajectory")
    lat1, lon1, latm1, lonm1, _lata1, _lona1, lat_map1, lon_map1 = load_traj(str(LEFT_TRAJECTORY_PATH))
    lat2, lon2, latm2, lonm2, _lata2, _lona2, lat_map2, lon_map2 = load_traj(str(RIGHT_TRAJECTORY_PATH))
    ax.plot(lon1, lat1, color="red", label="GNEISS trajectory", transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lonm1, latm1, color="red", s=15, transform=ccrs.PlateCarree(), zorder=7)
    ax.plot(lon2, lat2, color="red", transform=ccrs.PlateCarree(), zorder=7)
    ax.scatter(lonm2, latm2, color="red", s=15, transform=ccrs.PlateCarree(), zorder=7)
    if lat_map1 is not None and lon_map1 is not None:
        ax.scatter(lon_map1, lat_map1, color="orange", s=50, marker="o", zorder=8, label="Position at map time")
    if lat_map2 is not None and lon_map2 is not None:
        ax.scatter(lon_map2, lat_map2, color="orange", s=50, marker="o", zorder=8)

    frame = currentframe()
    args = frame.f_back.f_locals.get("args", None)
    if args is not None:
        date_str = args.date
        time_str = args.time
        left_tplus = format_time_since_launch(time_str, get_launch_start_from_traj_csv(str(LEFT_TRAJECTORY_PATH)))
        right_tplus = format_time_since_launch(time_str, get_launch_start_from_traj_csv(str(RIGHT_TRAJECTORY_PATH)))
        label_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}\n36.397 {left_tplus} | 36.398 {right_tplus}"
    else:
        label_str = ""
    ax.text(
        0.99,
        0.01,
        label_str,
        transform=ax.transAxes,
        fontsize=12,
        color="w",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
    )
    ax.set_title(f"Mapped ASIs and GNEISS trajectory ({color} channel)")
    ax.legend(loc="upper right")
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation="vertical")
    cbar.set_label(channel_label(color))

    plt.tight_layout()
    if output_path is None:
        if args is not None:
            output_path = f"../launch_science_pretty/GNEISS_launch_science_pretty_{date_str}_{sanitize_time_for_filename(time_str)}.png"
        else:
            output_path = "../launch_science_pretty/GNEISS_launch_science_pretty.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")
