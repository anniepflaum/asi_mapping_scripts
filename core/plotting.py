import csv
from inspect import currentframe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import magcoordmap as mcm
from core.brightness import best_rocket_brightness
from core.calc_ipp import calc_ipp
from core.paths import COAST_LAT_PATH, COAST_LON_PATH, LEFT_TRAJECTORY_PATH, RECEIVERS_PATH, RIGHT_TRAJECTORY_PATH
from core.plot_norm import choose_image_cmap, compute_linear_image_limits, compute_log_image_limits
from core.time_utils import format_time_label, sanitize_time_for_filename
from core.traj_utils import (
    build_traj_lookup,
    format_time_since_launch,
    get_launch_start_from_traj_csv,
    load_traj,
    lookup_traj_geodetic_position,
    mapped_apex_height,
)


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


def load_receivers(path=RECEIVERS_PATH):
    with open(path, "r", encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd)
        return [
            {
                "name": row["Name"].strip(),
                "acronym": row["acronym"].strip(),
                "lon": float(row["Lon"]),
                "lat": float(row["Lat"]),
                "alt_m": float(row.get("Alt_m", 0.0) or 0.0),
            }
            for row in reader
            if row.get("Lon") and row.get("Lat")
        ]


def draw_receivers(ax, receivers, axtrans):
    if not receivers:
        return
    lons = [receiver["lon"] for receiver in receivers]
    lats = [receiver["lat"] for receiver in receivers]
    ax.scatter(lons, lats, marker="^", s=45, color="white", edgecolors="black", linewidths=0.8, zorder=9, label="Receivers", transform=axtrans)
    for receiver in receivers:
        ax.text(receiver["lon"] + 0.12, receiver["lat"] + 0.05, receiver["acronym"], fontsize=8, color="black", zorder=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.15), transform=axtrans)


def compute_receiver_ipps(receivers, rocket_geo, ipp_height_km):
    if (
        not receivers
        or rocket_geo[0] is None
        or rocket_geo[1] is None
        or rocket_geo[2] is None
        or rocket_geo[2] < 110.0
    ):
        return []

    rocket_position = [rocket_geo[0], rocket_geo[1], rocket_geo[2] * 1000.0]
    ipps = []
    for receiver in receivers:
        ipp_lat, ipp_lon = calc_ipp(
            [receiver["lat"], receiver["lon"], receiver.get("alt_m", 0.0)],
            rocket_position,
            rockcoords="geo",
            height=ipp_height_km,
        )
        ipps.append(
            {
                "acronym": receiver["acronym"],
                "lat": float(ipp_lat),
                "lon": float(ipp_lon),
            }
        )
    return ipps


def get_plot_call_args():
    frame = currentframe()
    while frame is not None:
        args = frame.f_locals.get("args", None)
        if args is not None:
            return args
        frame = frame.f_back
    return None


def build_time_label(args):
    if args is None:
        return ""
    date_str = args.date
    time_str = args.time
    left_tplus = format_time_since_launch(time_str, get_launch_start_from_traj_csv(str(LEFT_TRAJECTORY_PATH)))
    right_tplus = format_time_since_launch(time_str, get_launch_start_from_traj_csv(str(RIGHT_TRAJECTORY_PATH)))
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}\n36.397 {left_tplus} | 36.398 {right_tplus}"


def load_trajectory_context(map_time, color, receivers, plot_ipps):
    lat1, lon1, latm1, lonm1, _lata1, _lona1, lat_map1, lon_map1 = load_traj(str(LEFT_TRAJECTORY_PATH), map_time=map_time, color=color)
    lat2, lon2, latm2, lonm2, _lata2, _lona2, lat_map2, lon_map2 = load_traj(str(RIGHT_TRAJECTORY_PATH), map_time=map_time, color=color)
    left_ipps = []
    right_ipps = []
    if plot_ipps:
        left_geo = lookup_traj_geodetic_position(build_traj_lookup(str(LEFT_TRAJECTORY_PATH), color=color), map_time)
        right_geo = lookup_traj_geodetic_position(build_traj_lookup(str(RIGHT_TRAJECTORY_PATH), color=color), map_time)
        ipp_height_km = mapped_apex_height(color)
        left_ipps = compute_receiver_ipps(receivers, left_geo, ipp_height_km)
        right_ipps = compute_receiver_ipps(receivers, right_geo, ipp_height_km)
    return {
        "left": {"lat": lat1, "lon": lon1, "lat_minute": latm1, "lon_minute": lonm1, "lat_map": lat_map1, "lon_map": lon_map1, "ipps": left_ipps},
        "right": {"lat": lat2, "lon": lon2, "lat_minute": latm2, "lon_minute": lonm2, "lat_map": lat_map2, "lon_map": lon_map2, "ipps": right_ipps},
    }


def prepare_image_layers(skymaps, imgs, colorbar_scale, norm_limits=None):
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
        vmin, vmax = norm_limits
    elif colorbar_scale == "log":
        vmin, vmax = compute_log_image_limits(norm_pool)
    else:
        vmin, vmax = compute_linear_image_limits(norm_pool)
    image_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax) if colorbar_scale == "log" and vmin is not None and vmax is not None else None
    return side_images, main_images, vmin, vmax, image_norm



def setup_fast_axes(imgs, bounds):
    coastlons = np.loadtxt(COAST_LON_PATH)
    coastlats = np.loadtxt(COAST_LAT_PATH)
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0])
    axtrans = ax.transData
    ax.plot(coastlons, coastlats, color="black")
    lon_min, lon_max, lat_min, lat_max = bounds if bounds is not None else (-170, -135, 57.5, 72)
    ax.set_ylim(ymin=lat_min, ymax=lat_max)
    ax.set_xlim(xmin=lon_min, xmax=lon_max)
    ax.set_aspect(2.2)
    ax.grid()
    ax1 = {}
    axtrans1 = {}
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1])
        axtrans1[site] = ax1[site].transData
        ax1[site].plot(coastlons, coastlats, color="black")
        ax1[site].set_ylim(ymin=lat_min, ymax=lat_max)
        ax1[site].set_xlim(xmin=lon_min, xmax=lon_max)
        ax1[site].set_aspect(2.2)
        ax1[site].grid()
        ax1[site].set_title(site)
    return fig, gs, ax, ax1, axtrans, axtrans1


def setup_pretty_axes(imgs, bounds, apex):
    proj = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=55, standard_parallels=(55, 65))
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, width_ratios=[4, 0.2, 0.2, 1])
    ax = fig.add_subplot(gs[:, 0], projection=proj)
    axtrans = ccrs.PlateCarree()
    lon_min, lon_max, lat_min, lat_max = bounds if bounds is not None else (-170, -140, 57, 72)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, zorder=2)
    ax.gridlines()
    mcm.maggridlines(ax, apex=apex, apex_height=110.0)
    ax1 = {}
    axtrans1 = {}
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1], projection=proj)
        axtrans1[site] = ccrs.PlateCarree()
        ax1[site].coastlines()
        ax1[site].gridlines()
        mcm.maggridlines(ax1[site], apex=apex, apex_height=110.0)
        ax1[site].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax1[site].set_title(site)
    return fig, gs, ax, ax1, axtrans, axtrans1


def draw_images(ax, ax1, skymaps, imgs, side_images, main_images, image_cmap, vmin, vmax, image_norm, axtrans, axtrans1):
    im_handle = None
    for site in imgs.keys():
        main_img = main_images[site]
        side_img = side_images[site]
        if image_norm is None and vmin is None:
            im_handle = ax.pcolor(skymaps[site]["lon"], skymaps[site]["lat"], main_img, cmap=image_cmap, zorder=3, transform=axtrans)
            ax1[site].pcolor(skymaps[site]["lon"], skymaps[site]["lat"], side_img, cmap=image_cmap, transform=axtrans1[site])
        elif image_norm is None:
            im_handle = ax.pcolor(skymaps[site]["lon"], skymaps[site]["lat"], main_img, cmap=image_cmap, vmin=vmin, vmax=vmax, zorder=3, transform=axtrans)
            ax1[site].pcolor(skymaps[site]["lon"], skymaps[site]["lat"], side_img, cmap=image_cmap, vmin=vmin, vmax=vmax, transform=axtrans1[site])
        else:
            im_handle = ax.pcolor(skymaps[site]["lon"], skymaps[site]["lat"], main_img, cmap=image_cmap, norm=image_norm, zorder=3, transform=axtrans)
            ax1[site].pcolor(skymaps[site]["lon"], skymaps[site]["lat"], side_img, cmap=image_cmap, norm=image_norm, transform=axtrans1[site])
    return im_handle


def draw_trajectory_and_ipps(ax, traj_ctx, axtrans):
    left = traj_ctx["left"]
    right = traj_ctx["right"]
    ax.plot(left["lon"], left["lat"], color="red", label="GNEISS trajectory", zorder=7, transform=axtrans)
    ax.scatter(left["lon_minute"], left["lat_minute"], color="red", s=15, zorder=7, transform=axtrans)
    ax.plot(right["lon"], right["lat"], color="red", zorder=7, transform=axtrans)
    ax.scatter(right["lon_minute"], right["lat_minute"], color="red", s=15, zorder=7, transform=axtrans)
    if left["lat_map"] is not None and left["lon_map"] is not None:
        ax.scatter(left["lon_map"], left["lat_map"], color="orange", s=50, marker="o", zorder=8, label="Position at map time", transform=axtrans)
    if right["lat_map"] is not None and right["lon_map"] is not None:
        ax.scatter(right["lon_map"], right["lat_map"], color="orange", s=50, marker="o", zorder=8, transform=axtrans)
    for ipps, ipp_color, text_dy, label in (
        (left["ipps"], "deepskyblue", 0.03, "397 IPP"),
        (right["ipps"], "magenta", -0.08, "398 IPP"),
    ):
        if not ipps:
            continue
        ax.scatter([ipp["lon"] for ipp in ipps], [ipp["lat"] for ipp in ipps], marker="x", s=45, color=ipp_color, linewidths=1.4, zorder=11, label=label, transform=axtrans)
        for ipp in ipps:
            ax.text(ipp["lon"] + 0.1, ipp["lat"] + text_dy, ipp["acronym"], fontsize=7, color=ipp_color, zorder=12, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.12), transform=axtrans)


def sample_rocket_brightnesses(traj_ctx, skymaps, imgs_raw):
    bright1 = None
    bright2 = None
    if imgs_raw is not None:
        left = traj_ctx["left"]
        right = traj_ctx["right"]
        if left["lat_map"] is not None and left["lon_map"] is not None:
            bright1 = best_rocket_brightness(left["lat_map"], left["lon_map"], skymaps, imgs_raw)
        if right["lat_map"] is not None and right["lon_map"] is not None:
            bright2 = best_rocket_brightness(right["lat_map"], right["lon_map"], skymaps, imgs_raw)
        if bright1 is not None:
            print(f"Rocket Left brightness percentile: {bright1['site']} P={bright1['percentile']:.1f} (nearest {bright1['distance_deg']:.3f} deg)")
        if bright2 is not None:
            print(f"Rocket Right brightness percentile: {bright2['site']} P={bright2['percentile']:.1f} (nearest {bright2['distance_deg']:.3f} deg)")
    return bright1, bright2


def finalize_plot(ax, fig, gs, im_handle, color, label_str, output_path, default_with_args, default_without_args, brightness_markers=None):
    ax.text(0.99, 0.01, label_str, transform=ax.transAxes, fontsize=12, color="w", ha="right", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))
    ax.set_title(f"Mapped ASIs and GNEISS trajectory ({color} channel)")
    ax.legend(loc="upper right")
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im_handle, cax=cax, orientation="vertical")
    cbar.set_label(channel_label(color))
    for tag, bright in brightness_markers or []:
        y = np.clip(bright["percentile"] / 100.0, 0.0, 1.0)
        cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="black", linewidth=4.0, zorder=1000, solid_capstyle="butt", clip_on=False)
        cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="white", linewidth=2.2, zorder=1001, solid_capstyle="butt", clip_on=False)
        cbar.ax.text(-0.05, y, f"{tag}: {bright['site']} P{bright['percentile']:.1f}", transform=cbar.ax.transAxes, color="black", fontsize=8, va="center", ha="right", clip_on=False)

    args = get_plot_call_args()
    plt.tight_layout()
    if output_path is None:
        if args is not None:
            output_path = default_with_args(args)
        else:
            output_path = default_without_args
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")


def plot_map(skymaps, imgs, pfisr, output_path=None, map_time=None, bounds=None, color="green", imgs_raw=None, norm_limits=None, colorbar_scale="linear", colorbar_color="viridis", apex=None, plot_receivers=False, plot_ipps=False, pretty=False):
    receivers = load_receivers()
    if pretty:
        fig, gs, ax, ax1, axt, axt1 = setup_pretty_axes(imgs, bounds, apex)
    else:
        fig, gs, ax, ax1, axt, axt1 = setup_fast_axes(imgs, bounds)
    side_images, main_images, vmin, vmax, image_norm = prepare_image_layers(skymaps, imgs, colorbar_scale, norm_limits=norm_limits)
    image_cmap = choose_image_cmap(colorbar_color, color)
    im_handle = draw_images(ax, ax1, skymaps, imgs, side_images, main_images, image_cmap, vmin, vmax, image_norm, axt, axt1)
    traj_ctx = load_trajectory_context(map_time, color, receivers, plot_ipps)
    draw_trajectory_and_ipps(ax, traj_ctx, axt)
    if plot_receivers:
        draw_receivers(ax, receivers, axt)
    bright1, bright2 = sample_rocket_brightnesses(traj_ctx, skymaps, imgs_raw)
    brightness_markers = []
    if bright1 is not None:
        brightness_markers.append(("397", bright1))
    if bright2 is not None:
        brightness_markers.append(("398", bright2))
    finalize_plot(
        ax,
        fig,
        gs,
        im_handle,
        color,
        build_time_label(get_plot_call_args()),
        output_path,
        lambda args: f"../mapped/GNEISS_launch_science_fast_{args.date}_{sanitize_time_for_filename(args.time)}.png",
        "../mapped/GNEISS_launch_science_fast.png",
        brightness_markers=brightness_markers,
    )



