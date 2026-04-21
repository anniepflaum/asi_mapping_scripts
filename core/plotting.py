import csv
from inspect import currentframe
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import magcoordmap as mcm
from core.brightness import best_rocket_brightness
from core.calc_ipp import calc_ipp
from core.paths import COAST_LAT_PATH, COAST_LON_PATH, GNEISS_LEFT_TRAJECTORY_PATH, RECEIVERS_PATH, GNEISS_RIGHT_TRAJECTORY_PATH
from core.plot_norm import choose_image_cmap, compute_linear_image_limits, compute_log_image_limits
from core.time_utils import format_time_label, hhmmss_fractional_to_seconds, sanitize_time_for_filename
from core.traj_utils import (
    build_traj_lookup,
    fixed_utc_minute_marker_indices,
    format_time_since_launch,
    get_launch_start_from_traj_csv,
    load_traj,
    load_traj_records,
    lookup_traj_geodetic_position,
    mapped_apex_height,
    trajectory_marker_second,
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


def print_warning(message):
    print(f"Warning: {message}")


def load_receivers(path=RECEIVERS_PATH):
    try:
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
    except FileNotFoundError:
        print_warning(f"Receiver file not found: {path}. Receiver and IPP overlays will be skipped.")
        return []


def empty_traj_context():
    return {"lat": None, "lon": None, "lat_minute": None, "lon_minute": None, "lat_map": None, "lon_map": None, "ipps": []}


def empty_geodetic_traj_context():
    return {"lat": None, "lon": None, "lat_minute": None, "lon_minute": None, "lat_map": None, "lon_map": None}


def load_single_trajectory_context(traj_path, map_time, color, receivers, plot_ipps, rocket_label):
    try:
        lat, lon, latm, lonm, _lata, _lona, lat_map, lon_map = load_traj(str(traj_path), map_time=map_time, color=color)
    except FileNotFoundError:
        print_warning(f"{rocket_label} trajectory file not found: {traj_path}. Trajectory overlay will be skipped.")
        return empty_traj_context()

    ipps = []
    if plot_ipps:
        try:
            rocket_geo = lookup_traj_geodetic_position(build_traj_lookup(str(traj_path), color=color), map_time)
            ipps = compute_receiver_ipps(receivers, rocket_geo, mapped_apex_height(color))
        except FileNotFoundError:
            print_warning(f"{rocket_label} trajectory file not found while computing IPPs: {traj_path}. IPP overlay will be skipped.")
        except ValueError as exc:
            print_warning(f"Could not compute {rocket_label} IPPs from {Path(traj_path).name}: {exc}")

    return {"lat": lat, "lon": lon, "lat_minute": latm, "lon_minute": lonm, "lat_map": lat_map, "lon_map": lon_map, "ipps": ipps}


def load_single_geodetic_trajectory_context(traj_path, map_time, rocket_label):
    try:
        utc_times, _flight_times, lats, lons, _alts = load_traj_records(str(traj_path))
    except FileNotFoundError:
        print_warning(f"{rocket_label} trajectory file not found: {traj_path}. Geodetic trajectory overlay will be skipped.")
        return empty_geodetic_traj_context()

    idx = fixed_utc_minute_marker_indices(utc_times, second_of_minute=trajectory_marker_second(str(traj_path)))

    lat_map = None
    lon_map = None
    if map_time is not None and utc_times.size > 0:
        map_sec = hhmmss_fractional_to_seconds(map_time)
        if utc_times[0] <= map_sec <= utc_times[-1]:
            sample_idx = int(np.argmin(np.abs(utc_times - map_sec)))
            lat_map = float(lats[sample_idx])
            lon_map = float(lons[sample_idx])

    return {
        "lat": lats,
        "lon": lons,
        "lat_minute": lats[idx].squeeze() if idx.size > 0 else None,
        "lon_minute": lons[idx].squeeze() if idx.size > 0 else None,
        "lat_map": lat_map,
        "lon_map": lon_map,
    }


def safe_launch_start_from_traj(traj_path, rocket_label):
    try:
        return get_launch_start_from_traj_csv(str(traj_path))
    except FileNotFoundError:
        print_warning(f"{rocket_label} trajectory file not found: {traj_path}. Time-since-launch label will be omitted.")
        return None
    except ValueError as exc:
        print_warning(f"Could not derive launch time from {Path(traj_path).name}: {exc}")
        return None


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
    left_tplus = format_time_since_launch(time_str, safe_launch_start_from_traj(GNEISS_LEFT_TRAJECTORY_PATH, "36.397"))
    right_tplus = format_time_since_launch(time_str, safe_launch_start_from_traj(GNEISS_RIGHT_TRAJECTORY_PATH, "36.398"))
    tplus_parts = []
    if left_tplus is not None:
        tplus_parts.append(f"36.397 {left_tplus}")
    if right_tplus is not None:
        tplus_parts.append(f"36.398 {right_tplus}")
    if not tplus_parts:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}"
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}\n" + " | ".join(tplus_parts)


def load_trajectory_context(map_time, color, receivers, plot_ipps):
    return {
        "left": load_single_trajectory_context(GNEISS_LEFT_TRAJECTORY_PATH, map_time, color, receivers, plot_ipps, "36.397"),
        "right": load_single_trajectory_context(GNEISS_RIGHT_TRAJECTORY_PATH, map_time, color, receivers, plot_ipps, "36.398"),
    }



def load_geodetic_trajectory_context(map_time):
    return {
        "left": load_single_geodetic_trajectory_context(GNEISS_LEFT_TRAJECTORY_PATH, map_time, "36.397"),
        "right": load_single_geodetic_trajectory_context(GNEISS_RIGHT_TRAJECTORY_PATH, map_time, "36.398"),
    }


def prepare_image_layers(skymaps, imgs, colorbar_scale, norm_limits=None, shared_norm=True):
    side_images = {}
    main_images = {}
    norm_pool = []
    site_limits = {}
    site_norms = {}
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
        if colorbar_scale == "log":
            site_vmin, site_vmax = compute_log_image_limits([vals]) if vals.size > 0 else (None, None)
        else:
            site_vmin, site_vmax = compute_linear_image_limits([vals]) if vals.size > 0 else (None, None)
        site_limits[site] = (site_vmin, site_vmax)
        site_norms[site] = mpl.colors.LogNorm(vmin=site_vmin, vmax=site_vmax) if colorbar_scale == "log" and site_vmin is not None and site_vmax is not None else None

    if shared_norm and norm_limits is not None and norm_limits[0] is not None and norm_limits[1] is not None:
        shared_vmin, shared_vmax = norm_limits
    elif shared_norm and colorbar_scale == "log":
        shared_vmin, shared_vmax = compute_log_image_limits(norm_pool)
    elif shared_norm:
        shared_vmin, shared_vmax = compute_linear_image_limits(norm_pool)
    else:
        shared_vmin, shared_vmax = None, None
    shared_image_norm = mpl.colors.LogNorm(vmin=shared_vmin, vmax=shared_vmax) if shared_norm and colorbar_scale == "log" and shared_vmin is not None and shared_vmax is not None else None
    return side_images, main_images, site_limits, site_norms, shared_vmin, shared_vmax, shared_image_norm



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


def draw_images(ax, ax1, skymaps, imgs, side_images, main_images, image_cmap, shared_vmin, shared_vmax, shared_image_norm, site_limits, site_norms, shared_norm, axtrans, axtrans1):
    im_handle = None
    for site in imgs.keys():
        main_img = main_images[site]
        side_img = side_images[site]
        vmin, vmax = site_limits[site] if not shared_norm else (shared_vmin, shared_vmax)
        image_norm = site_norms[site] if not shared_norm else shared_image_norm
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
    if left["lat"] is not None and left["lon"] is not None:
        ax.plot(left["lon"], left["lat"], color="red", label="GNEISS trajectory", zorder=7, transform=axtrans)
    if left["lat_minute"] is not None and left["lon_minute"] is not None:
        ax.scatter(left["lon_minute"], left["lat_minute"], color="red", s=15, zorder=7, transform=axtrans)
    if right["lat"] is not None and right["lon"] is not None:
        ax.plot(right["lon"], right["lat"], color="red", zorder=7, transform=axtrans)
    if right["lat_minute"] is not None and right["lon_minute"] is not None:
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


def draw_geodetic_trajectory(ax, geodetic_traj_ctx, axtrans):
    left = geodetic_traj_ctx["left"]
    right = geodetic_traj_ctx["right"]
    if left["lat"] is not None and left["lon"] is not None:
        ax.plot(left["lon"], left["lat"], color="blue", label="GNEISS geodetic trajectory", transform=axtrans, zorder=6)
    if left["lat_minute"] is not None and left["lon_minute"] is not None:
        ax.scatter(left["lon_minute"], left["lat_minute"], color="blue", s=15, transform=axtrans, zorder=6)
    if right["lat"] is not None and right["lon"] is not None:
        ax.plot(right["lon"], right["lat"], color="blue", transform=axtrans, zorder=6)
    if right["lat_minute"] is not None and right["lon_minute"] is not None:
        ax.scatter(right["lon_minute"], right["lat_minute"], color="blue", s=15, transform=axtrans, zorder=6)
    if left["lat_map"] is not None and left["lon_map"] is not None:
        ax.scatter(left["lon_map"], left["lat_map"], color="blue", s=40, marker="o", transform=axtrans, zorder=7)
    if right["lat_map"] is not None and right["lon_map"] is not None:
        ax.scatter(right["lon_map"], right["lat_map"], color="blue", s=40, marker="o", transform=axtrans, zorder=7)


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


def finalize_plot(ax, fig, gs, im_handle, color, label_str, output_path, default_with_args, default_without_args, brightness_markers=None, shared_norm=True):
    ax.text(0.99, 0.01, label_str, transform=ax.transAxes, fontsize=12, color="w", ha="right", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))
    ax.set_title(f"Mapped ASIs and GNEISS trajectory ({color} channel)")
    ax.legend(loc="upper right")
    if shared_norm:
        cax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(im_handle, cax=cax, orientation="vertical")
        cbar.set_label(channel_label(color))
        for tag, bright in brightness_markers or []:
            y = np.clip(bright["percentile"] / 100.0, 0.0, 1.0)
            cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="black", linewidth=4.0, zorder=1000, solid_capstyle="butt", clip_on=False)
            cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="white", linewidth=2.2, zorder=1001, solid_capstyle="butt", clip_on=False)
            cbar.ax.text(-0.05, y, f"{tag}: {bright['site']} P{bright['percentile']:.1f}", transform=cbar.ax.transAxes, color="black", fontsize=8, va="center", ha="right", clip_on=False)
    else:
        ax.text(0.01, 0.01, "Per-site normalization", transform=ax.transAxes, fontsize=10, color="w", ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))

    args = get_plot_call_args()
    plt.tight_layout()
    if output_path is None:
        if args is not None:
            output_path = default_with_args(args)
        else:
            output_path = default_without_args
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")


def plot_map(skymaps, imgs, pfisr, output_path=None, map_time=None, bounds=None, color="green", imgs_raw=None, norm_limits=None, colorbar_scale="linear", colorbar_color="viridis", apex=None, plot_receivers=False, plot_ipps=False, pretty=False, plot_geodetic_traj=False, shared_norm=True):
    receivers = load_receivers() if (plot_receivers or plot_ipps) else []
    if pretty:
        fig, gs, ax, ax1, axt, axt1 = setup_pretty_axes(imgs, bounds, apex)
    else:
        fig, gs, ax, ax1, axt, axt1 = setup_fast_axes(imgs, bounds)
    side_images, main_images, site_limits, site_norms, shared_vmin, shared_vmax, shared_image_norm = prepare_image_layers(
        skymaps,
        imgs,
        colorbar_scale,
        norm_limits=norm_limits,
        shared_norm=shared_norm,
    )
    image_cmap = choose_image_cmap(colorbar_color, color)
    im_handle = draw_images(
        ax,
        ax1,
        skymaps,
        imgs,
        side_images,
        main_images,
        image_cmap,
        shared_vmin,
        shared_vmax,
        shared_image_norm,
        site_limits,
        site_norms,
        shared_norm,
        axt,
        axt1,
    )
    traj_ctx = load_trajectory_context(map_time, color, receivers, plot_ipps)
    draw_trajectory_and_ipps(ax, traj_ctx, axt)
    if plot_geodetic_traj:
        draw_geodetic_trajectory(ax, load_geodetic_trajectory_context(map_time), axt)
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
        shared_norm=shared_norm,
    )
