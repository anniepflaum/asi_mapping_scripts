from inspect import currentframe
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import magcoordmap as mcm
from core.brightness import best_rocket_brightness
from core.calc_ipp import calc_ipp
from core.constants import NORMALIZATION_UPPER_PERCENTILE
from core.ezie import (
    EZIE_MEM_COLORS,
    EZIE_MEM_MARKERS,
    EZIE_SPACECRAFT_LINESTYLES,
    ezie_position_at_time,
    load_ezie_mem_tracks,
)
from core.missions import mission_output_dir, trajectory_configs, trajectory_display_labels
from core.paths import COAST_LAT_PATH, COAST_LON_PATH
from core.plot_norm import choose_image_cmap, compute_linear_image_limits, compute_log_image_limits
from core.receivers import filter_receivers_for_mission, load_receivers
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


LATLON_GRID_ZORDER = 30


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


def empty_traj_context():
    return {"lat": None, "lon": None, "lat_minute": None, "lon_minute": None, "lat_map": None, "lon_map": None, "ipps": []}


def empty_geodetic_traj_context():
    return {"lat": None, "lon": None, "lat_minute": None, "lon_minute": None, "lat_map": None, "lon_map": None}


def load_single_trajectory_context(traj_path, map_time, color, receivers, plot_ipps, rocket_label):
    try:
        lat, lon, latm, lonm, _lata, _lona, lat_map, lon_map = load_traj(
            str(traj_path),
            map_time=map_time,
            color=color,
        )
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


def draw_site_markers(ax, site_markers, axtrans):
    for marker in site_markers or []:
        lon = marker["lon"]
        lat = marker["lat"]
        label = marker.get("label", "Site")
        zorder = marker.get("zorder", 12)
        ax.scatter(
            [lon],
            [lat],
            marker=marker.get("marker", "^"),
            s=marker.get("size", 70),
            color=marker.get("color", "yellow"),
            edgecolors=marker.get("edgecolor", "black"),
            linewidths=marker.get("linewidth", 0.9),
            zorder=zorder,
            label=label,
            transform=axtrans,
        )
        ax.text(
            lon + marker.get("label_dx", 0.05),
            lat + marker.get("label_dy", 0.03),
            label,
            fontsize=marker.get("fontsize", 8),
            color=marker.get("text_color", "black"),
            zorder=zorder + 1,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.15),
            transform=axtrans,
        )


def compute_receiver_ipps(receivers, rocket_geo, ipp_height_km):
    if (
        not receivers
        or rocket_geo[0] is None
        or rocket_geo[1] is None
        or rocket_geo[2] is None
        or rocket_geo[2] < ipp_height_km
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
    mission = getattr(args, "mission", "GNEISS")
    tplus_parts = []
    for cfg in trajectory_configs(mission, date=date_str):
        tplus = format_time_since_launch(time_str, safe_launch_start_from_traj(cfg.path, cfg.label))
        if tplus is not None:
            tplus_parts.append(f"{cfg.label} {tplus}")
    if not tplus_parts:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}"
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {format_time_label(time_str)}\n" + " | ".join(tplus_parts)


def load_trajectory_context(map_time, color, receivers, plot_ipps, mission="GNEISS", date=None):
    return {
        cfg.key: load_single_trajectory_context(
            cfg.path,
            map_time,
            color,
            receivers,
            plot_ipps,
            cfg.label,
        )
        for cfg in trajectory_configs(mission, date=date)
    }



def load_geodetic_trajectory_context(map_time, mission="GNEISS", date=None):
    return {
        cfg.key: load_single_geodetic_trajectory_context(cfg.path, map_time, cfg.label)
        for cfg in trajectory_configs(mission, date=date)
    }


def prepare_image_layers(skymaps, imgs, colorbar_scale, norm_limits=None, shared_norm=True, upper_percentile=NORMALIZATION_UPPER_PERCENTILE):
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
            site_vmin, site_vmax = compute_log_image_limits([vals], upper_percentile=upper_percentile) if vals.size > 0 else (None, None)
        else:
            site_vmin, site_vmax = compute_linear_image_limits([vals], upper_percentile=upper_percentile) if vals.size > 0 else (None, None)
        site_limits[site] = (site_vmin, site_vmax)
        site_norms[site] = mpl.colors.LogNorm(vmin=site_vmin, vmax=site_vmax) if colorbar_scale == "log" and site_vmin is not None and site_vmax is not None else None

    if shared_norm and norm_limits is not None and norm_limits[0] is not None and norm_limits[1] is not None:
        shared_vmin, shared_vmax = norm_limits
    elif shared_norm and colorbar_scale == "log":
        shared_vmin, shared_vmax = compute_log_image_limits(norm_pool, upper_percentile=upper_percentile)
    elif shared_norm:
        shared_vmin, shared_vmax = compute_linear_image_limits(norm_pool, upper_percentile=upper_percentile)
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
    ax.set_axisbelow(False)
    ax.grid(True, color="0.35", alpha=0.55, linewidth=0.8, zorder=LATLON_GRID_ZORDER)
    ax1 = {}
    axtrans1 = {}
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1])
        axtrans1[site] = ax1[site].transData
        ax1[site].plot(coastlons, coastlats, color="black")
        ax1[site].set_ylim(ymin=lat_min, ymax=lat_max)
        ax1[site].set_xlim(xmin=lon_min, xmax=lon_max)
        ax1[site].set_aspect(2.2)
        ax1[site].set_axisbelow(False)
        ax1[site].grid(True, color="0.35", alpha=0.55, linewidth=0.8, zorder=LATLON_GRID_ZORDER)
        ax1[site].set_title(site)
    return fig, gs, ax, ax1, axtrans, axtrans1


def setup_pretty_axes(imgs, bounds, apex, apex_height):
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
    ax.gridlines(color="0.25", alpha=0.65, linewidth=0.8, zorder=LATLON_GRID_ZORDER)
    mcm.maggridlines(ax, apex=apex, apex_height=apex_height)
    ax1 = {}
    axtrans1 = {}
    for i, site in enumerate(imgs.keys()):
        ax1[site] = fig.add_subplot(gs[i, -1], projection=proj)
        axtrans1[site] = ccrs.PlateCarree()
        ax1[site].coastlines()
        ax1[site].gridlines(color="0.25", alpha=0.65, linewidth=0.8, zorder=LATLON_GRID_ZORDER)
        mcm.maggridlines(ax1[site], apex=apex, apex_height=apex_height)
        ax1[site].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax1[site].set_title(site)
    return fig, gs, ax, ax1, axtrans, axtrans1


def draw_image_pcolor(axis, lon, lat, image, image_cmap, image_norm, vmin, vmax, transform, zorder=3):
    if image_norm is None and vmin is None:
        return axis.pcolor(lon, lat, image, cmap=image_cmap, zorder=zorder, transform=transform)
    if image_norm is None:
        return axis.pcolor(lon, lat, image, cmap=image_cmap, vmin=vmin, vmax=vmax, zorder=zorder, transform=transform)
    return axis.pcolor(lon, lat, image, cmap=image_cmap, norm=image_norm, zorder=zorder, transform=transform)


def draw_image_points(axis, lon, lat, image, image_cmap, image_norm, vmin, vmax, transform, zorder=3):
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(image)
    if not np.any(valid):
        return None
    kwargs = {
        "c": image[valid],
        "cmap": image_cmap,
        "s": 0.25,
        "marker": "s",
        "linewidths": 0,
        "edgecolors": "none",
        "rasterized": True,
        "zorder": zorder,
        "transform": transform,
    }
    if image_norm is None:
        if vmin is not None:
            kwargs.update({"vmin": vmin, "vmax": vmax})
    else:
        kwargs["norm"] = image_norm
    return axis.scatter(lon[valid], lat[valid], **kwargs)


def axis_lon_lat_extent(axis, transform):
    if isinstance(transform, ccrs.CRS):
        return axis.get_extent(crs=ccrs.PlateCarree())
    lon_min, lon_max = axis.get_xlim()
    lat_min, lat_max = axis.get_ylim()
    return lon_min, lon_max, lat_min, lat_max


def regular_grid_for_axis(axis, transform, max_dim=900):
    lon_min, lon_max, lat_min, lat_max = axis_lon_lat_extent(axis, transform)
    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)
    scale = max(lon_span, lat_span)
    n_lon = max(100, int(np.ceil(max_dim * lon_span / scale)))
    n_lat = max(100, int(np.ceil(max_dim * lat_span / scale)))
    grid_lon = np.linspace(lon_min, lon_max, n_lon)
    grid_lat = np.linspace(lat_min, lat_max, n_lat)
    return np.meshgrid(grid_lon, grid_lat)


def draw_image_regrid(axis, lon, lat, image, image_cmap, image_norm, vmin, vmax, transform, zorder=3):
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(image)
    if np.count_nonzero(valid) < 3:
        return None
    grid_lon, grid_lat = regular_grid_for_axis(axis, transform)
    points = np.column_stack((lon[valid], lat[valid]))
    try:
        gridded = griddata(points, image[valid], (grid_lon, grid_lat), method="linear")
    except Exception as exc:
        print_warning(f"Linear image regridding failed ({exc}); falling back to nearest-neighbor regridding.")
        gridded = griddata(points, image[valid], (grid_lon, grid_lat), method="nearest")

    kwargs = {
        "cmap": image_cmap,
        "shading": "auto",
        "zorder": zorder,
        "transform": transform,
    }
    if image_norm is None:
        if vmin is not None:
            kwargs.update({"vmin": vmin, "vmax": vmax})
    else:
        kwargs["norm"] = image_norm
    return axis.pcolormesh(grid_lon, grid_lat, gridded, **kwargs)


def draw_images(ax, ax1, skymaps, imgs, side_images, main_images, image_cmap, shared_vmin, shared_vmax, shared_image_norm, site_limits, site_norms, shared_norm, axtrans, axtrans1, render_mode="pcolor"):
    im_handle = None
    for site in imgs.keys():
        lon = skymaps[site]["lon"]
        lat = skymaps[site]["lat"]
        main_img = main_images[site]
        side_img = side_images[site]
        vmin, vmax = site_limits[site] if not shared_norm else (shared_vmin, shared_vmax)
        image_norm = site_norms[site] if not shared_norm else shared_image_norm
        if render_mode == "points":
            main_handle = draw_image_points(ax, lon, lat, main_img, image_cmap, image_norm, vmin, vmax, axtrans)
            draw_image_points(ax1[site], lon, lat, side_img, image_cmap, image_norm, vmin, vmax, axtrans1[site])
        elif render_mode == "regrid":
            main_handle = draw_image_regrid(ax, lon, lat, main_img, image_cmap, image_norm, vmin, vmax, axtrans)
            draw_image_regrid(ax1[site], lon, lat, side_img, image_cmap, image_norm, vmin, vmax, axtrans1[site])
        else:
            main_handle = draw_image_pcolor(ax, lon, lat, main_img, image_cmap, image_norm, vmin, vmax, axtrans)
            draw_image_pcolor(ax1[site], lon, lat, side_img, image_cmap, image_norm, vmin, vmax, axtrans1[site])
        if main_handle is not None:
            im_handle = main_handle
    return im_handle


def draw_trajectory_and_ipps(ax, traj_ctx, axtrans, mission="GNEISS", date=None):
    labels = trajectory_display_labels(mission, date=date)
    traj_colors = {"left": "tab:blue", "right": "tab:red", "main": "red"}
    position_colors = {"left": "cyan", "right": "orange", "main": "orange"}
    ipp_colors = {"left": "deepskyblue", "right": "magenta", "main": "deepskyblue"}
    for traj_key, traj in traj_ctx.items():
        traj_color = traj_colors.get(traj_key, "red")
        position_color = position_colors.get(traj_key, "orange")
        ipp_color = ipp_colors.get(traj_key, "deepskyblue")
        traj_label = labels.get(traj_key, f"{mission} {traj_key}")
        tag_label = labels.get(f"{traj_key}_tag", traj_label)
        if traj["lat"] is not None and traj["lon"] is not None:
            ax.plot(traj["lon"], traj["lat"], color=traj_color, label=f"{traj_label} trajectory", zorder=7, transform=axtrans)
        if traj["lat_minute"] is not None and traj["lon_minute"] is not None:
            ax.scatter(traj["lon_minute"], traj["lat_minute"], color=traj_color, s=15, zorder=7, transform=axtrans)
        if traj["lat_map"] is not None and traj["lon_map"] is not None:
            ax.scatter(traj["lon_map"], traj["lat_map"], color=position_color, s=50, marker="o", zorder=8, label=f"{traj_label} at map time", transform=axtrans)
        ipps = traj["ipps"]
        if not ipps:
            continue
        ax.scatter([ipp["lon"] for ipp in ipps], [ipp["lat"] for ipp in ipps], marker="x", s=45, color=ipp_color, linewidths=1.4, zorder=11, label=f"{tag_label} IPP", transform=axtrans)
        for ipp in ipps:
            ax.text(ipp["lon"] + 0.1, ipp["lat"] + 0.03, ipp["acronym"], fontsize=7, color=ipp_color, zorder=12, bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.12), transform=axtrans)


def draw_geodetic_trajectory(ax, geodetic_traj_ctx, axtrans, mission="GNEISS"):
    colors = {"left": "tab:blue", "right": "tab:red", "main": "blue"}
    labels = {"left": "36.397 geodetic trajectory", "right": "36.398 geodetic trajectory", "main": f"{mission} geodetic trajectory"}
    for traj_key, traj in geodetic_traj_ctx.items():
        color = colors.get(traj_key, "blue")
        if traj["lat"] is not None and traj["lon"] is not None:
            ax.plot(traj["lon"], traj["lat"], color=color, label=labels.get(traj_key, f"{mission} {traj_key} geodetic trajectory"), transform=axtrans, zorder=6)
        if traj["lat_minute"] is not None and traj["lon_minute"] is not None:
            ax.scatter(traj["lon_minute"], traj["lat_minute"], color=color, s=15, transform=axtrans, zorder=6)
        if traj["lat_map"] is not None and traj["lon_map"] is not None:
            ax.scatter(traj["lon_map"], traj["lat_map"], color=color, s=40, marker="o", transform=axtrans, zorder=7)


def draw_ezie_trajectories(ax, map_time, axtrans):
    try:
        tracks = load_ezie_mem_tracks()
    except FileNotFoundError as exc:
        print_warning(f"EZIE MEM ephemeris file not found: {exc.filename}. EZIE overlay will be skipped.")
        return
    except ValueError as exc:
        print_warning(f"Could not load EZIE MEM ephemerides: {exc}")
        return

    for track in tracks:
        spacecraft = track["spacecraft"]
        mem_index = track["mem_index"]
        color = EZIE_MEM_COLORS.get(mem_index, "tab:gray")
        linestyle = EZIE_SPACECRAFT_LINESTYLES.get(spacecraft, "-")
        marker = EZIE_MEM_MARKERS.get(mem_index, "o")
        label = track["label"]
        ax.plot(
            track["lon"],
            track["lat"],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            alpha=0.9,
            label=f"{label} trajectory",
            zorder=6.5,
            transform=axtrans,
        )
        lat_map, lon_map = ezie_position_at_time(track, map_time)
        if lat_map is not None and lon_map is not None:
            ax.scatter(
                lon_map,
                lat_map,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                s=42,
                marker=marker,
                label=f"{label} at map time",
                zorder=8.5,
                transform=axtrans,
            )


def sample_rocket_brightnesses(traj_ctx, skymaps, imgs_raw):
    brightnesses = {}
    if imgs_raw is not None:
        for traj_key, traj in traj_ctx.items():
            if traj["lat_map"] is None or traj["lon_map"] is None:
                continue
            bright = best_rocket_brightness(traj["lat_map"], traj["lon_map"], skymaps, imgs_raw)
            if bright is not None:
                brightnesses[traj_key] = bright
                print(f"{traj_key} rocket brightness percentile: {bright['site']} P={bright['percentile']:.1f} (nearest {bright['distance_deg']:.3f} deg)")
    return brightnesses


def finalize_plot(ax, fig, gs, im_handle, color, label_str, output_path, default_with_args, default_without_args, brightness_markers=None, shared_norm=True, mission="GNEISS", legend_loc="upper right", map_alt_km=None, colorbar_label=None):
    ax.text(0.99, 0.01, label_str, transform=ax.transAxes, fontsize=12, color="w", ha="right", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))
    altitude_text = f" - {map_alt_km:g}km" if map_alt_km is not None else ""
    ax.set_title(f"Mapped ASIs and {mission} trajectory ({color} channel{altitude_text})")
    ax.legend(loc=legend_loc)
    if im_handle is not None:
        cax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(im_handle, cax=cax, orientation="vertical")
        cbar.set_label(colorbar_label or channel_label(color))
        for tag, bright in brightness_markers or []:
            y = np.clip(bright["percentile"] / 100.0, 0.0, 1.0)
            cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="black", linewidth=4.0, zorder=1000, solid_capstyle="butt", clip_on=False)
            cbar.ax.plot([0.0, 1.0], [y, y], transform=cbar.ax.transAxes, color="white", linewidth=2.2, zorder=1001, solid_capstyle="butt", clip_on=False)
            cbar.ax.text(-0.05, y, f"{tag}: {bright['site']} P{bright['percentile']:.1f}", transform=cbar.ax.transAxes, color="black", fontsize=8, va="center", ha="right", clip_on=False)
    elif im_handle is None:
        ax.text(
            0.01,
            0.01,
            "No valid ASI frames for requested time/sites",
            transform=ax.transAxes,
            fontsize=10,
            color="w",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
        )

    args = get_plot_call_args()
    plt.tight_layout()
    if output_path is None:
        if args is not None:
            output_path = default_with_args(args)
        else:
            output_path = default_without_args
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved mapped image to {output_path}")


def plot_map(skymaps, imgs, pfisr, output_path=None, map_time=None, map_date=None, bounds=None, color="green", imgs_raw=None, norm_limits=None, colorbar_scale="linear", colorbar_color="viridis", apex=None, plot_receivers=False, plot_ipps=False, pretty=False, plot_geodetic_traj=False, shared_norm=True, mission="GNEISS", upper_percentile=NORMALIZATION_UPPER_PERCENTILE, site_markers=None, plot_ezie=False, render_mode="auto", colorbar_label=None):
    receivers = filter_receivers_for_mission(load_receivers(warn=print_warning), mission) if (plot_receivers or plot_ipps) else []
    if pretty:
        fig, gs, ax, ax1, axt, axt1 = setup_pretty_axes(imgs, bounds, apex, mapped_apex_height(color))
    else:
        fig, gs, ax, ax1, axt, axt1 = setup_fast_axes(imgs, bounds)
    side_images, main_images, site_limits, site_norms, shared_vmin, shared_vmax, shared_image_norm = prepare_image_layers(
        skymaps,
        imgs,
        colorbar_scale,
        norm_limits=norm_limits,
        shared_norm=shared_norm,
        upper_percentile=upper_percentile,
    )
    image_cmap = choose_image_cmap(colorbar_color, color)
    resolved_render_mode = "regrid" if render_mode == "auto" and str(color).lower() == "red" else render_mode
    if resolved_render_mode == "auto":
        resolved_render_mode = "pcolor"
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
        render_mode=resolved_render_mode,
    )
    traj_ctx = load_trajectory_context(map_time, color, receivers, plot_ipps, mission=mission, date=map_date)
    draw_trajectory_and_ipps(ax, traj_ctx, axt, mission=mission, date=map_date)
    if plot_geodetic_traj:
        draw_geodetic_trajectory(ax, load_geodetic_trajectory_context(map_time, mission=mission, date=map_date), axt, mission=mission)
    if plot_ezie:
        draw_ezie_trajectories(ax, map_time, axt)
    if plot_receivers:
        draw_receivers(ax, receivers, axt)
    draw_site_markers(ax, site_markers, axt)
    brightnesses = sample_rocket_brightnesses(traj_ctx, skymaps, imgs_raw)
    brightness_markers = []
    if brightnesses:
        marker_labels = trajectory_display_labels(mission, date=map_date)
        for traj_key, bright in brightnesses.items():
            brightness_markers.append((marker_labels.get(f"{traj_key}_tag", traj_key), bright))
    finalize_plot(
        ax,
        fig,
        gs,
        im_handle,
        color,
        build_time_label(get_plot_call_args()),
        output_path,
        lambda args: mission_output_dir(mission, color=args.color, date=args.date) / f"{mission}_launch_science_fast_{args.date}_{sanitize_time_for_filename(args.time)}.png",
        mission_output_dir(mission, color=color) / f"{mission}_launch_science_fast.png",
        brightness_markers=brightness_markers,
        shared_norm=shared_norm,
        mission=mission,
        legend_loc="upper left" if plot_ezie else "upper right",
        map_alt_km=mapped_apex_height(color),
        colorbar_label=colorbar_label,
    )
