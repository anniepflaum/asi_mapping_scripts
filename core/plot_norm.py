import numpy as np

from core.constants import NORMALIZATION_LOWER_PERCENTILE, NORMALIZATION_UPPER_PERCENTILE, REFERENCE_NORMALIZATION_TIME
from core.remote_data import retrieve_image
from core.time_utils import parse_date_and_time
from core.fetch_url import closest_amisr_png_url
from core.tiff_utils import get_site_tiff_candidates, load_best_frame_from_tiffs


def compute_linear_image_limits(norm_pool):
    """Return finite vmin/vmax limits suitable for linear image plotting."""
    if not norm_pool:
        return None, None
    all_vals = np.concatenate(norm_pool)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        return None, None
    vmin = float(np.percentile(finite_vals, NORMALIZATION_LOWER_PERCENTILE))
    vmax = float(np.percentile(finite_vals, NORMALIZATION_UPPER_PERCENTILE))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite_vals))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite_vals))
    return vmin, vmax


def compute_log_image_limits(norm_pool):
    """Return positive vmin/vmax limits suitable for log-scaled image plotting."""
    if not norm_pool:
        return None, None
    all_vals = np.concatenate(norm_pool)
    pos_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if pos_vals.size == 0:
        return None, None
    vmin = float(np.percentile(pos_vals, NORMALIZATION_LOWER_PERCENTILE))
    vmax = float(np.percentile(pos_vals, NORMALIZATION_UPPER_PERCENTILE))
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = float(np.nanmin(pos_vals))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(pos_vals))
    return vmin, vmax


def choose_image_cmap(colorbar_color, color):
    if colorbar_color == "viridis":
        return "viridis"
    return "Reds" if str(color).lower() == "red" else "Greens"


def reference_normalization_time(mission, date, time_str):
    if str(mission).upper() == "GIRAFF":
        if str(date) == "20250202":
            return "071130"
        if str(date) == "20250209":
            return "083600"
        return time_str
    return REFERENCE_NORMALIZATION_TIME


def compute_reference_norm_limits(skymaps, selected_sites, date, ref_time_str, color, frame_interval, colorbar_scale="linear", mission="GNEISS"):
    """
    Compute shared normalization limits from a fixed reference time using
    post-mask main-map pixels across currently selected sites.
    """
    ref_dt = parse_date_and_time(date, ref_time_str)
    ref_imgs = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site not in selected_sites:
            continue
        try:
            tiff_candidates = get_site_tiff_candidates(site, date, color, mission=mission)
            im_raw, _im_display = load_best_frame_from_tiffs(
                site,
                tiff_candidates,
                ref_dt,
                frame_interval=frame_interval,
                color=color,
                verbose=False,
            )
            ref_imgs[site] = im_raw
        except Exception as exc:
            print(f"{site}: reference normalization frame unavailable at {ref_time_str}: {exc}")
    if "PKR" in selected_sites:
        try:
            pkr_lookup_time = ref_dt.strftime("%H%M%S")
            url_pkr = closest_amisr_png_url("PKR", date, pkr_lookup_time, color=color)
            ref_imgs["PKR"] = retrieve_image(url_pkr)
        except Exception as exc:
            print(f"PKR: reference normalization frame unavailable at {ref_time_str}: {exc}")

    norm_pool = []
    for site, img in ref_imgs.items():
        side_img = img.copy()
        side_img[skymaps[site]["mask"]] = np.nan
        main_img = side_img.copy()
        for mask in skymaps[site]["extra_masks"].values():
            main_img[mask] = np.nan
        vals = main_img[np.isfinite(main_img)]
        if vals.size > 0:
            norm_pool.append(vals)

    if not norm_pool:
        print(f"Reference normalization at {ref_time_str} found no valid pixels; falling back to per-frame normalization.")
        return None, None

    if colorbar_scale == "log":
        vmin, vmax = compute_log_image_limits(norm_pool)
        if vmin is None or vmax is None:
            print(f"Reference normalization at {ref_time_str} found no positive pixels; falling back to per-frame normalization.")
            return None, None
    else:
        vmin, vmax = compute_linear_image_limits(norm_pool)
        if vmin is None or vmax is None:
            print(f"Reference normalization at {ref_time_str} found no finite pixels; falling back to per-frame normalization.")
            return None, None
    print(f"Normalization fixed to {ref_time_str}: vmin={vmin:.2f}, vmax={vmax:.2f}")
    return vmin, vmax
