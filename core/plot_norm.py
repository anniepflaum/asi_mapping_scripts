import numpy as np

from core.constants import NORMALIZATION_LOWER_PERCENTILE, NORMALIZATION_UPPER_PERCENTILE


def compute_linear_image_limits(norm_pool, upper_percentile=NORMALIZATION_UPPER_PERCENTILE):
    """Return finite vmin/vmax limits suitable for linear image plotting."""
    if not norm_pool:
        return None, None
    all_vals = np.concatenate(norm_pool)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        return None, None
    vmin = float(np.percentile(finite_vals, NORMALIZATION_LOWER_PERCENTILE))
    vmax = float(np.percentile(finite_vals, upper_percentile))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite_vals))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite_vals))
    return vmin, vmax


def compute_log_image_limits(norm_pool, upper_percentile=NORMALIZATION_UPPER_PERCENTILE):
    """Return positive vmin/vmax limits suitable for log-scaled image plotting."""
    if not norm_pool:
        return None, None
    all_vals = np.concatenate(norm_pool)
    pos_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if pos_vals.size == 0:
        return None, None
    vmin = float(np.percentile(pos_vals, NORMALIZATION_LOWER_PERCENTILE))
    vmax = float(np.percentile(pos_vals, upper_percentile))
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = float(np.nanmin(pos_vals))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(pos_vals))
    return vmin, vmax


def choose_image_cmap(colorbar_color, color):
    if colorbar_color == "viridis":
        return "viridis"
    return "Reds" if str(color).lower() == "red" else "Greens"
