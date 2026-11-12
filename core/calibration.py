"""ASI camera calibration and corner-background helpers."""

import numpy as np
from asi_shared.background import (
    BACKGROUND_EDGE_BUFFER_PX, BACKGROUND_ELEVATION_CUTOFF_DEG,
    inset_mask, physical_offsky_mask, gauss, finite_values, estimate_corner_background,
)

from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED


GREEN_RAYLEIGH_SECONDS_PER_COUNT = {
    "ARV": 64.0,
    "VEE": 35.0,
    "BVR": 57.0,
}
RED_RAYLEIGH_SECONDS_PER_COUNT = {
    "ARV": 40.0,
    "VEE": 69.5,
    "BVR": 30.2,
}
BLUE_RAYLEIGH_SECONDS_PER_COUNT = {
    "ARV": 35.5,
    "VEE": 17.0,
    "BVR": 1.7,
}
RAYLEIGH_SECONDS_PER_COUNT_BY_COLOR = {
    "green": GREEN_RAYLEIGH_SECONDS_PER_COUNT,
    "red": RED_RAYLEIGH_SECONDS_PER_COUNT,
    "blue": BLUE_RAYLEIGH_SECONDS_PER_COUNT,
}
EXPOSURE_TIME_S_BY_COLOR = {
    "green": FRAME_INTERVAL_SECONDS_GREEN,
    "red": FRAME_INTERVAL_SECONDS_RED,
    "blue": 0.9,  # Exposure duration, distinct from the acquisition cadence.
}


def buffered_corner_mask(mask, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX):
    """Inset an explicitly constructed background region (diagnostic helper)."""
    return inset_mask(mask, edge_buffer_px=edge_buffer_px)


def calibration_factor(site, color):
    return RAYLEIGH_SECONDS_PER_COUNT_BY_COLOR.get(str(color).lower(), {}).get(str(site).upper())


def exposure_time_s(color):
    return EXPOSURE_TIME_S_BY_COLOR.get(str(color).lower())


def partition_calibrated_sites(sites, color):
    """Return requested sites split into calibrated and unsupported lists."""
    supported = [site for site in sites if calibration_factor(site, color) is not None]
    unsupported = [site for site in sites if calibration_factor(site, color) is None]
    return supported, unsupported


def calibration_metadata(sites, color):
    """Describe the physical calibration applied to exported brightness data."""
    return {
        "brightness_units": "Rayleighs",
        "color": str(color).lower(),
        "factors_r_s_per_count": {
            str(site).upper(): calibration_factor(site, color)
            for site in sites
            if calibration_factor(site, color) is not None
        },
        "exposure_time_s": exposure_time_s(color),
        "background_method": "gaussian_histogram_median_fallback",
        "background_elevation_cutoff_deg": BACKGROUND_ELEVATION_CUTOFF_DEG,
        "background_edge_buffer_px": BACKGROUND_EDGE_BUFFER_PX,
    }


def calibrate_image(site, im, mask, color, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX, *, elevation=None):
    factor = calibration_factor(site, color)
    exposure = exposure_time_s(color)
    if exposure is None:
        return None, None
    if factor is None:
        return None, None
    if elevation is None:
        raise ValueError("Native elevation is required for 0-degree background selection")
    fit_mask = physical_offsky_mask(elevation, edge_buffer_px=edge_buffer_px)
    if np.shape(im) != fit_mask.shape:
        raise ValueError("Image/elevation shape mismatch")
    bg = estimate_corner_background(im, fit_mask)
    dn = np.asarray(im, dtype=np.float32) - float(bg["center"])
    rayleighs = (dn / float(exposure)) * float(factor)
    return rayleighs.astype(np.float32), bg


def calibrate_image_cached(
    site,
    im,
    mask,
    color,
    frame_key,
    cache,
    edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX,
    *, elevation=None,
):
    """Calibrate a frame once, reusing it while a time series selects the same frame."""
    policy_key = (float(edge_buffer_px), id(elevation), str(color))
    cached = cache.get(site)
    if cached is not None and cached["frame_key"] == frame_key and cached.get("policy_key") == policy_key:
        return cached["image"], cached["background"]
    image, background = calibrate_image(site, im, mask, color, edge_buffer_px=edge_buffer_px, elevation=elevation)
    if image is not None:
        cache[site] = {
            "frame_key": frame_key,
            "policy_key": policy_key,
            "image": image,
            "background": background,
        }
    return image, background


def green_calibration_factor(site):
    return calibration_factor(site, "green")


def calibrate_green_image(site, im, mask, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX, *, elevation=None):
    return calibrate_image(site, im, mask, "green", edge_buffer_px=edge_buffer_px, elevation=elevation)
