"""ASI camera calibration and corner-background helpers."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from skimage.restoration import estimate_sigma

from core.constants import FRAME_INTERVAL_SECONDS_GREEN, FRAME_INTERVAL_SECONDS_RED


GREEN_RAYLEIGH_SECONDS_PER_COUNT = {
    "ARV": 64.0,
    "VEE": 35.0,
    "BVR": 57.0,
}
RED_RAYLEIGH_SECONDS_PER_COUNT = {
    "ARV": 23.0,
    "VEE": 37.0,
    "BVR": 23.0,
}
RAYLEIGH_SECONDS_PER_COUNT_BY_COLOR = {
    "green": GREEN_RAYLEIGH_SECONDS_PER_COUNT,
    "red": RED_RAYLEIGH_SECONDS_PER_COUNT,
}
EXPOSURE_TIME_S_BY_COLOR = {
    "green": FRAME_INTERVAL_SECONDS_GREEN,
    "red": FRAME_INTERVAL_SECONDS_RED,
}
BACKGROUND_EDGE_BUFFER_PX = 80.0


def gauss(x, amp, center, sigma):
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def buffered_corner_mask(mask, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX):
    distances = distance_transform_edt(np.asarray(mask, dtype=bool))
    return np.asarray(mask, dtype=bool) & (distances > float(edge_buffer_px))


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def estimate_corner_background(im, mask):
    imvec = finite_values(np.asarray(im)[mask])
    if imvec.size < 2:
        raise ValueError(f"corner mask produced only {imvec.size} finite pixels")

    data_min = float(np.nanmin(imvec))
    data_max = float(np.nanmax(imvec))
    center_guess = float(np.nanmedian(imvec))
    if data_max <= data_min:
        return {
            "center": center_guess,
            "sigma": 0.0,
            "median": center_guess,
            "fit_failed": True,
            "n_pixels": int(imvec.size),
        }

    bins = np.arange(np.floor(data_min), np.ceil(data_max) + 1)
    if bins.size < 3:
        bins = np.linspace(data_min, data_max + 1.0, 3)
    bincenters = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(imvec, bins=bins)

    sigma_guess = float(estimate_sigma(np.asarray(im)[np.isfinite(im)]))
    if not np.isfinite(sigma_guess) or sigma_guess <= 0:
        sigma_guess = float(np.nanstd(imvec))
    if not np.isfinite(sigma_guess) or sigma_guess <= 0:
        sigma_guess = 1.0

    fit_failed = False
    peak_guess = max(float(np.nanmax(hist)) / 5.0, 1.0)
    try:
        _peak, center, sigma = curve_fit(
            gauss,
            bincenters,
            hist,
            p0=[peak_guess, center_guess, sigma_guess],
            maxfev=10000,
        )[0]
        center = float(center)
        sigma = abs(float(sigma))
    except Exception:
        fit_failed = True
        center = center_guess
        sigma = sigma_guess

    hrange = float(bins[-1] - bins[0])
    if hrange > 0 and (sigma > (hrange / 2.0) or abs(center - center_guess) > (hrange / 4.0)):
        fit_failed = True
        center = center_guess
        sigma = sigma_guess

    return {
        "center": center,
        "sigma": sigma,
        "median": center_guess,
        "fit_failed": fit_failed,
        "n_pixels": int(imvec.size),
    }


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
        "background_method": "buffered_corners",
        "background_edge_buffer_px": BACKGROUND_EDGE_BUFFER_PX,
    }


def calibrate_image(site, im, mask, color, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX):
    factor = calibration_factor(site, color)
    exposure = exposure_time_s(color)
    if exposure is None:
        return None, None
    if factor is None:
        return None, None
    fit_mask = buffered_corner_mask(mask, edge_buffer_px=edge_buffer_px)
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
):
    """Calibrate a frame once, reusing it while a time series selects the same frame."""
    cached = cache.get(site)
    if cached is not None and cached["frame_key"] == frame_key:
        return cached["image"], cached["background"]
    image, background = calibrate_image(site, im, mask, color, edge_buffer_px=edge_buffer_px)
    if image is not None:
        cache[site] = {
            "frame_key": frame_key,
            "image": image,
            "background": background,
        }
    return image, background


def green_calibration_factor(site):
    return calibration_factor(site, "green")


def calibrate_green_image(site, im, mask, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX):
    return calibrate_image(site, im, mask, "green", edge_buffer_px=edge_buffer_px)
