"""ASI camera calibration and corner-background helpers."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from skimage.restoration import estimate_sigma

from core.constants import FRAME_INTERVAL_SECONDS_GREEN


GREEN_RAYLEIGH_SECONDS_PER_COUNT = {
    "ARV": 64.0,
    "VEE": 35.0,
    "BVR": 57.0,
}
GREEN_EXPOSURE_TIME_S = FRAME_INTERVAL_SECONDS_GREEN
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


def green_calibration_factor(site):
    return GREEN_RAYLEIGH_SECONDS_PER_COUNT.get(str(site).upper())


def calibrate_green_image(site, im, mask, edge_buffer_px=BACKGROUND_EDGE_BUFFER_PX):
    factor = green_calibration_factor(site)
    if factor is None:
        return None, None
    fit_mask = buffered_corner_mask(mask, edge_buffer_px=edge_buffer_px)
    bg = estimate_corner_background(im, fit_mask)
    dn = np.asarray(im, dtype=np.float32) - float(bg["center"])
    rayleighs = (dn / float(GREEN_EXPOSURE_TIME_S)) * float(factor)
    return rayleighs.astype(np.float32), bg
