import numpy as np


def sample_raw_brightness_at_latlon(site, lat0, lon0, skymaps, imgs_raw):
    """
    Sample raw image brightness at (lat0, lon0) using the mean of the 25
    closest valid pixels for a site.
    Returns dict with brightness, percentile, and nearest-pixel metadata, or None if unavailable.
    """
    if site not in skymaps or site not in imgs_raw:
        return None
    lat_grid = skymaps[site]["lat"]
    lon_grid = skymaps[site]["lon"]
    raw_img = imgs_raw[site]
    valid = (~skymaps[site]["mask"]) & np.isfinite(lat_grid) & np.isfinite(lon_grid) & np.isfinite(raw_img)
    for mask in skymaps[site].get("extra_masks", {}).values():
        valid &= ~mask
    if not np.any(valid):
        return None
    d2 = (lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2
    valid_flat = valid.ravel()
    d2_flat = d2.ravel()
    valid_idx = np.flatnonzero(valid_flat)
    if valid_idx.size == 0:
        return None
    k = min(25, valid_idx.size)
    nearest_order = np.argpartition(d2_flat[valid_idx], k - 1)[:k]
    nearest_idx = valid_idx[nearest_order]
    nearest_d2 = d2_flat[nearest_idx]
    raw_flat = raw_img.ravel()
    raw_val = float(np.mean(raw_flat[nearest_idx]))
    valid_vals = raw_img[valid]
    percentile = 100.0 * float(np.mean(valid_vals <= raw_val))
    closest_idx = int(nearest_idx[np.argmin(nearest_d2)])
    rr, cc = np.unravel_index(closest_idx, d2.shape)
    return {
        "site": site,
        "raw_brightness": raw_val,
        "percentile": percentile,
        "distance_deg": float(np.mean(np.sqrt(nearest_d2))),
        "n_pixels": int(k),
        "row": int(rr),
        "col": int(cc),
    }


def best_rocket_brightness(lat0, lon0, skymaps, imgs_raw):
    """
    Return nearest-pixel raw brightness info across all available sites for a rocket location.
    """
    samples = []
    for site in imgs_raw.keys():
        sample = sample_raw_brightness_at_latlon(site, lat0, lon0, skymaps, imgs_raw)
        if sample is not None:
            samples.append(sample)
    if not samples:
        return None
    return min(samples, key=lambda item: item["distance_deg"])
