import numpy as np


def local_pixel_radius_deg(valid, lat_grid, lon_grid, closest_idx):
    """Estimate the local mapped-pixel radius around a nearest valid pixel."""
    rows, cols = valid.shape
    rr, cc = np.unravel_index(int(closest_idx), valid.shape)
    lat_c = lat_grid[rr, cc]
    lon_c = lon_grid[rr, cc]
    neighbor_distances = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = rr + dr
        nc = cc + dc
        if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
            neighbor_distances.append(float(np.hypot(lat_grid[nr, nc] - lat_c, lon_grid[nr, nc] - lon_c)))
    if neighbor_distances:
        return 1.5 * max(neighbor_distances)

    return 0.0


def zero_brightness_sample(site, d2, closest_idx=None):
    if closest_idx is None:
        rr = cc = -1
        distance_deg = float("inf")
    else:
        rr, cc = np.unravel_index(int(closest_idx), d2.shape)
        distance_deg = float(np.sqrt(d2[rr, cc]))
    return {
        "site": site,
        "raw_brightness": 0.0,
        "percentile": 0.0,
        "distance_deg": distance_deg,
        "n_pixels": 0,
        "row": int(rr),
        "col": int(cc),
        "outside_footprint": True,
    }


def sample_raw_brightness_at_latlon(site, lat0, lon0, skymaps, imgs_raw):
    """
    Sample raw image brightness at (lat0, lon0) using the mean of the 25
    closest valid pixels for a site.
    Returns zero brightness when the point is outside the mapped ASI footprint.
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
    closest_valid_idx = int(valid_idx[np.argmin(d2_flat[valid_idx])])
    closest_distance = float(np.sqrt(d2_flat[closest_valid_idx]))
    if closest_distance > local_pixel_radius_deg(valid, lat_grid, lon_grid, closest_valid_idx):
        return zero_brightness_sample(site, d2, closest_valid_idx)

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
        "outside_footprint": False,
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
    in_footprint = [sample for sample in samples if not sample.get("outside_footprint", False)]
    if in_footprint:
        return min(in_footprint, key=lambda item: item["distance_deg"])
    return min(samples, key=lambda item: item["distance_deg"])
