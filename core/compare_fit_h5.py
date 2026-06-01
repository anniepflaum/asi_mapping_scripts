#!/usr/bin/env python3
"""Compare two SOK pixel-coordinate H5 files."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import h5py
import matplotlib.pyplot as plt
import numpy as np


STARMAP_DIR = Path("/Users/anniepflaum/asi_mapping/starmaps/VEE/GIRAFF")
REFERENCE_H5_PATH = STARMAP_DIR / "sok_pixelcoords.h5"
CANDIDATE_H5_PATH = Path("/Users/anniepflaum/Downloads/sok_pixelcoords2.h5")
ANGLE_DATASETS = {"Azimuth", "Elevation"}
DATASETS_TO_COMPARE = (
    "Azimuth",
    "Elevation",
    "FootpointLatitude",
    "FootpointLongitude",
    "MagneticLatitude",
    "MagneticLongitude",
)


def wrapped_az_difference(candidate_deg, reference_deg):
    """Return circular azimuth difference in degrees on [-180, 180)."""
    return (candidate_deg - reference_deg + 180.0) % 360.0 - 180.0


def symmetric_limits(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 99.5))
    if not np.isfinite(limit) or limit == 0:
        limit = float(np.nanmax(np.abs(finite))) or 1.0
    return -limit, limit


def load_h5_maps(path):
    with h5py.File(path, "r") as h5:
        data = {}
        for key in DATASETS_TO_COMPARE:
            values = h5[key][()].astype(float)
            if key in ANGLE_DATASETS:
                values = np.rad2deg(values)
            data[key] = values
        data["Mask"] = h5["Mask"][()].astype(bool)
    return data


def dataset_difference(name, candidate_values, reference_values):
    '''if name == "Azimuth":
        return wrapped_az_difference(candidate_values, reference_values)'''
    return candidate_values - reference_values


def load_differences():
    reference = load_h5_maps(REFERENCE_H5_PATH)
    candidate = load_h5_maps(CANDIDATE_H5_PATH)
    mask = reference["Mask"] | candidate["Mask"]
    diffs = {}
    for key in DATASETS_TO_COMPARE:
        if reference[key].shape != candidate[key].shape:
            raise ValueError(f"{key} shape mismatch: {reference[key].shape} vs {candidate[key].shape}")
        diff = dataset_difference(key, candidate[key], reference[key])
        diff = diff.astype(float)
        diff[mask] = np.nan
        diffs[key] = diff
    mask_diff = candidate["Mask"].astype(int) - reference["Mask"].astype(int)
    return diffs, mask_diff, reference, candidate


def north_line_x_by_row(az_deg, mask):
    """Return row -> x coordinate where azimuth wraps across the 0/360 north line."""
    az = np.mod(np.asarray(az_deg, dtype=float), 360.0)
    valid = np.isfinite(az) & ~mask
    left = az[:, :-1]
    right = az[:, 1:]
    edge_valid = valid[:, :-1] & valid[:, 1:]
    wrap_edge = edge_valid & (np.abs(right - left) > 180.0)
    center_x = (az.shape[1] - 1) / 2.0

    line = {}
    for row in np.flatnonzero(np.any(wrap_edge, axis=1)):
        edge_cols = np.flatnonzero(wrap_edge[row])
        edge_x = edge_cols.astype(float) + 0.5
        line[int(row)] = float(edge_x[np.argmin(np.abs(edge_x - center_x))])
    return line


def local_azimuth_scale_deg_per_pixel(az_deg, mask, row, seam_x):
    """Estimate local azimuth degrees per pixel next to a 0/360 seam."""
    az = np.asarray(az_deg, dtype=float)
    valid = np.isfinite(az[row]) & ~mask[row]
    seam_col = int(np.floor(seam_x))
    scale_values = []
    for col in (seam_col - 2, seam_col + 2):
        if col < 0 or col + 1 >= az.shape[1]:
            continue
        if not (valid[col] and valid[col + 1]):
            continue
        delta = wrapped_az_difference(az[row, col + 1], az[row, col])
        if np.isfinite(delta) and abs(delta) < 90.0:
            scale_values.append(delta)
    if not scale_values:
        return np.nan
    return float(np.nanmean(scale_values))


def north_line_vector(line_by_row):
    """Estimate an undirected vector along a north-line seam in pixel coordinates."""
    if len(line_by_row) < 2:
        return None
    rows = np.asarray(sorted(line_by_row), dtype=float)
    xs = np.asarray([line_by_row[int(row)] for row in rows], dtype=float)
    points = np.column_stack((xs, rows))
    points -= np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points, full_matrices=False)
    vec = vh[0]
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0:
        return None
    return vec / norm


def angle_between_undirected_vectors_deg(vec_a, vec_b):
    dot = abs(float(np.dot(vec_a, vec_b)))
    dot = min(max(dot, -1.0), 1.0)
    return float(np.rad2deg(np.arccos(dot)))


def print_north_line_difference(reference, candidate):
    ref_line = north_line_x_by_row(reference["Azimuth"], reference["Mask"])
    cand_line = north_line_x_by_row(candidate["Azimuth"], candidate["Mask"])
    common_rows = sorted(set(ref_line) & set(cand_line))
    if not common_rows:
        print("North line difference: no shared 0/360 azimuth seam rows found")
        return

    ref_vec = north_line_vector(ref_line)
    cand_vec = north_line_vector(cand_line)
    if ref_vec is not None and cand_vec is not None:
        angle = angle_between_undirected_vectors_deg(ref_vec, cand_vec)
        print(f"North line vector angle: {angle:.6f} deg")

    degree_offsets = []
    for row in common_rows:
        dx = cand_line[row] - ref_line[row]
        deg_per_pixel = local_azimuth_scale_deg_per_pixel(
            reference["Azimuth"],
            reference["Mask"],
            row,
            ref_line[row],
        )
        if np.isfinite(deg_per_pixel):
            degree_offsets.append(dx * deg_per_pixel)

    if not degree_offsets:
        print("North line difference: no finite degree-scale estimates found")
        return

    ddeg = np.asarray(degree_offsets, dtype=float)
    print(
        "North line difference (candidate - reference, deg): "
        f"rows={ddeg.size}, "
        f"min={np.min(ddeg):.6f}, max={np.max(ddeg):.6f}, "
        f"mean={np.mean(ddeg):.6f}, median={np.median(ddeg):.6f}, "
        f"rms={np.sqrt(np.mean(ddeg ** 2)):.6f}"
    )
    print(
        "North line rows missing: "
        f"reference_only={len(set(ref_line) - set(cand_line))}, "
        f"candidate_only={len(set(cand_line) - set(ref_line))}"
    )


def plot_difference(ax, values, title, label):
    vmin, vmax = symmetric_limits(values)
    im = ax.imshow(values, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)


def plot_map(ax, values, title, label, vmin=None, vmax=None):
    im = ax.imshow(values, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)


def main():
    diffs, mask_diff, reference, candidate = load_differences()

    fig_diff, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, key in zip(axes.ravel(), ("Azimuth", "Elevation")):
        label = "Difference (deg)"
        plot_difference(
            ax,
            diffs[key],
            f"{key}: candidate - reference",
            label,
        )

    print_north_line_difference(reference, candidate)
    print(f"Mask changed pixels: {np.count_nonzero(mask_diff)}")
    plt.show()


if __name__ == "__main__":
    main()
