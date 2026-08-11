#!/usr/bin/env python3
"""Compare inversion E0 and Q footprint series with CHIMPS measurements."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.paths import DATA_ROOT, MAPPED_DIR, OUTPUT_ROOT


DEFAULT_FOOTPRINT_H5 = OUTPUT_ROOT / "asispectralinversion" / "gneiss_rg_20260210_101900_102840_step1s_footprints.h5"
DEFAULT_CHIMPS_H5 = DATA_ROOT / "processed" / "gneiss" / "stackplot" / "chimps_397_downgoing_data.h5"
DEFAULT_BRIGHTNESS_H5 = MAPPED_DIR / "green" / "GNEISS" / "brightness_vs_time_20260210_101900_102848_step0p05.h5"
DEFAULT_OUTPUT = Path(__file__).with_name("compare_chimps_spectra_e0_footprint.png")
DEFAULT_Q_OUTPUT = Path(__file__).with_name("compare_chimps_total_counts_q_footprint.png")


def bin_edges(centers):
    """Construct pcolormesh bin edges from monotonically increasing centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("At least two one-dimensional bin centers are required")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("Bin centers must be strictly increasing")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        ([centers[0] - (midpoints[0] - centers[0])],
         midpoints,
         [centers[-1] + (centers[-1] - midpoints[-1])])
    )


def load_chimps(path):
    with h5py.File(path, "r") as h5:
        time_s = np.asarray(h5["time_since_TG_s"], dtype=float)
        energy_eV = np.asarray(h5["energy_eV"], dtype=float)
        log_counts = np.asarray(h5["log10_counts"], dtype=float)
        total_counts = np.asarray(h5["total_counts"], dtype=float)
        metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
    expected_shape = (energy_eV.size, time_s.size)
    if log_counts.shape != expected_shape:
        raise ValueError(
            f"CHIMPS log10_counts has shape {log_counts.shape}; expected {expected_shape}"
        )
    if total_counts.shape != time_s.shape:
        raise ValueError(
            f"CHIMPS total_counts has shape {total_counts.shape}; expected {time_s.shape}"
        )
    return time_s, energy_eV, log_counts, total_counts, metadata


def load_footprint_product(path, rocket, product):
    product_path = f"rockets/{rocket}/products/{product}"
    with h5py.File(path, "r") as h5:
        time_s = np.asarray(h5["time_since_tg_s"], dtype=float)
        values = np.asarray(h5[f"{product_path}/mean"], dtype=float)
        outside = np.asarray(h5[f"{product_path}/outside_footprint"], dtype=bool)
    if not (time_s.shape == values.shape == outside.shape):
        raise ValueError(
            f"Inversion time, {product}, and footprint-mask arrays have different shapes"
        )
    valid = np.isfinite(time_s) & np.isfinite(values) & ~outside
    return time_s[valid], values[valid]


def load_footpoint_brightness(path, rocket, mapped_altitude_km=110):
    dataset_path = f"rockets/{rocket}/brightness/{mapped_altitude_km}_km"
    with h5py.File(path, "r") as h5:
        time_s = np.asarray(h5["time_since_tg_s"], dtype=float)
        brightness = np.asarray(h5[dataset_path], dtype=float)
        units = h5[dataset_path].attrs.get(
            "units", h5.attrs.get("brightness_units", "Rayleighs")
        )
    if time_s.shape != brightness.shape:
        raise ValueError("Brightness time and value arrays have different shapes")
    valid = np.isfinite(time_s) & np.isfinite(brightness)
    return time_s[valid], brightness[valid], str(units)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footprint-h5", type=Path, default=DEFAULT_FOOTPRINT_H5)
    parser.add_argument("--chimps-h5", type=Path, default=DEFAULT_CHIMPS_H5)
    parser.add_argument("--brightness-h5", type=Path, default=DEFAULT_BRIGHTNESS_H5)
    parser.add_argument("--rocket", default="397", help="Rocket group in the footprint file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q-output", type=Path, default=DEFAULT_Q_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    time_s, energy_eV, log_counts, total_counts, metadata = load_chimps(args.chimps_h5)
    e0_time_s, e0_eV = load_footprint_product(
        args.footprint_h5, args.rocket, "characteristic_energy"
    )
    positive_e0 = e0_eV > 0
    e0_time_s, e0_eV = e0_time_s[positive_e0], e0_eV[positive_e0]
    q_time_s, q_mW_m2 = load_footprint_product(
        args.footprint_h5, args.rocket, "energy_flux"
    )

    limits = metadata.get("log10_counts_limits", [0.0, 4.5])
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    mesh = ax.pcolormesh(
        bin_edges(time_s),
        bin_edges(np.log10(energy_eV)),
        log_counts,
        shading="auto",
        cmap="jet",
        vmin=float(limits[0]),
        vmax=float(limits[1]),
    )
    # Leave room between the Q axis and the spectrogram colorbar.
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.08)
    colorbar.set_label("Log10(Counts)")

    e0_line = ax.plot(
        e0_time_s,
        np.log10(e0_eV),
        color="white",
        linewidth=2.5,
        label=r"Inversion $e_0$ at 397 footprint",
        zorder=3,
    )[0]
    # A thin dark underlay keeps the white curve visible in low-count regions.
    ax.plot(e0_time_s, np.log10(e0_eV), color="black", linewidth=4.0, zorder=2)

    spectrum_q_ax = ax.twinx()
    spectrum_q_ax.plot(
        q_time_s,
        q_mW_m2,
        color="white",
        linewidth=3.5,
        zorder=3,
    )
    spectrum_q_line = spectrum_q_ax.plot(
        q_time_s,
        q_mW_m2,
        color="tab:blue",
        linewidth=2.0,
        label=r"Inversion $Q$ at 397 footprint",
        zorder=4,
    )[0]
    spectrum_q_ax.set_ylabel(
        r"Inversion $Q$ (mW m$^{-2}$)", color="tab:blue"
    )
    spectrum_q_ax.tick_params(axis="y", labelcolor="tab:blue")

    ax.set_title(
        r"CHIMPS 36.397 Downgoing Electrons with Inversion $e_0$ and $Q$ Footprints",
        fontweight="bold",
    )
    ax.set_xlabel("Time since TG (s)")
    ax.set_ylabel("Energy (eV)")
    energy_ticks = np.array([300, 500, 1000, 2000, 5000, 10000], dtype=float)
    visible_ticks = energy_ticks[
        (energy_ticks >= energy_eV.min()) & (energy_ticks <= energy_eV.max())
    ]
    ax.set_yticks(np.log10(visible_ticks), [f"{value:g}" for value in visible_ticks])
    ax.set_xlim(time_s.min(), time_s.max())
    ax.set_ylim(*np.log10([energy_eV.min(), energy_eV.max()]))
    ax.legend(
        [e0_line, spectrum_q_line],
        [e0_line.get_label(), spectrum_q_line.get_label()],
        loc="upper right",
        framealpha=0.9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.output}")

    brightness_time_s, brightness, brightness_units = load_footpoint_brightness(
        args.brightness_h5, args.rocket
    )
    fig, counts_ax = plt.subplots(figsize=(12, 6))
    counts_line = counts_ax.plot(
        time_s,
        total_counts,
        color="tab:red",
        linewidth=1.2,
        label="CHIMPS total counts",
    )[0]
    counts_ax.set_xlabel("Time since TG (s)")
    counts_ax.set_ylabel("CHIMPS Total Counts", color="tab:red")
    counts_ax.tick_params(axis="y", labelcolor="tab:red")
    counts_ax.set_xlim(time_s.min(), time_s.max())
    counts_ax.grid(True, alpha=0.25)

    q_ax = counts_ax.twinx()
    q_line = q_ax.plot(
        q_time_s,
        q_mW_m2,
        color="tab:blue",
        linewidth=1.5,
        label=r"Inversion $Q$ at 397 footprint",
    )[0]
    q_ax.set_ylabel(r"Inversion $Q$ (mW m$^{-2}$)", color="tab:blue")
    q_ax.tick_params(axis="y", labelcolor="tab:blue")

    brightness_ax = counts_ax.twinx()
    brightness_ax.spines["right"].set_position(("axes", 1.11))
    brightness_line = brightness_ax.plot(
        brightness_time_s,
        brightness,
        color="tab:green",
        linewidth=1.2,
        alpha=0.9,
        label="397 footpoint brightness (110 km)",
    )[0]
    brightness_ax.set_ylabel(
        f"Footpoint Brightness ({brightness_units})", color="tab:green"
    )
    brightness_ax.tick_params(axis="y", labelcolor="tab:green")
    brightness_ax.set_ylim(0, 75000)
    counts_ax.set_title(
        r"CHIMPS 36.397 Total Counts, Inversion $Q$, and Footpoint Brightness",
        fontweight="bold",
    )
    counts_ax.legend(
        [counts_line, q_line, brightness_line],
        [
            counts_line.get_label(),
            q_line.get_label(),
            brightness_line.get_label(),
        ],
        loc="upper right",
    )
    fig.subplots_adjust(right=0.82)

    args.q_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.q_output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.q_output}")


if __name__ == "__main__":
    main()
