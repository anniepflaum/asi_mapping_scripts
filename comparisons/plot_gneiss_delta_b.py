#!/usr/bin/env python3
"""Plot GNEISS east/north delta-B with green brightness, plus filtered overlays.

Run from the repository root:
    python comparisons/plot_gneiss_delta_b.py
or:
    python -m comparisons.plot_gneiss_delta_b

Plot 3 contains only filtered magnitude, east, and up delta-B (no brightness).
CSV columns are time, magnitude, east, north, up, as in Roger's exports.
Times are converted from each file's attitude t0 header to the shared TG epoch.
Values are never refiltered or clipped; y limits only restrict the displayed range.
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import sys

# Direct execution places comparisons/, rather than the repository, on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.missions import rocket_launch_datetime
from core.paths import DATA_ROOT


DATA_DIR = DATA_ROOT / "raw" / "rocket" / "gneiss-delta-b"
BRIGHTNESS = DATA_ROOT / "processed" / "gneiss" / "stackplot" / "brightness_vs_time_20260210_101900_102848_step0p05.h5"
TG = rocket_launch_datetime("397")
COMPONENTS = {"Magnitude": 1, "East": 2, "North": 3, "Up": 4}


def load_delta_b(path, rocket):
    """Read independent raw/filtered time axes and retain every sample."""
    with path.open(encoding="utf-8") as stream:
        header = stream.readline()
    match = re.search(r"Time since (397|398) t0 \(s\):\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", header)
    if not match or match[1] != rocket:
        raise ValueError(f"Missing or mismatched flight/t0 header in {path}")
    fields = header.split(",")[1:]
    if len(fields) != 4 or any(
        not field.strip().startswith(name)
        for field, name in zip(fields, ("B_mag", "B_E", "B_N", "B_U"))
    ):
        raise ValueError(f"Unexpected magnetic column order in {path}")
    origin = dt.datetime.combine(TG.date(), dt.time.fromisoformat(match[2]))
    offset = (origin - TG).total_seconds()
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] != 5 or not np.all(np.isfinite(data[:, 0])):
        raise ValueError(f"Expected finite times and five columns in {path}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"Times must increase in {path}")
    data[:, 0] += offset
    print(f"{path.name}: add {offset:.4f} s to attitude time for TG")
    return data


def load_brightness(path, rocket, altitude):
    with h5py.File(path, "r") as h5:
        times = np.asarray(h5["time_since_tg_s"], dtype=float)
        dataset = h5[f"rockets/{rocket}/brightness/{altitude:g}_km"]
        values = np.asarray(dataset, dtype=float)
        units = dataset.attrs.get("units", h5.attrs.get("brightness_units", "Rayleighs"))
    if times.ndim != 1 or times.shape != values.shape:
        raise ValueError(f"Brightness time/value shape mismatch in {path}")
    if isinstance(units, bytes):
        units = units.decode()
    return times, values, units


def magnetic_axis(ax, component, limit):
    ax.axhline(0, color="black", linewidth=0.8, zorder=5)
    ax.set_ylabel(f"{component} ΔB (nT)")
    ax.set_ylim(-limit, limit)
    ax.legend(loc="upper left", fontsize=8)


def finish(fig, axes, title, sources, output, args):
    fig.suptitle(title + "\n" + "\n".join(sources), fontsize=10)
    axes[-1].set_xlabel(f"Time since TG (s); TG = {TG:%Y-%m-%d %H:%M:%S} UTC")
    if args.xlim:
        axes[-1].set_xlim(*args.xlim)
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for rocket in ("397", "398"):
        parser.add_argument(f"--raw-{rocket}", type=Path, default=DATA_DIR / f"deltaB_{rocket}_oct23.csv")
        parser.add_argument(f"--filtered-{rocket}", type=Path, default=DATA_DIR / f"deltaB_filt_{rocket}_oct23.csv")
    parser.add_argument("--brightness-h5", type=Path, default=BRIGHTNESS)
    parser.add_argument("--altitude-km", type=float, default=110)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("START_TG", "END_TG"))
    parser.add_argument("--individual-limit", type=float, default=1500, help="Symmetric magnetic y limit for plots 1/2 (nT)")
    parser.add_argument("--filtered-limit", type=float, default=500, help="Symmetric magnetic y limit for plot 3 (nT)")
    parser.add_argument("--filtered-only", action="store_true", help="Generate only plot 3; no brightness or raw files needed")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if min(args.individual_limit, args.filtered_limit, args.altitude_km, args.dpi) <= 0:
        parser.error("Limits, altitude, and dpi must be positive")
    if args.xlim and args.xlim[0] >= args.xlim[1]:
        parser.error("--xlim must be increasing")
    paths = {r: getattr(args, f"filtered_{r}") for r in ("397", "398")}
    required = list(paths.values())
    if not args.filtered_only:
        required += [args.brightness_h5, args.raw_397, args.raw_398]
    for path in required:
        if not path.is_file():
            parser.error(f"Input not found: {path}; supply its path using the corresponding CLI option")
    filtered = {r: load_delta_b(p, r) for r, p in paths.items()}
    if not args.filtered_only:
        for rocket in ("397", "398"):
            raw_path = getattr(args, f"raw_{rocket}")
            raw = load_delta_b(raw_path, rocket)
            smooth = filtered[rocket]
            times, values, units = load_brightness(args.brightness_h5, rocket, args.altitude_km)
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9), layout="constrained")
            for ax, component in ((axes[0], "East"), (axes[2], "North")):
                col = COMPONENTS[component]
                ax.plot(raw[:, 0], raw[:, col], color="tab:blue", lw=0.6, label="Raw")
                ax.plot(smooth[:, 0], smooth[:, col], color="tab:orange", lw=1.3, label="Filtered")
                magnetic_axis(ax, component, args.individual_limit)
            axes[1].plot(times, values, color="tab:green", lw=1)
            axes[1].set_ylabel(f"Green brightness\n({units})")
            finish(fig, axes, f"GNEISS 36.{rocket}: east / green brightness ({args.altitude_km:g} km) / north",
                   [raw_path.name, paths[rocket].name, args.brightness_h5.name],
                   args.output_dir / f"GNEISS_{rocket}_deltaB_green_brightness.png", args)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9), layout="constrained")
    for ax, component in zip(axes, ("Magnitude", "East", "Up")):
        for rocket, color in (("397", "tab:blue"), ("398", "tab:orange")):
            data = filtered[rocket]
            ax.plot(data[:, 0], data[:, COMPONENTS[component]], color=color, lw=1, label=f"36.{rocket} filtered")
        magnetic_axis(ax, component, args.filtered_limit)
    finish(fig, axes, "GNEISS 36.397 and 36.398: filtered ΔB",
           [p.name for p in paths.values()], args.output_dir / "GNEISS_397_398_deltaB_filtered.png", args)


if __name__ == "__main__":
    main()
