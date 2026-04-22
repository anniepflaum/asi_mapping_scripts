#!/usr/bin/env python3
"""Compare 397 IPP brightness series against receiver_data .mat series."""

import argparse
import csv
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from core.paths import GNEISS_LEFT_TRAJECTORY_PATH
from core.traj_utils import get_launch_start_from_traj_csv
from core.time_utils import hhmmss_fractional_to_seconds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ipp-csv",
        default="../mapped/green/ipps_brightness_series_101900_102900_step0p05.csv",
        help="Path to the IPP brightness CSV",
    )
    ap.add_argument(
        "--receiver-dir",
        default="../receiver_data",
        help="Directory containing receiver .mat files",
    )
    ap.add_argument(
        "--output-dir",
        default="./receiver_397_comparisons",
        help="Directory for output comparison plots",
    )
    return ap.parse_args()


def load_ipp_series(csv_path, receiver, launch_start):
    field = f"397_{receiver}_ipp_brightness"
    launch_sec = hhmmss_fractional_to_seconds(launch_start)
    times = []
    brightness = []
    with open(csv_path, newline="", encoding="utf-8") as fd:
        reader = csv.DictReader(fd)
        if field not in reader.fieldnames:
            raise KeyError(f"Missing CSV column: {field}")
        for row in reader:
            value = row.get(field, "")
            if not value:
                continue
            sample_dt = dt.datetime.fromisoformat(row["time"])
            sample_sec = sample_dt.hour * 3600.0 + sample_dt.minute * 60.0 + sample_dt.second + sample_dt.microsecond / 1e6
            times.append(sample_sec - launch_sec)
            brightness.append(float(value))
    return np.asarray(times, dtype=float), np.asarray(brightness, dtype=float)


def load_receiver_series(mat_path):
    data = loadmat(mat_path, squeeze_me=True)
    if "flighttime" not in data or "faradayangle" not in data:
        raise KeyError(f"{mat_path} missing flighttime/faradayangle")
    flighttime = np.asarray(data["flighttime"], dtype=float).reshape(-1)
    faradayangle = np.abs(np.asarray(data["faradayangle"], dtype=float).reshape(-1))
    return flighttime, faradayangle


def positive_values(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return finite


def positive_max(values):
    finite = positive_values(values)
    if finite.size == 0:
        return None
    return float(np.nanmax(finite))


def scale_to_match(source, target_max):
    source_max = positive_max(source)
    if source_max is None or target_max is None or target_max <= 0:
        return source, 1.0
    return np.asarray(source, dtype=float) * (target_max / source_max), target_max / source_max


def make_plot(receiver, ipp_t, ipp_y, rx_t, rx_y, output_path):
    ipp_mask = np.asarray(ipp_t, dtype=float) <= 490.0
    ipp_for_scale = np.asarray(ipp_y, dtype=float)[ipp_mask]
    ipp_max = positive_max(ipp_for_scale)
    rx_scaled, scale = scale_to_match(rx_y, ipp_max)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(ipp_t, ipp_y, linewidth=1.8, label=f"IPP 397 {receiver} brightness")
    ax.plot(rx_t, rx_scaled, linewidth=1.4, label=f"{receiver} Faraday angle")
    ax.set_xlabel("Flight Time (s)")
    ax.set_ylabel("Normalized Amplitude (scaled for comparison)")
    ax.set_title(f"{receiver} vs 397 IPP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ipp_for_limits = np.asarray(ipp_y, dtype=float)[ipp_mask]
    finite = np.concatenate(
        [
            ipp_for_limits[np.isfinite(ipp_for_limits)],
            np.asarray(rx_scaled, dtype=float)[np.isfinite(rx_scaled)],
        ]
    )
    if finite.size > 0:
        ymin = float(np.nanmin(finite))
        ymax = float(np.nanmax(finite))
        if ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = Path(args.ipp_csv)
    receiver_dir = Path(args.receiver_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_start = get_launch_start_from_traj_csv(str(GNEISS_LEFT_TRAJECTORY_PATH))
    mat_files = sorted(path for path in receiver_dir.glob("*.mat") if path.is_file())
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {receiver_dir}")

    for mat_path in mat_files:
        receiver = mat_path.stem.upper()
        ipp_t, ipp_y = load_ipp_series(csv_path, receiver, launch_start)
        rx_t, rx_y = load_receiver_series(mat_path)
        if ipp_y.size == 0:
            print(f"{receiver}: no 397 IPP samples in CSV, skipping")
            continue
        output_path = output_dir / f"{receiver}_397_comparison.png"
        make_plot(receiver, ipp_t, ipp_y, rx_t, rx_y, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
