#!/usr/bin/env python3
"""Compare IPP brightness series against receiver_data .mat series."""

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

from core.paths import GNEISS_LEFT_TRAJECTORY_PATH, GNEISS_RIGHT_TRAJECTORY_PATH, mission_output_dir
from core.traj_utils import get_launch_start_from_traj_csv
from core.time_utils import hhmmss_fractional_to_seconds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ipp-csv",
        default=str(mission_output_dir("GNEISS", color="green") / "ipps_brightness_series_101900_102900_step0p05.csv"),
        help="Path to the IPP brightness CSV",
    )
    ap.add_argument(
        "--receiver-dir",
        default="../receiver_data",
        help="Directory containing receiver .mat files under receiver_data/{rocket}/",
    )
    ap.add_argument(
        "--output-dir",
        default=str(mission_output_dir("GNEISS", color="green") / "faraday_ipp_comparisons"),
        help="Directory for output comparison plots",
    )
    return ap.parse_args()


def load_ipp_series(csv_path, rocket, receiver, launch_start):
    field = f"{rocket}_{receiver}_ipp_brightness"
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
    if "flighttime" not in data:
        raise KeyError(f"{mat_path} missing flighttime")
    flighttime = np.asarray(data["flighttime"], dtype=float).reshape(-1)
    if "faraday_minus_smoothed" in data:
        faradayangle = np.asarray(data["faraday_minus_smoothed"], dtype=float).reshape(-1)
    elif "smoothed_faraday" in data:
        faradayangle = np.asarray(data["smoothed_faraday"], dtype=float).reshape(-1)
    elif "faradayangle" in data:
        faradayangle = np.abs(np.asarray(data["faradayangle"], dtype=float).reshape(-1))
    else:
        raise KeyError(f"{mat_path} missing faradayangle/smoothed_faraday/faraday_minus_smoothed")
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


def parse_receiver_mat_path(mat_path):
    rocket = mat_path.parent.name
    stem_parts = mat_path.stem.upper().split("_")
    if len(stem_parts) >= 2 and stem_parts[-1] == rocket:
        receiver = "_".join(stem_parts[:-1])
    else:
        receiver = stem_parts[0]
    return rocket, receiver


def get_launch_start_by_rocket(rocket):
    traj_paths = {
        "397": GNEISS_LEFT_TRAJECTORY_PATH,
        "398": GNEISS_RIGHT_TRAJECTORY_PATH,
    }
    if rocket not in traj_paths:
        raise KeyError(f"Unsupported rocket for launch timing: {rocket}")
    return get_launch_start_from_traj_csv(str(traj_paths[rocket]))


def select_mat_files(receiver_dir):
    mat_files = sorted(path for path in receiver_dir.glob("*/*.mat") if path.is_file())
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {receiver_dir}")

    selected = {}
    for mat_path in mat_files:
        rocket, receiver = parse_receiver_mat_path(mat_path)
        key = (rocket, receiver)
        current = selected.get(key)
        is_minus_smooth = mat_path.stem.upper().endswith("_MINUS_SMOOTH")
        current_is_minus_smooth = current is not None and current.stem.upper().endswith("_MINUS_SMOOTH")
        if current is None or (is_minus_smooth and not current_is_minus_smooth):
            selected[key] = mat_path
    return sorted(selected.values())


def make_plot(rocket, receiver, ipp_t, ipp_y, rx_t, rx_y, output_path):
    ipp_mask = np.asarray(ipp_t, dtype=float) <= 490.0
    ipp_for_scale = np.asarray(ipp_y, dtype=float)[ipp_mask]
    ipp_max = positive_max(ipp_for_scale)
    rx_scaled, scale = scale_to_match(rx_y, ipp_max)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(ipp_t, ipp_y, linewidth=1.8, label=f"IPP {rocket} {receiver} brightness")
    ax.plot(rx_t, rx_scaled, linewidth=1.4, label=f"{receiver} {rocket} Faraday angle")
    ax.set_xlabel("Flight Time (s)")
    ax.set_ylabel("Normalized Amplitude (scaled for comparison)")
    ax.set_title(f"{receiver} vs {rocket} IPP")
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

    mat_files = select_mat_files(receiver_dir)

    for mat_path in mat_files:
        rocket, receiver = parse_receiver_mat_path(mat_path)
        launch_start = get_launch_start_by_rocket(rocket)
        ipp_t, ipp_y = load_ipp_series(csv_path, rocket, receiver, launch_start)
        rx_t, rx_y = load_receiver_series(mat_path)
        if ipp_y.size == 0:
            print(f"{receiver} {rocket}: no IPP samples in CSV, skipping")
            continue
        output_path = output_dir / f"{receiver}_{rocket}_comparison.png"
        make_plot(rocket, receiver, ipp_t, ipp_y, rx_t, rx_y, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
