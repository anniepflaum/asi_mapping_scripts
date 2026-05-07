#!/usr/bin/env python3
"""Compare trajectory brightness for several green mapped altitudes."""

import csv
import datetime as dt
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from core.missions import mission_output_dir, rocket_default_date


GREEN_ALTS_KM = (95, 100, 105, 110)
PLOT_CONFIGS = [
    {
        "mission": "GNEISS",
        "output_dir": mission_output_dir("GNEISS", color="green"),
        "filename_prefix": "brightness_vs_time",
        "output_name": "GNEISS_brightness_green_alt_comparison.png",
        "panels": [
            ("397", "left_brightness", "36.397"),
            ("398", "right_brightness", "36.398"),
        ],
    },
    {
        "mission": "GIRAFF",
        "output_dir": mission_output_dir("GIRAFF", color="green", date=rocket_default_date("380")),
        "filename_prefix": "GIRAFF_brightness_vs_time",
        "output_name": "GIRAFF_380_brightness_green_alt_comparison.png",
        "panels": [
            ("380", "main_brightness", "36.380"),
        ],
    },
    {
        "mission": "GIRAFF",
        "output_dir": mission_output_dir("GIRAFF", color="green", date=rocket_default_date("381")),
        "filename_prefix": "GIRAFF_brightness_vs_time",
        "output_name": "GIRAFF_381_brightness_green_alt_comparison.png",
        "panels": [
            ("381", "main_brightness", "36.381"),
        ],
    },
]


def alt_token(alt_km):
    return str(float(alt_km)).replace(".", "p").rstrip("0").rstrip("p")


def csv_sort_key(path):
    match = re.search(r"_(\d{6}(?:p\d+)?)_(\d{6}(?:p\d+)?)_step(\d+(?:p\d+)?)", path.name)
    if not match:
        return (float("inf"), path.name)
    start, end, step = match.groups()
    return (float(step.replace("p", ".")), start, end, path.name)


def find_brightness_csv(output_dir, filename_prefix, alt_km):
    if alt_km == 110:
        explicit = sorted(output_dir.glob(f"{filename_prefix}_*_alt_{alt_token(alt_km)}km.csv"), key=csv_sort_key)
        if explicit:
            return explicit[0]
        candidates = [
            path
            for path in output_dir.glob(f"{filename_prefix}_*.csv")
            if "_alt_" not in path.name and "ipps" not in path.name
        ]
    else:
        candidates = sorted(output_dir.glob(f"{filename_prefix}_*_alt_{alt_token(alt_km)}km.csv"), key=csv_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No brightness CSV found for green altitude {alt_km} km in {output_dir}")
    return sorted(candidates, key=csv_sort_key)[0]


def load_brightness_series(csv_path, brightness_column):
    times = []
    brightness = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if brightness_column not in (reader.fieldnames or []):
            raise KeyError(f"{csv_path} missing {brightness_column}")
        for row in reader:
            value = row.get(brightness_column, "")
            if not value:
                continue
            sample_dt = dt.datetime.fromisoformat(row["time"])
            times.append(sample_dt)
            brightness.append(float(value))
    return times, brightness


def plot_config(config):
    csv_by_alt = {
        alt: find_brightness_csv(config["output_dir"], config["filename_prefix"], alt)
        for alt in GREEN_ALTS_KM
    }

    panels = config["panels"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    colors = {
        95: "tab:blue",
        100: "tab:green",
        105: "tab:orange",
        110: "tab:red",
    }

    all_times = []
    for ax, (rocket, brightness_column, rocket_label) in zip(axes, panels):
        for alt in GREEN_ALTS_KM:
            times, brightness = load_brightness_series(csv_by_alt[alt], brightness_column)
            all_times.extend(times)
            ax.plot(times, brightness, linewidth=1.0, color=colors[alt], label=f"{alt} km")
        ax.set_title(f"{rocket_label} trajectory brightness")
        ax.set_xlabel("UTC time")
        ax.set_ylabel("ASI brightness")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.legend(title="Green altitude")

    if all_times:
        xlim = (min(all_times), max(all_times))
        for ax in axes:
            ax.set_xlim(*xlim)

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = config["output_dir"] / config["output_name"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    for alt, csv_path in csv_by_alt.items():
        print(f"{alt} km: {csv_path}")


def main():
    for config in PLOT_CONFIGS:
        plot_config(config)


if __name__ == "__main__":
    main()
