#!/usr/bin/env python3
"""Plot GNEISS E_North despin data against trajectory brightness."""

import csv
import datetime as dt
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.missions import mission_output_dir


DESPIN_DIR = Path("/Users/anniepflaum/asi_mapping/GNEISS_E_field_data")
BRIGHTNESS_CSV = mission_output_dir("GNEISS", color="green") / "brightness_vs_time_101900_102900_step0p05.csv"
OUTPUT_DIR = mission_output_dir("GNEISS", color="green")
ROCKETS = ("397", "398")
ROCKET_BRIGHTNESS_COLUMNS = {
    "397": "left_brightness",
    "398": "right_brightness",
}


def parse_despin_header(csv_path):
    with Path(csv_path).open("r", encoding="utf-8") as f:
        first_line = f.readline()
    rocket_match = re.search(r"\b(397|398)\s+t0\b", first_line, flags=re.IGNORECASE)
    time_match = re.search(r"t0\s*\(s\):\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", first_line, flags=re.IGNORECASE)
    if not rocket_match:
        raise ValueError(f"Could not parse rocket ID from first line of {csv_path}")
    if not time_match:
        raise ValueError(f"Could not parse t0 time from first line of {csv_path}")
    return rocket_match.group(1), dt.time.fromisoformat(time_match.group(1))


def load_despin_enorth(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{csv_path} must have at least 3 columns: time, E_east, E_north")
    return data[:, 0], data[:, 2]


def load_brightness(csv_path, brightness_column):
    times = []
    brightness = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if brightness_column not in fieldnames:
            brightness_column = brightness_column.replace("left_", "397_").replace("right_", "398_")
        if brightness_column not in fieldnames:
            raise KeyError(f"Missing brightness column {brightness_column!r} in {csv_path}")
        for row in reader:
            value = row.get(brightness_column, "")
            if not value:
                continue
            times.append(dt.datetime.fromisoformat(row["time"]))
            brightness.append(float(value))
    if not times:
        raise ValueError(f"No brightness samples found in {csv_path} column {brightness_column}")
    return np.asarray(times, dtype=object), np.asarray(brightness, dtype=float)


def default_output_path(rocket):
    return Path(f"GNEISS_{rocket}_enorth_vs_brightness.png")


def despin_path_for_rocket(rocket):
    return DESPIN_DIR / f"despin_v2_{rocket}.csv"


def plot_rocket(rocket):
    despin_csv = despin_path_for_rocket(rocket)
    brightness_column = ROCKET_BRIGHTNESS_COLUMNS[rocket]
    despin_seconds, enorth = load_despin_enorth(despin_csv)
    brightness_datetimes, brightness = load_brightness(BRIGHTNESS_CSV, brightness_column)
    despin_rocket, t0_time = parse_despin_header(despin_csv)
    if despin_rocket != rocket:
        raise ValueError(f"Rocket {rocket} plot was given {despin_csv}, but its header says rocket {despin_rocket}")
    t0_datetime = dt.datetime.combine(brightness_datetimes[0].date(), t0_time)
    brightness_seconds = np.asarray(
        [(sample_dt - t0_datetime).total_seconds() for sample_dt in brightness_datetimes],
        dtype=float,
    )

    output_path = OUTPUT_DIR / default_output_path(rocket)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_enorth = plt.subplots(figsize=(12, 5))
    ax_brightness = ax_enorth.twinx()

    enorth_color = "tab:purple"
    brightness_color = "tab:orange"
    ax_enorth.plot(despin_seconds, enorth, color=enorth_color, linewidth=1.0, label=f"E_North ({despin_csv.name})")
    ax_brightness.plot(brightness_seconds, brightness, color=brightness_color, linewidth=1.0, label=brightness_column)

    ax_enorth.set_xlabel(f"Seconds since {rocket} T0")
    ax_enorth.set_ylabel("E_North (V/m)", color=enorth_color)
    ax_brightness.set_ylabel("ASI brightness", color=brightness_color)
    ax_enorth.set_xlim(100, 520)
    ax_enorth.set_ylim(-0.03, 0.04)
    ax_enorth.tick_params(axis="y", labelcolor=enorth_color)
    ax_brightness.tick_params(axis="y", labelcolor=brightness_color)
    ax_enorth.grid(True, alpha=0.3)
    ax_enorth.set_title(f"GNEISS/{rocket} E_North vs trajectory brightness | {despin_csv.name}")

    lines = ax_enorth.get_lines() + ax_brightness.get_lines()
    ax_enorth.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path} using {despin_csv}")


def main():
    for rocket in ROCKETS:
        plot_rocket(rocket)


if __name__ == "__main__":
    main()
