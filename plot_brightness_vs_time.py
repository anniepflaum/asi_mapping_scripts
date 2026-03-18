#!/usr/bin/env python3
"""
Plot left/right rocket brightness versus time from a brightness-vs-time dataset CSV.
"""

import argparse
import csv
from pathlib import Path

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def parse_optional_float(value):
    value = (value or "").strip()
    return float(value) if value else None


def load_dataset(csv_path):
    times = []
    left = []
    right = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            iso_time = (row.get("iso_time") or "").strip()
            if not iso_time:
                continue
            times.append(dt.datetime.fromisoformat(iso_time))
            left.append(parse_optional_float(row.get("left_brightness", "")))
            right.append(parse_optional_float(row.get("right_brightness", "")))
    return times, left, right


def make_output_path(input_path, output_arg):
    if output_arg:
        return Path(output_arg)
    return Path(input_path).with_suffix(".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="../mapped/green/brightness_vs_time_20260210_101800_103100_step0p3 copy.csv", help="Brightness-vs-time CSV file")
    ap.add_argument("--output", default="../mapped/green/brightness_vs_time.png", help="Output plot path")
    ap.add_argument("--title", default="Brightness vs Time", help="Plot title")
    args = ap.parse_args()

    times, left, right = load_dataset(args.input_csv)
    if not times:
        raise ValueError(f"No rows with iso_time found in {args.input_csv}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(times, left, color="tab:red", s=10, label="Left brightness")
    ax.scatter(times, right, color="tab:blue", s=10, label="Right brightness")

    ax.set_title(args.title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Brightness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = make_output_path(args.input_csv, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
