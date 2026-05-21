#!/usr/bin/env python3
"""Compare GIRAFF default-site and PKR-only trajectory brightness series."""

import csv
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


BASE_DIR = Path("/Users/anniepflaum/asi_mapping/mapped/green/GIRAFF")
VALUE_COLUMN = "main_reference_norm_brightness"
PANELS = [
    {
        "rocket": "381",
        "title": "GIRAFF/381 trajectory brightness",
        "default": BASE_DIR / "381" / "GIRAFF_brightness_vs_time_20250202_070715_071636_step0p05.csv",
        "pkr": BASE_DIR / "381" / "GIRAFF_brightness_vs_time_20250202_070715_071636_step0p05_sites_PKR.csv",
    },
    {
        "rocket": "380",
        "title": "GIRAFF/380 trajectory brightness",
        "default": BASE_DIR / "380" / "GIRAFF_brightness_vs_time_20250209_083501_084410_step0p05.csv",
        "pkr": BASE_DIR / "380" / "GIRAFF_brightness_vs_time_20250209_083501_084410_step0p05_sites_PKR.csv",
    },
]


def load_series(csv_path, value_column=VALUE_COLUMN):
    times = []
    values = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if value_column not in fieldnames:
            candidates = [field for field in fieldnames if field.endswith("_reference_norm_brightness")]
            if not candidates:
                raise KeyError(f"{csv_path} missing {value_column}")
            value_column = candidates[0]
        for row in reader:
            value = row.get(value_column, "")
            if not row.get("time") or not value:
                continue
            try:
                values.append(float(value))
                times.append(dt.datetime.fromisoformat(row["time"]))
            except ValueError:
                continue
    return times, values


def plot_panel(ax, config):
    for path in (config["default"], config["pkr"]):
        if not path.exists():
            raise FileNotFoundError(f"Missing brightness CSV: {path}")

    default_times, default_values = load_series(config["default"])
    pkr_times, pkr_values = load_series(config["pkr"])

    ax.plot(default_times, default_values, color="tab:blue", linewidth=1.2, label="VEE")
    ax.plot(pkr_times, pkr_values, color="tab:orange", linewidth=1.2, label="PKR")
    ax.set_title(config["title"])
    ax.set_ylabel("ASI norm.")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def main():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for ax, config in zip(axes, PANELS):
        plot_panel(ax, config)
    axes[-1].set_xlabel("UTC time")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = BASE_DIR / "GIRAFF_PKR_vs_default_brightness.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    for config in PANELS:
        print(f"{config['rocket']} default: {config['default']}")
        print(f"{config['rocket']} PKR: {config['pkr']}")


if __name__ == "__main__":
    main()
