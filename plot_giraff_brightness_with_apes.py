#!/usr/bin/env python3
"""Plot GIRAFF trajectory brightness below an APES overview image."""

import argparse
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
from PIL import Image

from core.missions import mission_output_dir
from core.missions import default_time_range
from core.time_utils import parse_date_and_time
from core.traj_utils import get_launch_start_from_traj_csv, mission_trajectory_paths


APES_CONFIG = {
    "380": {
        "date": "20250209",
        "image": Path("/Users/anniepflaum/Downloads/APES_GIR380_ov_pad.jpg"),
        "xlim": (100.0, 520.0),
        "axis_left_frac": 323 / 2596,
        "axis_right_frac": 2269 / 2596,
        "axis_top_frac": 60 / 800,
        "axis_bottom_frac": 712 / 800,
    },
    "381": {
        "date": "20250202",
        "image": Path("/Users/anniepflaum/Downloads/APES_GIR381_ov_pad.jpg"),
        "xlim": (100.0, 520.0),
        "axis_left_frac": 323 / 2582,
        "axis_right_frac": 2248 / 2582,
        "axis_top_frac": 60 / 800,
        "axis_bottom_frac": 712 / 800,
    },
}


def infer_rocket_from_path(path):
    match = re.search(r"GIR(?:AFF)?[_-]?(\d{3})", str(path), re.IGNORECASE)
    return match.group(1) if match else None


def brightness_csv_candidates(rocket, date, color):
    out_dir = mission_output_dir("GIRAFF", color=color, date=date)
    return sorted(out_dir.glob(f"GIRAFF_brightness_vs_time_{date}_*_step*.csv"))


def parse_brightness_csv_name(path):
    match = re.match(
        r"^GIRAFF_brightness_vs_time_(\d{8})_(\d{6}(?:p\d+)?)_(\d{6}(?:p\d+)?)_step",
        path.name,
    )
    if not match:
        return None
    return {
        "date": match.group(1),
        "start": match.group(2).replace("p", "."),
        "end": match.group(3).replace("p", "."),
    }


def find_brightness_csv(rocket, date, color):
    candidates = brightness_csv_candidates(rocket, date, color)
    if not candidates:
        raise FileNotFoundError(
            f"No GIRAFF trajectory brightness CSV found for rocket {rocket} in "
            f"{mission_output_dir('GIRAFF', color=color, date=date)}"
        )
    default_start, default_end = default_time_range("GIRAFF", date)
    exact = [
        candidate
        for candidate in candidates
        if (parsed := parse_brightness_csv_name(candidate)) is not None
        and parsed["start"] == default_start
        and parsed["end"] == default_end
    ]
    if exact:
        return max(exact, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_brightness_series(csv_path):
    times = []
    values = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd)
        fieldnames = reader.fieldnames or []
        value_field = "main_reference_norm_brightness" if "main_reference_norm_brightness" in fieldnames else "main_brightness"
        if value_field not in fieldnames:
            raise ValueError(f"{csv_path} is missing main brightness columns")
        for row in reader:
            value = row.get(value_field, "")
            if not row.get("time") or not value:
                continue
            try:
                times.append(dt.datetime.fromisoformat(row["time"]))
                values.append(float(value))
            except ValueError:
                continue
    if not times:
        raise ValueError(f"No plottable brightness samples found in {csv_path}")
    return times, np.asarray(values, dtype=float), value_field


def seconds_since_t0(times, date):
    traj_path = mission_trajectory_paths("GIRAFF", date=date)["main"]
    launch_start = get_launch_start_from_traj_csv(str(traj_path))
    if launch_start is None:
        raise ValueError(f"Could not determine GIRAFF launch start from {traj_path}")
    launch_dt = parse_date_and_time(date, launch_start)
    return np.asarray([(time - launch_dt).total_seconds() for time in times], dtype=float), launch_start


def default_output_path(image_path, rocket, date, color):
    stem = Path(image_path).stem
    return mission_output_dir("GIRAFF", color=color, date=date) / f"{stem}_traj_brightness.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["380", "381"], default=None, help="GIRAFF rocket number")
    ap.add_argument("--date", default=None, help="Date YYYYMMDD; defaults from rocket")
    ap.add_argument("--image", default=None, help="APES overview JPEG path")
    ap.add_argument("--brightness-csv", default=None, help="Trajectory brightness CSV path")
    ap.add_argument("--color", choices=["green"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output PNG path")
    ap.add_argument("--x-min", type=float, default=None, help="Left x-limit in seconds since T0")
    ap.add_argument("--x-max", type=float, default=None, help="Right x-limit in seconds since T0")
    ap.add_argument("--axis-left-frac", type=float, default=None, help="APES data-axis left edge as image-width fraction")
    ap.add_argument("--axis-right-frac", type=float, default=None, help="APES data-axis right edge as image-width fraction")
    ap.add_argument("--axis-top-frac", type=float, default=None, help="APES data-axis top edge as image-height fraction")
    ap.add_argument("--axis-bottom-frac", type=float, default=None, help="APES data-axis bottom edge as image-height fraction")
    args = ap.parse_args()

    image_path = Path(args.image) if args.image else None
    rocket = args.rocket or (infer_rocket_from_path(image_path) if image_path else None)
    if rocket is None:
        ap.error("--rocket is required when --image does not contain GIR380 or GIR381")
    config = APES_CONFIG[rocket]
    date = args.date or config["date"]
    image_path = image_path or config["image"]
    if not image_path.exists():
        ap.error(f"APES image not found: {image_path}")

    x_min, x_max = config["xlim"]
    if args.x_min is not None:
        x_min = args.x_min
    if args.x_max is not None:
        x_max = args.x_max
    axis_left = config["axis_left_frac"] if args.axis_left_frac is None else args.axis_left_frac
    axis_right = config["axis_right_frac"] if args.axis_right_frac is None else args.axis_right_frac
    axis_top = config["axis_top_frac"] if args.axis_top_frac is None else args.axis_top_frac
    axis_bottom = config["axis_bottom_frac"] if args.axis_bottom_frac is None else args.axis_bottom_frac
    if not 0.0 <= axis_left < axis_right <= 1.0:
        ap.error("--axis-left-frac and --axis-right-frac must satisfy 0 <= left < right <= 1")
    if not 0.0 <= axis_top < axis_bottom <= 1.0:
        ap.error("--axis-top-frac and --axis-bottom-frac must satisfy 0 <= top < bottom <= 1")

    csv_path = Path(args.brightness_csv) if args.brightness_csv else find_brightness_csv(rocket, date, args.color)
    times, brightness, value_field = load_brightness_series(csv_path)
    t_since, launch_start = seconds_since_t0(times, date)

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    panel_h = int(round(img_h * (axis_bottom - axis_top)))
    gap_px = 30
    bottom_margin_px = 95
    dpi = 150
    fig_w = img_w / dpi
    fig_h = (img_h + gap_px + panel_h + bottom_margin_px) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    total_h = img_h + gap_px + panel_h + bottom_margin_px
    img_y = (bottom_margin_px + panel_h + gap_px) / total_h
    img_ax = fig.add_axes([0.0, img_y, 1.0, img_h / total_h])
    img_ax.imshow(image)
    img_ax.axis("off")

    ax = fig.add_axes([axis_left, bottom_margin_px / total_h, axis_right - axis_left, panel_h / total_h])
    ax.plot(t_since, brightness, color="#1b7837", linewidth=1.4)
    ax.set_xlim(x_min, x_max)
    positive = brightness[np.isfinite(brightness) & (brightness > 0)]
    if positive.size:
        ax.set_yscale("log")
        y_min = float(np.nanmin(positive))
        y_max = float(np.nanmax(positive))
        if y_max > y_min:
            ax.set_ylim(y_min * 0.85, y_max * 1.15)
    if value_field == "main_reference_norm_brightness":
        ax.set_ylabel("ASI norm.")
    else:
        ax.set_ylabel("ASI brightness")
    ax.set_xlabel("Seconds since launch")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"GIRAFF/{rocket} trajectory brightness | T0 {launch_start}", fontsize=10)

    output_path = Path(args.output) if args.output else default_output_path(image_path, rocket, date, args.color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Brightness CSV: {csv_path}")


if __name__ == "__main__":
    main()
