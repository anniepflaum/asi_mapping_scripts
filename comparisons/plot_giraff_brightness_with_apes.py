#!/usr/bin/env python3
"""Overlay GIRAFF trajectory brightness on APES overview images."""

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
from scipy.io import loadmat

from core.missions import default_time_range, mission_output_dir, mission_trajectory_paths
from core.time_utils import parse_date_and_time
from core.traj_utils import get_launch_start_from_traj_csv


APES_MINUTE_DIR = Path("/Users/anniepflaum/asi_mapping/apes_GIRAFF_plots/APES_GIR_1minplots")
APES_MINUTE_ORIGINS = {
    "380": 110.0,
    "381": 127.0,
}
APES_MINUTE_AXIS = {
    "axis_left_frac": 197 / 1600,
    "axis_right_frac": 1369 / 1600,
    "axis_top_frac": 60 / 800,
    "axis_bottom_frac": 712 / 800,
}


APES_CONFIG = {
    "380": {
        "date": "20250209",
        "image": Path("/Users/anniepflaum/asi_mapping/apes_GIRAFF_plots/APES_GIR380_ov_pad.jpg"),
        "xlim": (100.0, 520.0),
        "axis_left_frac": 323 / 2596,
        "axis_right_frac": 2269 / 2596,
        "axis_top_frac": 60 / 800,
        "axis_bottom_frac": 712 / 800,
    },
    "381": {
        "date": "20250202",
        "image": Path("/Users/anniepflaum/asi_mapping/apes_GIRAFF_plots/APES_GIR381_ov_pad.jpg"),
        "xlim": (100.0, 520.0),
        "axis_left_frac": 323 / 2582,
        "axis_right_frac": 2248 / 2582,
        "axis_top_frac": 60 / 800,
        "axis_bottom_frac": 712 / 800,
    },
}


def minute_window(rocket, minute):
    minute_origin = APES_MINUTE_ORIGINS.get(str(rocket), 0.0)
    x_min = minute_origin + (minute - 1) * 60.0
    return x_min, x_min + 60.0


def default_minute_fig_path(rocket, minute):
    path = APES_MINUTE_DIR / f"APES_GIR{rocket}_min{minute}.fig"
    if path.exists():
        return path
    if str(rocket) == "381" and minute == 2:
        fallback = APES_MINUTE_DIR / "APES_GIR381_min2.fig"
        if fallback.exists():
            return fallback
    return path


def apes_xlim_from_fig(fig_path):
    data = loadmat(fig_path, squeeze_me=True, struct_as_record=False)
    roots = [value for key, value in data.items() if key.startswith("hgS_")]
    if not roots:
        raise ValueError(f"No MATLAB figure root found in {fig_path}")
    children = np.asarray(roots[0].children, dtype=object).flat
    for child in children:
        if getattr(child, "type", None) == "axes":
            xlim = np.asarray(child.properties.XLim, dtype=float)
            if xlim.size == 2 and np.all(np.isfinite(xlim)):
                return float(xlim[0]), float(xlim[1])
    raise ValueError(f"No axes XLim found in {fig_path}")


def minute_xlim(rocket, minute):
    fig_path = default_minute_fig_path(rocket, minute)
    if fig_path.exists():
        try:
            return apes_xlim_from_fig(fig_path)
        except Exception as exc:
            print(f"Warning: could not read APES x-limits from {fig_path}: {exc}")
    return minute_window(rocket, minute)


def default_minute_image_path(rocket, minute):
    path = APES_MINUTE_DIR / f"APES_GIR{rocket}_min{minute}.jpg"
    if path.exists():
        return path
    if str(rocket) == "381" and minute == 2:
        fallback = APES_MINUTE_DIR / "APES_GIR391_min2.jpg"
        if fallback.exists():
            return fallback
    return path


def brightness_csv_candidates(rocket, date, color):
    out_dir = mission_output_dir("GIRAFF", color=color, date=date)
    return sorted(out_dir.glob(f"GIRAFF_brightness_vs_time_{date}_*_step*.csv"))


def parse_brightness_csv_name(path):
    match = re.match(
        r"^GIRAFF_brightness_vs_time_(\d{8})_"
        r"(\d{6}(?:p\d+)?)_(\d{6}(?:p\d+)?)_"
        r"step\d+(?:p\d+)?"
        r"(?:_alt_(\d+(?:p\d+)?)km)?"
        r"(?P<site_suffix>_sites_[A-Z0-9_]+)?\.csv$",
        path.name,
    )
    if not match:
        return None
    return {
        "date": match.group(1),
        "start": match.group(2).replace("p", "."),
        "end": match.group(3).replace("p", "."),
        "alt": float(match.group(4).replace("p", ".")) if match.group(4) else 110.0,
        "site_suffix": match.group("site_suffix") or "",
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
        and parsed["alt"] == 110.0
        and not parsed["site_suffix"]
    ]
    if exact:
        return max(exact, key=lambda path: path.stat().st_mtime)
    exact_110 = [
        candidate
        for candidate in candidates
        if (parsed := parse_brightness_csv_name(candidate)) is not None
        and parsed["start"] == default_start
        and parsed["end"] == default_end
        and parsed["alt"] == 110.0
    ]
    if exact_110:
        return max(exact_110, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_all_alt_brightness_csvs(rocket, date, color):
    default_start, default_end = default_time_range("GIRAFF", date)
    by_alt = {}
    for candidate in brightness_csv_candidates(rocket, date, color):
        parsed = parse_brightness_csv_name(candidate)
        if parsed is None or parsed["start"] != default_start or parsed["end"] != default_end:
            continue
        existing = by_alt.get(parsed["alt"])
        prefer_candidate = existing is None
        if existing is not None:
            existing_parsed = parse_brightness_csv_name(existing)
            existing_has_site_suffix = bool(existing_parsed and existing_parsed["site_suffix"])
            candidate_has_site_suffix = bool(parsed["site_suffix"])
            prefer_candidate = (
                existing_has_site_suffix
                and not candidate_has_site_suffix
            ) or (
                existing_has_site_suffix == candidate_has_site_suffix
                and candidate.stat().st_mtime > existing.stat().st_mtime
            )
        if prefer_candidate:
            by_alt[parsed["alt"]] = candidate
    if not by_alt:
        raise FileNotFoundError(
            f"No GIRAFF trajectory brightness CSVs found for rocket {rocket} in "
            f"{mission_output_dir('GIRAFF', color=color, date=date)}"
        )
    return sorted(by_alt.items(), key=lambda item: item[0])


def load_brightness_series(csv_path):
    times = []
    values = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fd:
        reader = csv.DictReader(fd)
        fieldnames = reader.fieldnames or []
        if "main_reference_norm_brightness" in fieldnames:
            value_field = "main_reference_norm_brightness"
        elif "main_brightness" in fieldnames:
            value_field = "main_brightness"
        else:
            candidates = [field for field in fieldnames if field.endswith("_reference_norm_brightness")]
            if not candidates:
                candidates = [field for field in fieldnames if field.endswith("_brightness")]
            value_field = candidates[0] if candidates else ""
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


def all_alt_output_path(image_path, rocket, date, color):
    stem = Path(image_path).stem
    return mission_output_dir("GIRAFF", color=color, date=date) / f"{stem}_traj_brightness_all_alts.png"


def plot_brightness_with_apes(
    image_path,
    output_path,
    rocket,
    date,
    color,
    series,
    launch_start,
    x_min,
    x_max,
    axis_left,
    axis_right,
    axis_top,
    axis_bottom,
    title_suffix=None,
):
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    dpi = 150
    fig_w = img_w / dpi
    fig_h = img_h / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    img_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    img_ax.imshow(image)
    img_ax.axis("off")

    ax = fig.add_axes([axis_left, 1.0 - axis_bottom, axis_right - axis_left, axis_bottom - axis_top])
    ax.patch.set_alpha(0.0)
    colors = plt.get_cmap("tab10")
    line_color = "#f28e2b" if len(series) == 1 else "black"
    positive_values = []
    value_fields = set()
    for idx, item in enumerate(series):
        t_since = item["t_since"]
        brightness = item["brightness"]
        in_window = (t_since >= x_min) & (t_since <= x_max)
        color_value = "#f28e2b" if len(series) == 1 else colors(idx % 10)
        ax.plot(t_since[in_window], brightness[in_window], color=color_value, linewidth=2.0, label=item["label"])
        positive_values.append(brightness[in_window & np.isfinite(brightness) & (brightness > 0)])
        value_fields.add(item["value_field"])
    ax.set_xlim(x_min, x_max)
    positive = np.concatenate([values for values in positive_values if values.size]) if any(values.size for values in positive_values) else np.array([])
    if positive.size:
        ax.set_yscale("log")
        y_min = float(np.nanmin(positive))
        y_max = float(np.nanmax(positive))
        if y_max > y_min:
            ax.set_ylim(y_min * 0.85, y_max * 1.15)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", colors=line_color, labelsize=8)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ylabel = "ASI norm." if value_fields == {"main_reference_norm_brightness"} else "ASI brightness"
    ax.set_ylabel(ylabel, color=line_color, fontsize=9)
    ax.spines["right"].set_color(line_color)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    if len(series) > 1:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")
    for item in series:
        print(f"{item['label']}: {item['csv_path']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocket", choices=["380", "381"], required=True, help="GIRAFF rocket number")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--minute", type=int, nargs="+", default=None, help="APES 1-minute panel number(s) to compare")
    ap.add_argument("--all-alts", action="store_true", help="Overlay all available green-altitude brightness CSVs")
    args = ap.parse_args()
    if args.minute is not None and any(minute < 1 for minute in args.minute):
        ap.error("--minute values must be >= 1")

    rocket = args.rocket
    config = APES_CONFIG[rocket]
    date = config["date"]

    if args.all_alts:
        csv_specs = [(alt, path) for alt, path in find_all_alt_brightness_csvs(rocket, date, args.color)]
    else:
        csv_specs = [(110.0, find_brightness_csv(rocket, date, args.color))]

    series = []
    launch_start = None
    for alt, csv_path in csv_specs:
        times, brightness, value_field = load_brightness_series(csv_path)
        t_since, launch_start = seconds_since_t0(times, date)
        alt_label = f"{alt:g} km" if args.all_alts else "ASI brightness"
        series.append(
            {
                "label": alt_label,
                "csv_path": csv_path,
                "t_since": t_since,
                "brightness": brightness,
                "value_field": value_field,
            }
        )

    if args.minute is not None:
        plot_specs = []
        for minute in args.minute:
            minute_image = default_minute_image_path(rocket, minute)
            x_min, x_max = minute_xlim(rocket, minute)
            plot_specs.append((minute_image, x_min, x_max, f"APES min {minute}", APES_MINUTE_AXIS))
    else:
        plot_specs = [(config["image"], config["xlim"][0], config["xlim"][1], None, config)]

    for spec_image_path, x_min, x_max, title_suffix, axis_config in plot_specs:
        if not spec_image_path.exists():
            ap.error(f"APES image not found: {spec_image_path}")
        if not any(np.any((item["t_since"] >= x_min) & (item["t_since"] <= x_max)) for item in series):
            print(f"{spec_image_path.name}: no brightness samples between {x_min:g} and {x_max:g} s, skipping")
            continue
        output_path = all_alt_output_path(spec_image_path, rocket, date, args.color) if args.all_alts else default_output_path(spec_image_path, rocket, date, args.color)
        plot_brightness_with_apes(
            spec_image_path,
            output_path,
            rocket,
            date,
            args.color,
            series,
            launch_start,
            x_min,
            x_max,
            axis_config["axis_left_frac"],
            axis_config["axis_right_frac"],
            axis_config["axis_top_frac"],
            axis_config["axis_bottom_frac"],
            title_suffix=title_suffix,
        )


if __name__ == "__main__":
    main()
