#!/usr/bin/env python3
"""Export GNEISS rocket magnetic latitude and longitude versus time since TG."""

import argparse
import csv
import datetime as dt
from pathlib import Path
import sys

import numpy as np
from apexpy import Apex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.missions import GNEISS_DEFAULT_DATE, mission_output_dir
from core.paths import GNEISS_LEFT_TRAJECTORY_PATH, GNEISS_RIGHT_TRAJECTORY_PATH
from core.time_utils import hhmmss_fractional_to_seconds
from core.traj_utils import get_launch_start_from_traj_csv, load_traj_records


STEP_SECONDS = 0.05
TG_TIME = "101900"
TRAJECTORY_PATHS = {
    "397": GNEISS_LEFT_TRAJECTORY_PATH,
    "398": GNEISS_RIGHT_TRAJECTORY_PATH,
}


def format_step_token(step):
    return str(float(step)).replace(".", "p")


def default_output_path():
    return mission_output_dir("GNEISS", color="green") / f"tg_to_maglat.csv"


def interpolate_longitudes(sample_times, source_times, source_lons):
    """Interpolate longitude without introducing jumps at the +/-180 seam."""
    unwrapped = np.unwrap(np.deg2rad(source_lons))
    return np.rad2deg(np.interp(sample_times, source_times, unwrapped))


def resample_magnetic_trajectory(rocket, step, apex):
    """Return TG-relative time and apex coordinates for one GNEISS rocket."""
    traj_path = TRAJECTORY_PATHS[rocket]
    _utc_times, flight_times, lats, lons, alts_km = load_traj_records(traj_path)
    launch_start = get_launch_start_from_traj_csv(traj_path)
    launch_offset_s = hhmmss_fractional_to_seconds(launch_start) - hhmmss_fractional_to_seconds(TG_TIME)

    start_s = max(0.0, float(np.nanmin(flight_times)))
    end_s = float(np.nanmax(flight_times))
    sample_flight_times = np.arange(start_s, end_s + 1e-9, step, dtype=float)
    sample_lats = np.interp(sample_flight_times, flight_times, lats)
    sample_lons = interpolate_longitudes(sample_flight_times, flight_times, lons)
    sample_alts_km = np.interp(sample_flight_times, flight_times, alts_km)
    magnetic_lats, magnetic_lons = apex.convert(
        sample_lats,
        sample_lons,
        "geo",
        "apex",
        height=sample_alts_km,
    )
    return {
        "tg_times": sample_flight_times + launch_offset_s,
        "magnetic_lats": np.asarray(magnetic_lats, dtype=float),
        "magnetic_lons": np.asarray(magnetic_lons, dtype=float),
        "launch_offset_s": launch_offset_s,
    }


def values_on_tg_grid(tg_grid, series):
    """Interpolate one magnetic-coordinate series onto the shared TG grid."""
    times = series["tg_times"]
    in_range = (tg_grid >= times[0] - 1e-9) & (tg_grid <= times[-1] + 1e-9)
    magnetic_lats = np.full(tg_grid.shape, np.nan, dtype=float)
    magnetic_lons = np.full(tg_grid.shape, np.nan, dtype=float)
    magnetic_lats[in_range] = np.interp(tg_grid[in_range], times, series["magnetic_lats"])
    magnetic_lons[in_range] = np.interp(tg_grid[in_range], times, series["magnetic_lons"])
    return magnetic_lats, magnetic_lons


def write_csv(output_path, step, series_by_rocket):
    end_s = max(float(series["tg_times"][-1]) for series in series_by_rocket.values())
    tg_grid = np.arange(0.0, end_s + 1e-9, step, dtype=float)
    sampled = {
        rocket: values_on_tg_grid(tg_grid, series)
        for rocket, series in series_by_rocket.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_since_TG_s",
                "397_magnetic_lat_deg",
                "397_magnetic_lon_deg",
                "398_magnetic_lat_deg",
                "398_magnetic_lon_deg",
            ]
        )
        for idx, tg_time in enumerate(tg_grid):
            row = [f"{tg_time:.6f}"]
            for rocket in ("397", "398"):
                magnetic_lats, magnetic_lons = sampled[rocket]
                if np.isfinite(magnetic_lats[idx]) and np.isfinite(magnetic_lons[idx]):
                    row.extend([f"{magnetic_lats[idx]:.8f}", f"{magnetic_lons[idx]:.8f}"])
                else:
                    row.extend(["", ""])
            writer.writerow(row)

    print(f"Saved magnetic trajectory CSV to {output_path}")
    for rocket in ("397", "398"):
        series = series_by_rocket[rocket]
        print(
            f"{rocket}: T0 TG+{series['launch_offset_s']:.1f}s, "
            f"magnetic coordinates through TG+{series['tg_times'][-1]:.1f}s"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output_path()
    apex = Apex(date=dt.datetime.strptime(GNEISS_DEFAULT_DATE, "%Y%m%d"))
    series_by_rocket = {
        rocket: resample_magnetic_trajectory(rocket, STEP_SECONDS, apex)
        for rocket in ("397", "398")
    }
    write_csv(output_path, STEP_SECONDS, series_by_rocket)


if __name__ == "__main__":
    main()
