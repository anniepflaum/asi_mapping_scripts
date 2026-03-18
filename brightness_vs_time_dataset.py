#!/usr/bin/env python3
"""
Build a brightness-vs-time dataset for both GNEISS trajectories.

For each time step in a requested range, this script:
1) Loads the closest ASI frame per selected site.
2) Finds each rocket position (left/right trajectory) at that time.
3) Samples brightness at rocket position using the same logic as map_asi_archive.py
   (mean of 25 nearest valid pixels, with percentile metadata).
4) Writes one CSV row per timestamp with left/right values.
"""

import argparse
import datetime as dt
from pathlib import Path

import check_intersect as ci
from asi_time_utils import (
    parse_date_and_time,
    parse_hhmmss_fractional,
    sanitize_time_for_filename,
)
from map_asi_archive import (
    FRAME_INTERVAL_SECONDS_GREEN,
    FRAME_INTERVAL_SECONDS_RED,
    best_rocket_brightness,
    closest_amisr_png_url,
    load_skymaps,
    retrieve_image,
)
from tiff_utils import build_tiff_metadata, get_site_tiff_candidates, load_best_frame_from_cached_tiffs
from traj_utils import build_traj_lookup, lookup_traj_position


def format_time_arg(t):
    hhmmss = t.strftime("%H%M%S")
    frac = f"{t.microsecond:06d}".rstrip("0")
    return f"{hhmmss}.{frac}" if frac else hhmmss


def make_output_path(out_arg, date, start, end, step):
    if out_arg:
        return Path(out_arg)
    start_tok = sanitize_time_for_filename(start)
    end_tok = sanitize_time_for_filename(end)
    step_tok = str(step).replace(".", "p")
    return Path(f"brightness_vs_time_{date}_{start_tok}_{end_tok}_step{step_tok}.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260210", help="Date YYYYMMDD")
    ap.add_argument("--start", required=True, help="Start time HHMMSS(.fraction)")
    ap.add_argument("--end", required=True, help="End time HHMMSS(.fraction)")
    ap.add_argument("--step", type=float, default=0.3, help="Step size in seconds (default: 0.3)")
    ap.add_argument("--sites", nargs="*", default=["ARV", "BVR", "VEE", "PKR"], help="Sites to include")
    ap.add_argument("--color", choices=["green", "red"], default="green", help="ASI color channel")
    ap.add_argument("--output", default=None, help="Output CSV path")
    ap.add_argument("--arv-tiffs", nargs="*", default=None, help="Optional ARV TIFF folder(s)")
    ap.add_argument("--vee-tiffs", nargs="*", default=None, help="Optional VEE TIFF folder(s)")
    ap.add_argument("--bvr-tiffs", nargs="*", default=None, help="Optional BVR TIFF folder(s)")
    args = ap.parse_args()

    try:
        parse_hhmmss_fractional(args.start)
        parse_hhmmss_fractional(args.end)
    except ValueError as exc:
        ap.error(str(exc))
    if args.step <= 0:
        ap.error("--step must be > 0")

    start_dt = parse_date_and_time(args.date, args.start)
    end_dt = parse_date_and_time(args.date, args.end)
    if end_dt < start_dt:
        ap.error("--end must be >= --start")

    selected_sites = set(s.upper() for s in args.sites)
    skymaps = load_skymaps(selected_sites, color=args.color)

    # Build intersection masks once.
    sites = list(skymaps.keys())
    for s0 in sites:
        red_sites = sites.copy()
        red_sites.remove(s0)
        skymaps[s0]["extra_masks"] = {}
        for s1 in red_sites:
            m0, m1 = ci.calculate_masks(
                skymaps[s0]["site_lat"],
                skymaps[s0]["site_lon"],
                skymaps[s0]["azmt"],
                skymaps[s0]["elev"],
                skymaps[s1]["site_lat"],
                skymaps[s1]["site_lon"],
                skymaps[s1]["azmt"],
                skymaps[s1]["elev"],
            )
            skymaps[s0]["extra_masks"][s1] = m0

    tiff_overrides = {
        "ARV": args.arv_tiffs,
        "VEE": args.vee_tiffs,
        "BVR": args.bvr_tiffs,
    }
    tiff_candidates = {}
    tiff_metadata = {}
    for site in ["ARV", "VEE", "BVR"]:
        if site in selected_sites:
            tiff_candidates[site] = get_site_tiff_candidates(site, args.date, args.color, tiff_overrides[site])

    frame_interval = FRAME_INTERVAL_SECONDS_GREEN if args.color == "green" else FRAME_INTERVAL_SECONDS_RED
    for site in ["ARV", "VEE", "BVR"]:
        if site in tiff_candidates:
            tiff_metadata[site] = build_tiff_metadata(tiff_candidates[site], frame_interval)

    left_traj = build_traj_lookup("36.397_AttitudeSolution.csv")
    right_traj = build_traj_lookup("36.398_AttitudeSolution.csv")

    out_path = make_output_path(args.output, args.date, args.start, args.end, args.step)

    fieldnames = [
        "date",
        "time",
        "iso_time",
        "left_site",
        "left_percentile",
        "left_brightness",
        "left_distance_deg",
        "left_n_pixels",
        "right_site",
        "right_percentile",
        "right_brightness",
        "right_distance_deg",
        "right_n_pixels",
        "sites_loaded",
    ]

    rows = []
    step_td = dt.timedelta(seconds=args.step)
    t = start_dt
    while t <= end_dt:
        time_arg = format_time_arg(t)
        imgs_raw = {}

        for site in ["ARV", "VEE", "BVR"]:
            if site not in selected_sites:
                continue
            try:
                im_raw = load_best_frame_from_cached_tiffs(
                    site,
                    tiff_metadata.get(site, []),
                    t,
                    frame_interval=frame_interval,
                )
                imgs_raw[site] = im_raw
            except Exception as exc:
                print(f"{time_arg} {site}: frame load failed: {exc}")

        if "PKR" in selected_sites:
            try:
                pkr_lookup_time = t.strftime("%H%M%S")
                url_pkr = closest_amisr_png_url("PKR", args.date, pkr_lookup_time, color=args.color)
                imgs_raw["PKR"] = retrieve_image(url_pkr)
            except Exception as exc:
                print(f"{time_arg} PKR: frame load failed: {exc}")

        lat_l, lon_l = lookup_traj_position(left_traj, time_arg)
        lat_r, lon_r = lookup_traj_position(right_traj, time_arg)

        left = best_rocket_brightness(lat_l, lon_l, skymaps, imgs_raw) if lat_l is not None and lon_l is not None else None
        right = best_rocket_brightness(lat_r, lon_r, skymaps, imgs_raw) if lat_r is not None and lon_r is not None else None

        rows.append(
            {
                "date": args.date,
                "time": time_arg,
                "iso_time": t.isoformat(),
                "left_site": left["site"] if left else "",
                "left_percentile": f"{left['percentile']:.3f}" if left else "",
                "left_brightness": f"{left['raw_brightness']:.3f}" if left else "",
                "left_distance_deg": f"{left['distance_deg']:.6f}" if left else "",
                "left_n_pixels": left["n_pixels"] if left else "",
                "right_site": right["site"] if right else "",
                "right_percentile": f"{right['percentile']:.3f}" if right else "",
                "right_brightness": f"{right['raw_brightness']:.3f}" if right else "",
                "right_distance_deg": f"{right['distance_deg']:.6f}" if right else "",
                "right_n_pixels": right["n_pixels"] if right else "",
                "sites_loaded": ",".join(sorted(imgs_raw.keys())),
            }
        )
        t += step_td

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
