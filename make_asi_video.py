#!/usr/bin/env python3
"""
make_asi_video.py

Generates a video of mapped ASI images for a given time range using map_asi_archive.py logic.

Usage:
    python make_asi_video.py --date 20260210 --start 100000 --end 102000 --step 60 [--pretty] [--output video.mp4]

Arguments:
    --date      Date for the ASI images (format: YYYYMMDD)
    --start     Start time (format: HHMMSS)
    --end       End time (format: HHMMSS)
    --step      Step in seconds between frames (default: 60)
    --pretty    Use pretty plotting mode (default: fast)
    --output    Output video filename (default: asi_video.mp4)

Notes:
    - Requires ffmpeg (via imageio[ffmpeg])
    - Deletes intermediate PNGs after video creation
"""
import argparse
import subprocess
import os
import sys
import datetime as dt
import imageio.v3 as iio
from tqdm import tqdm
import imageio.v2 as imageio
import shutil

def time_range(start, end, step):
    """Yield time strings from start to end (inclusive) with given step in seconds."""
    t0 = dt.datetime.strptime(start, "%H%M%S")
    t1 = dt.datetime.strptime(end, "%H%M%S")
    while t0 <= t1:
        yield t0.strftime("%H%M%S")
        t0 += dt.timedelta(seconds=step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260210", type=str, help="Date for the ASI images (YYYYMMDD)")
    parser.add_argument("--start", required=True, type=str, help="Start time (HHMMSS)")
    parser.add_argument("--end", required=True, type=str, help="End time (HHMMSS)")
    parser.add_argument("--step", type=int, default=60, help="Step in seconds between frames")
    parser.add_argument("--pretty", action="store_true", help="Use pretty plotting mode")
    parser.add_argument("--sites", nargs='*', default=['ARV', 'BVR', 'VEE', 'PKR'], help="List of sites to process (default: all sites)")
    args = parser.parse_args()

    # Directory for intermediate PNGs
    png_dir = "_asi_video_frames"
    os.makedirs(png_dir, exist_ok=True)
    frame_paths = []

    # Compose sites string for output filenames
    sites_str = '_'.join(sorted([s.upper() for s in args.sites]))

    # Set output video path to ../mapped/GNEISS_launch_{sites_str}.mp4
    args.output = f"../mapped/GNEISS_launch_{sites_str}.mp4"

    # Generate frames
    print("Generating frames...")
    for tstr in tqdm(list(time_range(args.start, args.end, args.step))):
        mode = "pretty" if args.pretty else "fast"
        outname = f"GNEISS_launch_{mode}_{sites_str}_{args.date}_{tstr}.png"
        outpath = os.path.join(png_dir, outname)
        cmd = [sys.executable, "map_asi_archive.py",
                "--time", tstr, "--sites"] + args.sites
        if args.pretty:
            cmd.append("--pretty")
        # Suppress output from map_asi_archive.py
        with open(os.devnull, 'w') as devnull:
            subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
        # Move the output PNG to png_dir
        src = f"../mapped/{outname}"
        if not os.path.exists(src):
            print(f"Warning: {src} not found, skipping.")
            continue
        os.rename(src, outpath)
        frame_paths.append(outpath)

    if not frame_paths:
        print("No frames generated. Exiting.")
        return

    # Create video
    print(f"Writing video to {args.output} ...")
    frames = [imageio.imread(frame) for frame in frame_paths]
    # Use ffmpeg writer explicitly for mp4 output
    imageio.mimsave(args.output, frames, fps=10, format='ffmpeg', macro_block_size=1)

    # Delete intermediate PNGs
    print("Cleaning up intermediate PNGs...")
    for f in os.listdir(png_dir):
        try:
            os.remove(os.path.join(png_dir, f))
        except Exception as e:
            print(f"Warning: could not remove {f}: {e}")
    try:
        os.rmdir(png_dir)
    except Exception as e:
        print(f"Warning: could not remove directory {png_dir}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
