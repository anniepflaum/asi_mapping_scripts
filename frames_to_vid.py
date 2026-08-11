#!/usr/bin/env python3
"""Convert a folder of image frames into an MP4 movie."""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def natural_sort_key(path):
    """Sort embedded numbers numerically instead of lexicographically."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolution(value):
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("resolution must have the form WIDTHxHEIGHT")
    width, height = (int(part) for part in match.groups())
    if width < 2 or height < 2:
        raise argparse.ArgumentTypeError("resolution dimensions must be at least 2")
    # H.264 with yuv420p requires even dimensions.
    if width % 2 or height % 2:
        raise argparse.ArgumentTypeError("resolution dimensions must be even")
    return width, height


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder containing image frames")
    parser.add_argument("--fps", type=float, default=10.0, help="Movie frames per second")
    parser.add_argument(
        "--resolution",
        type=resolution,
        metavar="WIDTHxHEIGHT",
        default=(1024, 682),
        help="Output resolution, e.g. 1920x1080; preserves aspect ratio and pads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path; default is FOLDER/FOLDER_NAME.mp4",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality from 0 (lossless) to 51 (lowest); default: 18",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_frames(folder):
    frames = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not frames:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise FileNotFoundError(f"No image frames ({extensions}) found in {folder}")
    return frames


def concat_line(path):
    # FFmpeg's concat format escapes a single quote as: '\''
    escaped = str(path.resolve()).replace("'", r"'\''")
    return f"file '{escaped}'\n"


def write_concat_manifest(path, frames, fps):
    duration = 1.0 / fps
    with path.open("w", encoding="utf-8") as manifest:
        for index, frame in enumerate(frames):
            manifest.write(concat_line(frame))
            if index < len(frames) - 1:
                manifest.write(f"duration {duration:.12f}\n")


def video_filter(output_resolution):
    if output_resolution is None:
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    width, height = output_resolution
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
        "force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


def main():
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Frame folder does not exist: {folder}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")

    frames = find_frames(folder)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else folder / f"../{folder.name}.mp4"
    )
    if output.exists():
        if output.stat().st_size == 0:
            output.unlink()
        elif not args.overwrite:
            raise FileExistsError(f"Output already exists: {output} (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="frames_to_vid_") as temp_dir:
        manifest = Path(temp_dir) / "frames.txt"
        write_concat_manifest(manifest, frames, args.fps)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if args.overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-vf",
            video_filter(args.resolution),
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)

    duration = len(frames) / args.fps
    print(
        f"Saved {output} ({len(frames)} frames, {args.fps:g} fps, "
        f"{duration:.2f} s)"
    )


if __name__ == "__main__":
    main()
