#!/usr/bin/env python3
"""Download PKR DASC PNG files ending in 0558.png for a date/time range."""

import argparse
import datetime as dt
import re
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


BASE_URL = "https://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG"
FILENAME_RE = re.compile(r"^PFRR_(\d{8})_(\d{6})_0558\.png$", re.IGNORECASE)
HREF_RE = re.compile(r'href=["\']([^"\']*0558\.png)["\']', re.IGNORECASE)


def parse_datetime(date_str, time_str):
    date_digits = re.sub(r"\D", "", date_str)
    time_digits = re.sub(r"\D", "", time_str)
    if len(date_digits) != 8:
        raise ValueError("date must be YYYYMMDD")
    if len(time_digits) != 6:
        raise ValueError("time must be HHMMSS")
    return dt.datetime.strptime(date_digits + time_digits, "%Y%m%d%H%M%S")


def hourly_directory_urls(start_dt, end_dt):
    hour = start_dt.replace(minute=0, second=0, microsecond=0)
    end_hour = end_dt.replace(minute=0, second=0, microsecond=0)
    while hour <= end_hour:
        yield f"{BASE_URL}/{hour:%Y}/{hour:%Y%m%d}/{hour:%H}/"
        hour += dt.timedelta(hours=1)


def png_datetime(filename):
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def list_0558_png_urls(directory_url, start_dt, end_dt):
    response = requests.get(directory_url, timeout=30, verify=False)
    response.raise_for_status()
    urls = []
    seen = set()
    for href in HREF_RE.findall(response.text):
        filename = unquote(href).split("/")[-1]
        if filename in seen:
            continue
        file_dt = png_datetime(filename)
        if file_dt is None or not (start_dt <= file_dt <= end_dt):
            continue
        seen.add(filename)
        urls.append((file_dt, filename, urljoin(directory_url, filename)))
    return sorted(urls, key=lambda item: item[0])


def download_file(url, output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        return "exists"
    response = requests.get(url, timeout=60, stream=True, verify=False)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fd:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                fd.write(chunk)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(
        description="Download PKR DASC PNG files ending in 0558.png for a date/time range."
    )
    parser.add_argument("date", help="Date as YYYYMMDD")
    parser.add_argument("start", help="Start time as HHMMSS")
    parser.add_argument("end", help="End time as HHMMSS")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for downloaded PNGs; default: /Users/anniepflaum/asi_mapping/images/green/PKR/YYYYMMDD",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace files that already exist")
    args = parser.parse_args()

    try:
        start_dt = parse_datetime(args.date, args.start)
        end_dt = parse_datetime(args.date, args.end)
    except ValueError as exc:
        parser.error(str(exc))
    if end_dt < start_dt:
        parser.error("end time must be >= start time")

    output_dir = Path(args.output_dir) if args.output_dir else Path("/Users/anniepflaum/asi_mapping/images/green/PKR") / start_dt.strftime("%Y%m%d")

    files = []
    for directory_url in hourly_directory_urls(start_dt, end_dt):
        print(f"Listing {directory_url}")
        try:
            files.extend(list_0558_png_urls(directory_url, start_dt, end_dt))
        except requests.HTTPError as exc:
            print(f"  skipped: HTTP {exc.response.status_code}")
        except requests.RequestException as exc:
            print(f"  skipped: {exc}")

    if not files:
        print("No matching 0558 PNG files found.")
        return

    downloaded = 0
    skipped = 0
    for _file_dt, filename, url in files:
        output_path = output_dir / filename
        status = download_file(url, output_path, overwrite=args.overwrite)
        if status == "downloaded":
            downloaded += 1
            print(f"Downloaded {filename}")
        else:
            skipped += 1
            print(f"Exists {filename}")

    print(f"Done: {downloaded} downloaded, {skipped} already existed, {len(files)} total.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
