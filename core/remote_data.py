from io import BytesIO
import datetime as dt
import re
from pathlib import Path

from core.paths import IMAGE_DIR

import numpy as np
import requests
from PIL import Image
from core.fetch_url import closest_amisr_png_url
PKR_LOCAL_ROOT = IMAGE_DIR
PKR_FILENAME_RE = re.compile(r"^PFRR_(\d{8})_(\d{6})_(0558|0630)\.png$", re.IGNORECASE)


def retrieve_image(url, verbose=True):
    """
    Download a single-channel image from a URL and return it as float32.
    """
    if verbose:
        print(f"PKR: {url} ...")
    resp = requests.get(url, verify=False)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_image_file(path):
    img = Image.open(path)
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def pkr_png_datetime(path):
    match = PKR_FILENAME_RE.match(Path(path).name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def pkr_png_datetime_from_source(source):
    frame_dt = pkr_png_datetime(Path(str(source)))
    if frame_dt is not None:
        return frame_dt
    match = re.search(r"(\d{8})_(\d{6})", str(source))
    if not match:
        raise ValueError(f"Could not parse PKR frame timestamp from {source}")
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def closest_local_pkr_png(date, time, color="green", tolerance_seconds=15):
    color = str(color).lower().strip()
    suffix = "0558" if color == "green" else "0630"
    target = dt.datetime.strptime(re.sub(r"\D", "", date) + re.sub(r"\D", "", time), "%Y%m%d%H%M%S")
    base_dir = PKR_LOCAL_ROOT / color / "PKR"
    search_dirs = [
        base_dir / target.strftime("%Y%m%d"),
        base_dir / target.strftime("%Y") / target.strftime("%Y%m%d") / target.strftime("%H"),
        base_dir / target.strftime("%Y") / target.strftime("%Y%m%d"),
        base_dir,
    ]
    candidates = []
    seen = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob(f"PFRR_*_*_{suffix}.png"):
            if path in seen:
                continue
            seen.add(path)
            frame_dt = pkr_png_datetime(path)
            if frame_dt is None:
                continue
            delta = abs((frame_dt - target).total_seconds())
            if delta <= tolerance_seconds:
                candidates.append((delta, frame_dt, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[2].name))
    return candidates[0][2], candidates[0][1]


def load_pkr_image(date, time, color="green", verbose=True):
    """
    Load the closest PKR PNG, preferring local images files and falling back
    to the AMISR archive URL if no local file is available.
    """
    local = closest_local_pkr_png(date, time, color=color)
    if local is not None:
        path, frame_dt = local
        if verbose:
            print(f"PKR: local {path} ...")
        return load_image_file(path), str(path), frame_dt

    url = closest_amisr_png_url("PKR", date, time, color=color)
    return retrieve_image(url, verbose=verbose), url, pkr_png_datetime_from_source(url)
