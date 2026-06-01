#!/usr/bin/env python3
"""Extract a GIRAFF SOK time window into a local disk-backed frame cache."""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import tifffile

from core.paths import WORKSPACE_DIR
from core.tiff_utils import parse_giraff_log


FILTER_COLORS = {
    "8446": "red",
    "5577": "green",
    "4278": "blue",
}
GIRAFF_CACHE_WINDOWS = {
    "20250202": ("070618", "071637"),
    "20250209": ("083140", "084415"),
}


def frame_bounds(tiff_start_dt, frame_interval, start_dt, end_dt):
    """Return inclusive source-frame bounds covering start_dt..end_dt."""
    start_idx = int(np.floor((start_dt - tiff_start_dt).total_seconds() / frame_interval))
    end_idx = int(np.ceil((end_dt - tiff_start_dt).total_seconds() / frame_interval))
    return max(start_idx, 0), max(end_idx, 0)


def write_metadata(path, metadata):
    path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def parse_hhmmss_for_date(date_str, time_str):
    return dt.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")


def date_key_from_log(log_meta):
    start_dt = log_meta.get("start_dt")
    if start_dt is not None:
        return start_dt.strftime("%Y%m%d")
    date_value = log_meta.get("values", {}).get("Date")
    if date_value:
        return str(date_value).replace("-", "")
    raise ValueError("Could not determine acquisition date from source log")


def filter_token_from_tiff_or_log(tiff_path, log_meta):
    text = str(tiff_path)
    for token in FILTER_COLORS:
        if token in text:
            return token
    values = log_meta.get("values", {})
    for key in ("Filter(Ang)", "Filter(nm)", "Filter"):
        value = values.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token.endswith(".0"):
            token = token[:-2]
        if token in FILTER_COLORS:
            return token
    raise ValueError(f"Could not determine TIFF color from path/log; expected one of {', '.join(FILTER_COLORS)}")


def source_token_from_tiff(tiff_path):
    return Path(tiff_path).stem.split("_", 1)[0]


def cache_paths_for_tiff(tiff_path, log_meta):
    date_key = date_key_from_log(log_meta)
    if date_key not in GIRAFF_CACHE_WINDOWS:
        supported = ", ".join(sorted(GIRAFF_CACHE_WINDOWS))
        raise ValueError(f"No GIRAFF cache window configured for {date_key}; supported dates: {supported}")

    start_token, end_token = GIRAFF_CACHE_WINDOWS[date_key]
    filter_token = filter_token_from_tiff_or_log(tiff_path, log_meta)
    color = FILTER_COLORS[filter_token]
    cache_dir = WORKSPACE_DIR / "images" / color / "VEE" / "GIRAFF"
    source_token = source_token_from_tiff(tiff_path)
    base = f"{source_token}_{start_token}_{end_token}"
    return {
        "color": color,
        "filter_token": filter_token,
        "start_dt": parse_hhmmss_for_date(date_key, start_token),
        "end_dt": parse_hhmmss_for_date(date_key, end_token),
        "output": cache_dir / f"{base}_uint16.npy",
        "metadata": cache_dir / f"{base}_metadata.json",
    }


def chunk_sort_key(path):
    stem = path.stem
    marker = "_X"
    if marker not in stem:
        return 1
    suffix = stem.rsplit(marker, 1)[1]
    return int(suffix) if suffix.isdigit() else 9999


def source_tiff_chunks(first_tiff):
    """Return acquisition chunk TIFFs in frame order."""
    first_tiff = Path(first_tiff)
    base_stem = first_tiff.stem
    chunks = [first_tiff, *first_tiff.parent.glob(f"{base_stem}_X*.tif")]
    chunks = [path for path in chunks if path.exists()]
    return sorted(set(chunks), key=chunk_sort_key)


def tiff_page_count(path):
    with tifffile.TiffFile(path) as tif:
        return len(tif.pages)


def source_chunk_metadata(first_tiff):
    chunks = []
    cursor = 0
    for path in source_tiff_chunks(first_tiff):
        page_count = tiff_page_count(path)
        chunks.append(
            {
                "path": path,
                "page_count": page_count,
                "start_index": cursor,
                "end_index": cursor + page_count - 1,
            }
        )
        cursor += page_count
    if not chunks:
        raise FileNotFoundError(f"No source TIFF chunks found for {first_tiff}")
    return chunks


def chunk_for_global_index(chunks, global_idx):
    for chunk in chunks:
        if chunk["start_index"] <= global_idx <= chunk["end_index"]:
            return chunk
    return None


def read_global_frame(chunks, global_idx):
    chunk = chunk_for_global_index(chunks, global_idx)
    if chunk is None:
        raise IndexError(f"Global frame index {global_idx} is outside the available TIFF chunks")
    with tifffile.TiffFile(chunk["path"]) as tif:
        frame = tif.pages[global_idx - chunk["start_index"]].asarray()
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame


def parse_args():
    ap = argparse.ArgumentParser(description="Cache GIRAFF SOK frames locally without loading the stack into RAM.")
    ap.add_argument("--tiff", type=Path, required=True, help="Source GIRAFF multipage TIFF")
    return ap.parse_args()


def main():
    args = parse_args()
    source_log = args.tiff.with_suffix(".log")
    if not args.tiff.exists():
        raise FileNotFoundError(f"Source TIFF not found: {args.tiff}")
    if not source_log.exists():
        raise FileNotFoundError(f"Source log not found: {source_log}")

    log_meta = parse_giraff_log(source_log)
    cache_info = cache_paths_for_tiff(args.tiff, log_meta)
    output_path = cache_info["output"]
    metadata_path = cache_info["metadata"]
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Cache output already exists: {output_path} or {metadata_path}")

    tiff_start_dt = log_meta["start_dt"]
    frame_interval = log_meta["frame_interval"]
    log_n_frames = log_meta["n_frames"]
    if tiff_start_dt is None or frame_interval is None or log_n_frames is None:
        raise ValueError(f"Incomplete timing metadata in source log: {source_log}")

    chunks = source_chunk_metadata(args.tiff)
    source_n_frames = sum(chunk["page_count"] for chunk in chunks)
    if source_n_frames != log_n_frames:
        print(f"Warning: log reports {log_n_frames} frames, TIFF chunks contain {source_n_frames} frames")

    cache_start_request = cache_info["start_dt"]
    cache_end_request = cache_info["end_dt"]
    if cache_end_request < cache_start_request:
        raise ValueError("Cache end time must be at or after start time")

    start_idx, end_idx = frame_bounds(tiff_start_dt, frame_interval, cache_start_request, cache_end_request)
    end_idx = min(end_idx, source_n_frames - 1)
    cache_frame_count = end_idx - start_idx + 1
    if cache_frame_count <= 0:
        raise ValueError("Requested cache window does not overlap the source TIFF.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = read_global_frame(chunks, start_idx)

    stack = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=first_frame.dtype,
        shape=(cache_frame_count, *first_frame.shape),
    )
    stack[0] = first_frame
    stack.flush()

    print(f"Writing {cache_frame_count} {cache_info['color']} frames to {output_path}")
    print(f"Source frames {start_idx + 1}..{end_idx + 1} of {source_n_frames}")
    print(f"Requested window {cache_start_request.isoformat()}..{cache_end_request.isoformat()}")
    print("Source chunks:")
    for chunk in chunks:
        print(f"  {chunk['path'].name}: global frames {chunk['start_index'] + 1}..{chunk['end_index'] + 1}")

    out_idx = 1
    for chunk in chunks:
        copy_start = max(start_idx + 1, chunk["start_index"])
        copy_end = min(end_idx, chunk["end_index"])
        if copy_start > copy_end:
            continue
        with tifffile.TiffFile(chunk["path"]) as tif:
            for source_idx in range(copy_start, copy_end + 1):
                frame = tif.pages[source_idx - chunk["start_index"]].asarray()
                if frame.ndim == 3:
                    frame = frame[:, :, 0]
                stack[out_idx] = frame
                if out_idx % 100 == 0 or out_idx == cache_frame_count - 1:
                    stack.flush()
                    print(f"  wrote {out_idx + 1}/{cache_frame_count} frames", flush=True)
                out_idx += 1

    stack.flush()
    cache_start_dt = tiff_start_dt + dt.timedelta(seconds=start_idx * frame_interval)
    cache_end_dt = tiff_start_dt + dt.timedelta(seconds=end_idx * frame_interval)
    metadata = {
        "format": "numpy_npy_uncompressed",
        "dtype": str(first_frame.dtype),
        "image_shape": list(first_frame.shape),
        "cache_shape": [cache_frame_count, *first_frame.shape],
        "source_tiff": str(args.tiff),
        "source_tiff_chunks": [
            {
                "path": str(chunk["path"]),
                "page_count": chunk["page_count"],
                "start_frame_index_zero_based": chunk["start_index"],
                "end_frame_index_zero_based": chunk["end_index"],
            }
            for chunk in chunks
        ],
        "source_log": str(source_log),
        "source_filter": cache_info["filter_token"],
        "color": cache_info["color"],
        "source_start_time": tiff_start_dt.isoformat(),
        "source_end_time": log_meta["end_dt"].isoformat() if log_meta["end_dt"] else None,
        "source_frame_count": source_n_frames,
        "source_log_frame_count": log_n_frames,
        "source_start_frame_index_zero_based": start_idx,
        "source_end_frame_index_zero_based": end_idx,
        "requested_start_time": cache_start_request.isoformat(),
        "requested_end_time": cache_end_request.isoformat(),
        "cache_start_time": cache_start_dt.isoformat(),
        "cache_end_time": cache_end_dt.isoformat(),
        "cache_frame_count": cache_frame_count,
        "frame_interval_seconds": frame_interval,
    }
    write_metadata(metadata_path, metadata)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
