#!/usr/bin/env python3
"""Extract a GIRAFF SOK time window into a local disk-backed frame cache."""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import tifffile

from core.tiff_utils import GIRAFF_CACHE_DIR, parse_giraff_log


DEFAULT_SOURCE_TIFF = Path("/Volumes/LynchK/GIRAFF/GIRAFF/SOK_250209_5577/ut07/SOK250209_07495931_16bit.tif")
DEFAULT_SOURCE_LOG = DEFAULT_SOURCE_TIFF.with_suffix(".log")
DEFAULT_START = "2025-02-09T08:31:40"
DEFAULT_END = "2025-02-09T08:44:15"
DEFAULT_CACHE_NPY_PATH = GIRAFF_CACHE_DIR / "SOK250209_083140_084415_uint16.npy"
DEFAULT_CACHE_METADATA_PATH = GIRAFF_CACHE_DIR / "SOK250209_083140_084415_metadata.json"


def frame_bounds(tiff_start_dt, frame_interval, start_dt, end_dt):
    """Return inclusive source-frame bounds covering start_dt..end_dt."""
    start_idx = int(np.floor((start_dt - tiff_start_dt).total_seconds() / frame_interval))
    end_idx = int(np.ceil((end_dt - tiff_start_dt).total_seconds() / frame_interval))
    return max(start_idx, 0), max(end_idx, 0)


def write_metadata(path, metadata):
    path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def parse_datetime_arg(value):
    value = value.strip()
    if "T" in value:
        return dt.datetime.fromisoformat(value)
    return dt.datetime.strptime(value, "%Y%m%d%H%M%S")


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
    ap.add_argument("--tiff", type=Path, default=DEFAULT_SOURCE_TIFF, help="Source GIRAFF multipage TIFF")
    ap.add_argument("--log", type=Path, default=DEFAULT_SOURCE_LOG, help="Source GIRAFF sidecar log")
    ap.add_argument("--start", default=DEFAULT_START, help="Cache start time, ISO or YYYYMMDDHHMMSS")
    ap.add_argument("--end", default=DEFAULT_END, help="Cache end time, ISO or YYYYMMDDHHMMSS")
    ap.add_argument("--output", type=Path, default=DEFAULT_CACHE_NPY_PATH, help="Output uncompressed .npy stack")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_CACHE_METADATA_PATH, help="Output JSON metadata")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.tiff.exists():
        raise FileNotFoundError(f"Source TIFF not found: {args.tiff}")
    if not args.log.exists():
        raise FileNotFoundError(f"Source log not found: {args.log}")
    if (args.output.exists() or args.metadata.exists()) and not args.overwrite:
        raise FileExistsError("Cache output already exists. Pass --overwrite to replace it.")

    log_meta = parse_giraff_log(args.log)
    tiff_start_dt = log_meta["start_dt"]
    frame_interval = log_meta["frame_interval"]
    log_n_frames = log_meta["n_frames"]
    if tiff_start_dt is None or frame_interval is None or log_n_frames is None:
        raise ValueError(f"Incomplete timing metadata in source log: {args.log}")

    chunks = source_chunk_metadata(args.tiff)
    source_n_frames = sum(chunk["page_count"] for chunk in chunks)
    if source_n_frames != log_n_frames:
        print(f"Warning: log reports {log_n_frames} frames, TIFF chunks contain {source_n_frames} frames")

    cache_start_request = parse_datetime_arg(args.start)
    cache_end_request = parse_datetime_arg(args.end)
    if cache_end_request < cache_start_request:
        raise ValueError("--end must be at or after --start")

    start_idx, end_idx = frame_bounds(tiff_start_dt, frame_interval, cache_start_request, cache_end_request)
    end_idx = min(end_idx, source_n_frames - 1)
    cache_frame_count = end_idx - start_idx + 1
    if cache_frame_count <= 0:
        raise ValueError("Requested cache window does not overlap the source TIFF.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_frame = read_global_frame(chunks, start_idx)

    stack = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=first_frame.dtype,
        shape=(cache_frame_count, *first_frame.shape),
    )
    stack[0] = first_frame
    stack.flush()

    print(f"Writing {cache_frame_count} frames to {args.output}")
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
        "source_log": str(args.log),
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
    write_metadata(args.metadata, metadata)
    print(f"Wrote metadata to {args.metadata}")


if __name__ == "__main__":
    main()
