# Alaska ASI Mapping

Utilities for mapping, sampling, and visualizing Alaska all-sky imager (ASI) data for the GNEISS sounding rocket mission.

This repository may be used for:
- generating mapped ASI images from ARV, BVR, VEE, and optionally PKR
- sampling brightness along the two rocket trajectories
- plotting derived brightness-vs-time products
- building trajectory keograms

## Layout

```text
asi_mapping_scripts/
  map_asi_archive.py
  traj_brightness_series.py
  trajectory_keogram.py
  coast/
    coastlat.txt
    coastlon.txt
  trajectories/
    36397_GPS_Time_Export_01.csv
    36398_GPS_Time_Export_00.csv
  core/
    ...
```

- Top-level scripts are the main entrypoints.
- `core/` contains shared code used by those scripts.
- `trajectories/` contains the two GPS-export trajectory files.
- `coast/` contains the coastline assets used by the fast map plot.

## Main Scripts

`map_asi_archive.py`
- Loads ASI images, maps them geographically, applies overlap masks, overlays the rocket trajectories, and writes a map image.
- Supports fast and Cartopy-based plotting.

`traj_brightness_series.py`
- Steps through a time range and samples left/right rocket brightness from the selected sites.
- Writes a CSV dataset with brightness and percentile metadata.
- Also writes a scatter plot PNG of left/right brightness versus time unless `--no-plot` is used.

`map_asi_archive_series.py`
- Repeats `map_asi_archive.py` over a requested time range.
- Useful for batch-producing mapped PNGs without re-pasting an inline Python loop.

`trajectory_keogram.py`
- Builds a keogram along each rocket trajectory.
- X-axis is UTC time and y-axis is flight time since launch.

## Shared Modules

The most important shared modules in `core/` are:

- `core/skymaps.py`: load site skymaps and geographic projections
- `core/masks.py`: build overlap masks between sites
- `core/plotting.py`: fast and pretty map plotting
- `core/plot_norm.py`: image normalization helpers
- `core/traj_utils.py`: trajectory parsing and lookup
- `core/tiff_utils.py`: TIFF discovery and frame loading
- `core/fetch_url.py`: PKR PNG URL lookup
- `core/remote_data.py`: PKR image download and PFISR retrieval
- `core/paths.py`: shared paths for static assets

## Inputs

This code expects:

- trajectory CSVs in `trajectories/`
- coastline files in `coast/`
- skymap assets in the expected `../starmaps/...` directories
- local TIFF archives discoverable by `core/tiff_utils.py`

PFISR retrieval is optional. If PFISR download or the `resolvedvelocities` package is unavailable, the map scripts still run without that overlay.

## Common Usage

Create a mapped ASI image:

```bash
python3 map_asi_archive.py --time 102400 --sites ARV BVR VEE
```

Pretty map:

```bash
python3 map_asi_archive.py --time 102400 --sites ARV BVR VEE --pretty
```

Red-channel map:

```bash
python3 map_asi_archive.py --time 102400 --sites ARV BVR VEE --color red
```

Batch map a time range:

```bash
python3 map_asi_archive_series.py \
  --date 20260210 \
  --start 101900.0 \
  --end 102900.0 \
  --step 10 \
  --sites ARV BVR VEE \
  --bounds -150 -142 65 69 \
  --colorbar-color monochromatic \
  --plot-ipps
```

Build a brightness-vs-time CSV and PNG plot:

```bash
python3 traj_brightness_series.py \
  --start 101830 \
  --end 102900 \
  --sites ARV BVR VEE
```

Build trajectory keograms:

```bash
python3 trajectory_keogram.py \
  --start 101800 \
  --end 103100 \
  --sites ARV BVR VEE
```

## Output

Outputs are typically written under color and mission directories:

- `../mapped/<color>/GNEISS/`
- `../mapped/<color>/GIRAFF/380/`
- `../mapped/<color>/GIRAFF/381/`

These include:

- mapped PNGs from `map_asi_archive.py`
- brightness-vs-time CSVs and PNG plots from `traj_brightness_series.py`
- keogram PNGs from `trajectory_keogram.py`
