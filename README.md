# Alaska ASI Mapping

Utilities for mapping, sampling, and visualizing Alaska all-sky imager (ASI)
data for the GNEISS and GIRAFF sounding rocket missions.

This workspace is used for:
- generating mapped ASI images from mission-specific site sets
- overlaying rocket trajectories and receiver IPPs
- sampling brightness along rocket trajectories
- sampling brightness at receiver ionospheric pierce points
- plotting brightness-vs-time products
- building trajectory keograms

## Mission Support

Mission behavior is centralized in `core/missions.py`.

| Mission | Default date | Rockets | Default image sites | Receivers |
| --- | --- | --- | --- | --- |
| GNEISS | `20260210` | `397`, `398` | `ARV`, `VEE`, `BVR` | all receivers in `receivers.csv` |
| GIRAFF | `20250202` | `381` on `20250202`, `380` on `20250209` | `VEE` only | `VEE`, `TOO`, `PKR` |

GIRAFF uses one `main` trajectory. There are no left/right GIRAFF
trajectories. Selecting `--mission GIRAFF --date 20250209` automatically uses
rocket `380`; `20250202` uses rocket `381`.

GIRAFF supports green- and red-channel VEE imagery. Non-VEE image sites for
GIRAFF will stop with an argument error.

## Layout

```text
scripts/
  map_asi_archive.py
  map_asi_archive_series.py
  traj_brightness_series.py
  ipps_brightness_series.py
  trajectory_keogram.py
  compare_faraday_ipp_series.py
  core/
    missions.py
    receivers.py
    paths.py
    plot_norm.py
    plotting.py
    tiff_utils.py
    traj_utils.py
    ...

../trajectories/
  GNEISS/
    36397_GPS_Time_Export_01.csv
    36398_GPS_Time_Export_00.csv
  GIRAFF/
    36380_MAIN_PAYLOAD_GPS.xlsx
    36381_MAIN_PAYLOAD_GPS.xlsx

../mapped/
../images/
../receivers.csv
../starmap/
  green/
    ARV/
    BVR/
    PKR/
    VEE/
  red/
    ARV/
```

Important shared modules:
- `core/missions.py`: mission defaults, trajectory configs, receiver sets, output dirs
- `core/receivers.py`: receiver CSV loading and mission filtering
- `core/paths.py`: filesystem constants only
- `core/plot_norm.py`: shared and reference normalization helpers
- `core/tiff_utils.py`: TIFF/cache discovery and nearest-frame loading
- `core/traj_utils.py`: trajectory parsing, mapping, and lookup
- `core/plotting.py`: map plotting and trajectory/IPP overlays

## Normalization

`--shared-norm` is enabled by default in both map scripts. It normalizes all
selected sites against one reference frame, not independently per timestamp.
The flag is still accepted, but there is no `--no-shared-norm` option.

Reference normalization times:
- GNEISS: `core.constants.REFERENCE_NORMALIZATION_TIME`
- GIRAFF `20250202` / rocket `381`: `071130`
- GIRAFF `20250209` / rocket `380`: `083600`

`traj_brightness_series.py` also uses the GIRAFF reference normalization time
when writing and plotting GIRAFF brightness series.

## Default Time Ranges

Scripts that accept `--start` and `--end` now default to rocket-specific
mission windows. If a product includes both GNEISS rockets, the broadest
combined range is used.

| Rocket/product | Default start | Default end |
| --- | --- | --- |
| GNEISS 397 | `101900` | `102818` |
| GNEISS 398 | `101930` | `102848` |
| GNEISS combined products | `101900` | `102848` |
| GIRAFF 381 (`20250202`) | `070715` | `071636` |
| GIRAFF 380 (`20250209`) | `083501` | `084410` |

## Main Scripts

### `map_asi_archive.py`

Creates one mapped ASI image.

```bash
python3 map_asi_archive.py --time 102400 --sites ARV BVR VEE
```

GIRAFF 381:

```bash
python3 map_asi_archive.py \
  --mission GIRAFF \
  --date 20250202 \
  --time 071130
```

GIRAFF 380:

```bash
python3 map_asi_archive.py \
  --mission GIRAFF \
  --date 20250209 \
  --time 083600
```

Useful options:
- `--pretty`: use Cartopy plotting
- `--color green|red`: ASI channel
- `--bounds LON_MIN LON_MAX LAT_MIN LAT_MAX`: map bounds override
- `--colorbar-scale linear|log`
- `--colorbar-color monochromatic|viridis`
- `--plot-receivers`
- `--plot-ipps`
- `--plot-geodetic-traj`

### `map_asi_archive_series.py`

Runs `map_asi_archive.py` over a time range and moves each frame into a series
subdirectory.

```bash
python3 map_asi_archive_series.py \
  --date 20260210 \
  --step 10 \
  --sites ARV BVR VEE \
  --bounds -150 -142 65 69 \
  --colorbar-color monochromatic \
  --plot-ipps
```

GIRAFF 380 example:

```bash
python3 map_asi_archive_series.py \
  --mission GIRAFF \
  --date 20250209 \
  --step 10
```

### `traj_brightness_series.py`

Samples brightness at rocket trajectory positions through a time range and
writes a CSV plus a brightness plot unless `--no-plot` is used.

```bash
python3 traj_brightness_series.py \
  --sites ARV BVR VEE
```

GIRAFF 380 example:

```bash
python3 traj_brightness_series.py \
  --mission GIRAFF \
  --date 20250209
```

Trajectory CSV outputs include rocket geodetic columns:
- `<trajectory>_rocket_lat`
- `<trajectory>_rocket_lon`
- `<trajectory>_rocket_alt_km`

GIRAFF CSV outputs also include:
- `main_reference_norm_brightness`

The old receiver IPP sidecar CSV from `traj_brightness_series.py` has been
removed. Use `ipps_brightness_series.py` for receiver IPP brightness products.

### `ipps_brightness_series.py`

Samples brightness at receiver IPPs for each mission trajectory. It generates a
CSV and plot by default; pass `--no-csv` to plot from an existing CSV instead.

```bash
python3 ipps_brightness_series.py \
  --sites ARV BVR VEE PKR
```

GIRAFF 381 example:

```bash
python3 ipps_brightness_series.py \
  --mission GIRAFF \
  --date 20250202
```

IPP CSV outputs include per-receiver columns:
- `<rocket>_<receiver>_ipp_lat`
- `<rocket>_<receiver>_ipp_lon`
- `<rocket>_<receiver>_ipp_site`
- `<rocket>_<receiver>_ipp_percentile`
- `<rocket>_<receiver>_ipp_brightness`

For GIRAFF, receivers are filtered to `VEE`, `TOO`, and `PKR`.

### `trajectory_keogram.py`

Builds trajectory keograms. GNEISS produces two panels for rockets 397 and 398;
GIRAFF produces one panel for rocket 381 or 380, selected by date.

```bash
python3 trajectory_keogram.py \
  --sites ARV BVR VEE
```

GIRAFF 380 example:

```bash
python3 trajectory_keogram.py \
  --mission GIRAFF \
  --date 20250209
```

### `compare_faraday_ipp_series.py`

Compares GNEISS IPP brightness series against receiver `.mat` files.

```bash
python3 compare_faraday_ipp_series.py \
  --ipp-csv ../mapped/green/GNEISS/ipps_brightness_series_20260210_101900_102848_step0p05.csv \
  --receiver-dir ../receiver_data
```

## Inputs

Expected workspace inputs:
- `../trajectories/GNEISS/36397_GPS_Time_Export_01.csv`
- `../trajectories/GNEISS/36398_GPS_Time_Export_00.csv`
- `../trajectories/GIRAFF/36380_MAIN_PAYLOAD_GPS.xlsx`
- `../trajectories/GIRAFF/36381_MAIN_PAYLOAD_GPS.xlsx`
- `../receivers.csv`
- `../starmap/{color}/{site}/...`
- local TIFF archives under `../images/...` or the hard-coded GIRAFF source/cache paths in `core/tiff_utils.py`
- coastline assets under `coast/`

PFISR retrieval is optional. If PFISR download or the `resolvedvelocities`
package is unavailable, map generation continues without that overlay.

## GIRAFF TIFF/Cache Notes

GIRAFF imagery is loaded through VEE/SOK paths and local cache metadata in
`core/tiff_utils.py`.

Known GIRAFF dates:
- `20250202`: rocket `381`
- `20250209`: rocket `380`

The `20250209` cache window is expected to cover `08:31:40` to `08:44:15`.

## Output

Outputs are written under color, mission, and for GIRAFF rocket directories:

```text
../mapped/<color>/GNEISS/
../mapped/<color>/GIRAFF/380/
../mapped/<color>/GIRAFF/381/
```

Examples:
- mapped images: `GIRAFF_launch_red_VEE_20250209_083600.png`
- map series folders: `GIRAFF_launch_red_VEE_20250209_083501_to_084410_step_10/`
- trajectory brightness CSVs: `GIRAFF_brightness_vs_time_20250209_083501_084410_step0p05.csv`
- IPP brightness CSVs: `GIRAFF_ipps_brightness_series_20250209_083501_084410_step0p05.csv`
- GIRAFF keograms: `GIRAFF_trajectory_keogram_green_20250209_083501_084410.png`
- GNEISS keograms: `trajectory_keogram_green_20260210_101900_102848.png`
