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

Paths are configurable through `ASI_WORKSPACE_ROOT`, `ASI_IMAGE_ROOT`, and
`ASI_OUTPUT_ROOT`; `LAB317_OUTPUT_ROOT` provides the shared output fallback.
See `config/example.env`. The image archive now defaults to
`$LAB317_DATA_ROOT/raw/asi/images`, shared starmaps and receiver metadata to
`$LAB317_DATA_ROOT/reference`, trajectories to
`$LAB317_DATA_ROOT/raw/rocket/trajectories`, and mapped products to
`$LAB317_OUTPUT_ROOT/asi-mapping/mapped`. Checksum catalogs under
`data-manifests/` preserve the verified migration baseline.

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

$LAB317_DATA_ROOT/raw/rocket/trajectories/
  GNEISS/
    36397_GPS_Time_Export_01.csv
    36398_GPS_Time_Export_00.csv
  GIRAFF/
    36380_MAIN_PAYLOAD_GPS.xlsx
    36381_MAIN_PAYLOAD_GPS.xlsx

$LAB317_OUTPUT_ROOT/asi-mapping/mapped/
$LAB317_DATA_ROOT/raw/asi/images/
$LAB317_DATA_ROOT/reference/asi-mapping/receivers.csv
$LAB317_DATA_ROOT/reference/starmaps/
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
- `core/plot_norm.py`: shared image color-limit helpers
- `core/tiff_utils.py`: TIFF/cache discovery and nearest-frame loading
- `core/traj_utils.py`: trajectory parsing, mapping, and lookup
- `core/plotting.py`: map plotting and trajectory/IPP overlays

## Normalization

`--shared-norm` is enabled by default in both map scripts. It uses one color
scale across the selected sites from a fixed calibrated reference frame.

Reference normalization times:
- GNEISS: `core.constants.REFERENCE_NORMALIZATION_TIME` (`102400.0`)
- GIRAFF `20250202` / rocket `381`: `071130`
- GIRAFF `20250209` / rocket `380`: `083600`

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
writes a calibrated HDF5 file plus a brightness plot unless `--no-plot` is used.

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

Trajectory HDF5 files store `time_iso`, optional `time_since_tg_s`, and one
`rockets/<rocket>` group per trajectory. Each rocket group contains geodetic
coordinates, magnetic latitude where available, and mapped-altitude brightness
datasets. Brightness is calibrated in Rayleighs using a per-frame buffered-corner
background, the channel exposure time, and the site calibration factor. File
attributes record the units, calibration, selected cameras, time range, and cadence.

Use `--plot-existing` to redraw the PNG from an existing HDF5 file without
reprocessing camera frames. `traj_brightness_series.py` does not write CSV output.

The old receiver IPP sidecar CSV from `traj_brightness_series.py` has been
removed. Use `ipps_brightness_series.py` for receiver IPP brightness products.

### `ipps_brightness_series.py`

Samples brightness at receiver IPPs for each mission trajectory. It generates a
CSV and plot by default; pass `--no-csv` to plot from an existing CSV instead.
Brightness values are calibrated Rayleighs, with calibration provenance stored in
the `brightness_units` and `calibration_json` columns. The PKR camera is skipped
because no PKR Rayleigh calibration is configured.

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
The plotted and HDF5 brightness arrays are calibrated Rayleighs. The HDF5 file
stores units and calibration provenance alongside the redrawable arrays.

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
  --ipp-csv "$LAB317_OUTPUT_ROOT/asi-mapping/mapped/green/GNEISS/ipps_brightness_series_20260210_101900_102848_step0p05.csv" \
  --receiver-dir "$LAB317_DATA_ROOT/raw/receiver-data/asi-mapping"
```

## Inputs

Expected shared inputs:
- `$LAB317_DATA_ROOT/raw/rocket/trajectories/GNEISS/36397_GPS_Time_Export_01.csv`
- `$LAB317_DATA_ROOT/raw/rocket/trajectories/GNEISS/36398_GPS_Time_Export_00.csv`
- `$LAB317_DATA_ROOT/raw/rocket/trajectories/GIRAFF/36380_MAIN_PAYLOAD_GPS.xlsx`
- `$LAB317_DATA_ROOT/raw/rocket/trajectories/GIRAFF/36381_MAIN_PAYLOAD_GPS.xlsx`
- `$LAB317_DATA_ROOT/reference/asi-mapping/receivers.csv`
- `$LAB317_DATA_ROOT/reference/starmaps/{color}/{site}/...`
- local TIFF archives under `$LAB317_DATA_ROOT/raw/asi/images/...` or a direct `ASI_IMAGE_ROOT` override
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
$LAB317_OUTPUT_ROOT/asi-mapping/mapped/<color>/GNEISS/
$LAB317_OUTPUT_ROOT/asi-mapping/mapped/<color>/GIRAFF/380/
$LAB317_OUTPUT_ROOT/asi-mapping/mapped/<color>/GIRAFF/381/
```

Examples:
- mapped images: `GIRAFF_launch_red_VEE_20250209_083600.png`
- map series folders: `GIRAFF_launch_red_VEE_20250209_083501_to_084410_step_10/`
- trajectory brightness HDF5: `GIRAFF_brightness_vs_time_20250209_083501_084410_step0p05.h5`
- IPP brightness CSVs: `GIRAFF_ipps_brightness_series_20250209_083501_084410_step0p05.csv`
- GIRAFF keograms: `GIRAFF_trajectory_keogram_green_20250209_083501_084410.png`
- GNEISS keograms: `trajectory_keogram_green_20260210_101900_102848.png`
