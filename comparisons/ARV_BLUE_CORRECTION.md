# ARV 4278 column correction

The shared implementation is `asispectralinversion/src/asi_shared/detector.py`
under lab317, imported through the shared `.venv` as `asi_shared.detector`.
It subtracts the offsets from `fix_ZOR_4278_2026.pro` on native 512×512
ARV/ZORBAS blue images in floating-point DN, before coadding, background
subtraction, denoising, or radiometric calibration. Raw source TIFFs are not
modified.

NumPy row/column assignments (zero-based, first row inclusive):

| Column | First row | Subtracted DN |
|---|---|---|
|281|153|77|
|282|153|47|
|342|146|47|
|343|146|429|
|344|146|10|

Integrated readers:
- Mapping: `core/tiff_utils.py`, `core/series_utils.py`, `core/select_tiffs.py`.
- Inversion: TIFF readers in `gneiss_rgb.ipynb`, `gneiss_rg.ipynb`,
  `minimalexample-new.ipynb`, and `comparisons/color_contours.py`.
- Star calibration: raw/coadded readers in `scrub.py` and the explicit
  image reader in `parameter_fitting.ipynb`.

New raw-image readers must call `correct_detector_frame` immediately after
reading an unrotated native frame. This is not a global patch of tifffile or
PIL: arbitrary direct reads of original TIFFs still return stored values.
Do not call this function again on an already processed in-memory array.
The selection exporter writes `arv_4278_column_offsets_v1` into corrected
TIFF metadata, which the shared reader recognizes to prevent double correction.

Restart existing notebook kernels to discard old imported code and cached
frames. Previously generated plots/HDF5 products must be regenerated to include
the correction. Existing source TIFFs and derived products are not rewritten
in bulk. The before/after diagnostic explicitly opts out of ingestion-time
correction, then applies the five offsets once for comparison.
