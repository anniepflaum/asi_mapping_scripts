# Shared ASI background policy

All production ASI mapping, brightness series, keograms, RGB/RG inversion, and
color/starmap comparisons use `asi_shared.background` from the sibling
`asispectralinversion` package (installed in the lab317 shared environment).

Select **finite native elevation <= 0 degrees**, then retain pixels whose
Euclidean distance inside that region is **strictly greater than 50 detector
pixels**. Nonfinite elevations are excluded. This is independent of the 22-degree
mapping/display cutoff. Zero-valued starmap regions are a proxy, not proof of a
physical horizon. Do not enlarge the sample region when too few pixels remain.

Fit the same DN histogram Gaussian as `background_subtraction_diagnostic.ipynb`,
with the median fallback for rejected/failed/constant-data fits. Subtract that
single scalar before the existing exposure/calibration conversion. Detector
correction, coadding, denoising and exposure conventions are otherwise unchanged.

Legacy HDF5 preparation reads the native `Elevation` dataset, or accepts an
explicit elevation array. A geographic NaN mask cannot substitute for elevation;
old datasets lacking native elevation now require that input. Legacy experimental
`patches`/`corners` algorithms remain explicit opt-ins, not production defaults.

Restart existing notebook kernels and rerun from setup. Previously generated
images/HDF5 files are unchanged and must be regenerated to use the new policy.
