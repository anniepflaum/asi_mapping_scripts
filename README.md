# Alaska ASI Mapping

This repo holds scripts for mapping ASI FoV on Alaska, primarily to support the GNEISS sounding rocket campaign.

## Fast Mode
```
python map_asi_realtime_restructure.py
```
This puts all maps on a cartesian lat/lon grid, do the image will be warped, especially at higher latitudes.

## Pretty Mode
```
python map_asi_realtime_restructure.py --pretty
```
This plots FoVs on a Albers Equal Area projection with cartopy.  It prduces an image that is more physically realisitc, but takes 3-5 minutes to run.
