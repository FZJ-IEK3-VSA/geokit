# Raster Operations Based on Extents

The Extent object represents geographic extents of an area and exposes useful methods which depend on those extents.

## Extent Creation and Manipulation

### From Coordinates
See [_3_creating_and_transforming_extents.ipynb](../../Examples/_04_combine_data_from_multiple_files/_3_creating_and_transforming_extents.ipynb)
- Create Extent objects from coordinate bounds (minX, minY, maxX, maxY)
- Extract boundaries in different formats

### From Data Sources
See [_3_creating_and_transforming_extents.ipynb](../../Examples/_04_combine_data_from_multiple_files/_3_creating_and_transforming_extents.ipynb)
- Create Extents from raster files (via `fromRaster()`)
- Create Extents from vector files, like shapefiles (via `fromVector()`)
- Create Extents from in-memory geometries (via `fromGeom()`)

### Extent Transformation
See [_3_creating_and_transforming_extents.ipynb](../../Examples/_04_combine_data_from_multiple_files/_3_creating_and_transforming_extents.ipynb)
- Transform Extents to different coordinate systems (via `castTo()`)
- Pad Extents by a percentage or fixed distance (via `pad()`)
- Fit Extents to grid resolutions (via `fit()`)

## Raster Operations with Extents

### Raster Creation
See [_1_create_raster_from_extent.ipynb](../../Examples/_04_combine_data_from_multiple_files/_1_create_raster_from_extent.ipynb)
- Create rasters from Extent objects with automatic pixel size computation
- Generate physical TIFF files from Extent and data matrices

### Raster Clipping
See [_2_clipping_rasters_to_an_extent.ipynb](../../Examples/_04_combine_data_from_multiple_files/_2_clipping_rasters_to_an_extent.ipynb)
- Warp and clip rasters to specified Extents (via `warp()`)
- Support multiple resampling algorithms (e.g., bilinear interpolation)

## Extent Combination

### Merging Overlapping Extents
See [_4_combine_overlapping_extents.ipynb](../../Examples/_04_combine_data_from_multiple_files/_4_combine_overlapping_extents.ipynb)
- Combine overlapping Extents to create a bounding box covering both (via `+` operator)
- Convert Extents to OGR Geometry objects (via `.box` property) for visualization and spatial operations 