# Raster Operations Based on Extents

For many workflows, it may be easier to base your analysis on the extent of a region. In more complicated analyses, it may be useful to analyze data from one file based on the extent of another file. In these cases, GeoKit's Extent object could be useful.

You can use the Extent object to:

## [Create Rasters based on Extents](../../Examples/_04_raster_operations_based_on_extents/_1_create_raster_from_extent.ipynb)
- Create rasters from Extent objects with automatic pixel size computation

## [Clip Rasters](../../Examples/_04_raster_operations_based_on_extents/_2_clipping_rasters_to_an_extent.ipynb)
- Warp and clip rasters to specified Extents (via `warp()`)
- Support multiple resampling algorithms (e.g., bilinear interpolation)

## [Extent Creation and Transformation](../../Examples/_04_raster_operations_based_on_extents/_3_creating_and_transforming_extents.ipynb)

- You can create Extent objects from various sources 
    - From coordinates
    - Create Extents from raster files (via `fromRaster()`)
    - Create Extents from vector files, like shapefiles (via `fromVector()`)
    - Create Extents from in-memory geometries (via `fromGeom()`)
- Transform Extents to different coordinate systems (via `castTo()`)


## [Merging Overlapping Extents](../../Examples/_04_raster_operations_based_on_extents/_4_combine_overlapping_extents.ipynb)
- Combine overlapping Extents to create a bounding box covering both (via `+` operator)
- Convert Extents to OGR Geometry objects (via `.box` property) for visualization and spatial operations 


