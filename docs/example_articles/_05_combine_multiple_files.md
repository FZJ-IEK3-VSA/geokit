# Combine Multiple Files

GeoKit's RegionMask object provides a powerful way to combine and analyze raster and vector data within a specific region of interest. RegionMasks can be created from shapefiles, geometries, or extent objects combined with Boolean numpy arrays, making them flexible tools for spatial analysis workflows.

You can use the RegionMask object to:

## [Create and Inspect a RegionMask](../../Examples/_05_combine_multiple_files/_01_setup_regionmask_aachen.ipynb)
- Create RegionMask objects from various sources
    - From shapefiles (via `.load()`)
    - From geometry objects such as boxes or polygons (via `.load()`)
    - From extent objects combined with Boolean numpy arrays (via `.fromMask()`)
- Query RegionMask properties such as extent, pixel size, and area

## [Warp Rasters to RegionMask Context](../../Examples/_05_combine_multiple_files/_02_warp_raster_to_region_mask_context.ipynb)
- Align and warp raster data to a RegionMask's coordinate system and extent
- Filter raster data to extract only values within the region boundary
- Extract statistical information from the masked raster data

## [Create Rasters from RegionMask](../../Examples/_05_combine_multiple_files/_03_create_raster_from_regionmask.ipynb)
- Create raster files from numpy arrays using a RegionMask as the spatial extent and coordinate reference system
- Automatically handle raster geospatial metadata (CRS, projection, pixel alignment)

## [Extract Vector Features within a Region](../../Examples/_05_combine_multiple_files/_04_extract_vector_features_within_region.ipynb)
- Extract vector features (such as point locations or geometries) that fall within a RegionMask's boundaries (via `.extractFeatures()`)
- Filter vector data based on spatial location

## [Indicate Features with Optional Buffering](../../Examples/_05_combine_multiple_files/_05_indicate_features_and_buffer.ipynb)
- Create binary raster representations of vector features within a RegionMask (via `.indicateFeatures()`)
- Apply buffer zones around features to expand their influence area
- Use '1' to indicate feature presence and '0' for absence

## [Indicate Raster Values with Thresholds and Buffering](../../Examples/_05_combine_multiple_files/_06_indicate_values_threshold_and_buffer.ipynb)
- Identify regions within a raster that contain specific values or fall within a value range (via `.indicateValues()`)
- Create binary masks indicating raster value thresholds
- Apply buffer zones around identified value regions for expanded analysis areas