# Working with Raster Data

GeoKit provides comprehensive functionality for working with raster data, which is geospatial information stored as grids of pixels (cells). Raster data is typically stored in formats like GeoTIFF (.tif) files and is commonly used to represent continuous phenomena such as elevation, temperature, land cover, and solar radiation.

## Key Raster Functionality

GeoKit enables you to perform the following operations on raster data:


### [Create Raster Data](../../Examples/_01_raster/_2_create_raster_data.ipynb)
GeoKit allows you to create raster datasets from scratch by specifying:
- Spatial reference system (SRS)
- Bounds (either as a tuple or Extent object)
- Pixel dimensions (width and height)
- Data matrix with values for each pixel

You can save rasters as GeoTIFF files and work with various data types including floats, integers, booleans, and NaN values.

###  [Visualize Raster Data](../../Examples/_01_raster/_1_visualize_raster_data.ipynb)

Visualizing raster data is an effective way to explore and analyze spatial datasets. GeoKit provides the `drawRaster()` function that leverages Matplotlib to create publication-quality visualizations. You can:
- Plot raster data from various formats including GeoTIFF
- Change the spatial reference system for display
- Overlay multiple rasters and vector geometries for comparison and analysis
- Extract and display spatial reference system information and coordinate bounds


### [Extract and Interpolate Data](../../Examples/_01_raster/_3_extract_data_from_raster.ipynb)
Extract values from rasters and interpolate data at locations that may fall between pixel centers. Supported interpolation methods include:
- Nearest neighbor
- Linear spline
- Cubic spline
- Average

###  [Convert Rasters to Vector Geometries](../../Examples/_01_raster/_4_polygonize_raster.ipynb)
The `polygonizeRaster()` function converts raster cells into polygon geometries, enabling seamless conversion between raster and vector data formats. This is useful for creating boundary definitions from raster data. GeoKits capabilities to work with geometries such as polygons is introduced in the [next section](_0_2_vector_introduction.md)

###  [Warp and Transform Rasters](../../Examples/_01_raster/_5_warp_raster.ipynb)
Transform rasters between different spatial reference systems and resample them using various algorithms. This functionality enables you to align rasters from different sources or prepare them for analysis with other geospatial data. 