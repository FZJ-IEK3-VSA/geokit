# Introduction to GeoKit

GeoKit enables you to work efficiently with both **vector** and **raster** data, and is especially powerful when these data types need to be used **together**. The main areas of functionality include:

### Geometry Operations
Create and manipulate vector geometries (points, lines, polygons), transform them between coordinate systems, and perform spatial operations like buffering, intersection, and union. [Learn more about geometries](./_03_00_geometries.md)

### Raster Operations
Create, read, visualize, warp, resample, and analyze raster datasets (GeoTIFF and other formats). Extract values at specific locations using various interpolation methods, and convert between raster and vector representations. [Learn more about raster data](./_01_raster_overview.md)

### Vector Operations
Read, analyze, and manipulate vector datasets (Shapefiles, GeoPackages) with their associated attribute tables. Filter, buffer, and clip vector data based on spatial relationships. [Learn more about vector data](./_02_vector_introduction.md)

### Extent and Region Operations
Create and manipulate Extent objects to define spatial regions, warp and clip rasters to specific extents, and combine multiple extents. [Learn more about extent operations](./_04_raster_operations_based_on_extents.md)

### Multi-File Integration
Use the powerful RegionMask object to combine and analyze raster and vector data within a specific region of interest. Create binary representations of spatial features, apply buffer zones, and extract masked data from multiple sources. [Learn more about combining multiple files](./_05_combine_multiple_files.md)

