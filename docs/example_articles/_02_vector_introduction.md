## Vector datasets and file structure

Vector datasets store geometries together with attribute data in a table-like structure.
Each row represents a geographic feature, while one column contains the geometry and the remaining columns store descriptive attributes.

Vector data represents  geographic features and is commonly stored in formats such as Shapefiles or GeoPackages.
These formats define how geometries, attributes, and coordinate reference systems are stored.


![Vector Files Overview](vector_files_overview.svg)

### General structure of vector data

The figure above summarizes how vector data is organized on disk. A vector dataset is made up of:

- One or more geometry files that store the actual shapes (points, lines, polygons).
- Attribute tables that hold descriptive fields for each feature.
- A coordinate reference system definition, which tells software how to interpret coordinates.
- Optional index or metadata files that speed up access and record provenance.

Together, these components form a single logical dataset: a table of features with geometry plus attributes. The exact set of files depends on the format (e.g., Shapefile uses multiple sidecar files, while GeoPackage stores everything in one container), but the logical structure is the same.

### GeoKit capabilities (examples)

Using the examples in [Examples/_03_combine_data_from_multiple_files](Examples/_03_combine_data_from_multiple_files), GeoKit supports:

- Creating and transforming extents to normalize spatial coverage across multiple datasets.
- Deriving rasters from extents to build aligned grid products.
- Merging overlapping extents to produce consolidated coverage.
- Clipping rasters to an extent to focus analysis on a target area.
- Generating region masks to combine and filter data from multiple sources.

See the full workflow examples in [Examples/_03_combine_data_from_multiple_files](Examples/_03_combine_data_from_multiple_files).