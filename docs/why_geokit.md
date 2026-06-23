# Why ETHOS.GeoKit?

ETHOS.GeoKit's central contribution is a single, high-level interface that treats
raster and vector data as first-class citizens within a **shared spatial
context** — the [`Extent`](example_articles/_04_raster_operations_based_on_extents.md)
and [`RegionMask`](example_articles/_05_combine_multiple_files.md) objects.
Operations that would otherwise require juggling two libraries (and a lot of
CRS/extent/resolution bookkeeping) become a few consistent, reproducible calls.

Other common geospatial Python libraries, such as GeoPandas or Rasterio, specialise in a single data type. Real-world
analyses — especially in energy-system and environmental modelling — constantly
need to combine raster **and** vector data: clip a raster to an administrative
boundary, sample raster values at point locations, rasterize polygons, or mask a
grid by a region of interest. Doing this across separate, single-purpose
libraries means manually reconciling coordinate reference systems, resolutions,
extents, and nodata handling at every step.


## Built on, and complementary to, established tools

GeoKit is a high-level layer on top of [GDAL/OGR](https://gdal.org/), which provides more functionality but also requires much more boilerplate code. GeoKit requires substantially less code for the same analysis of combined vector and raster data. It automates many operations around coordinate reference system (CRS), extent, resolution, and nodata bookkeeping for you. To showcase the boilerplate reduction, we implemented one
identical North Sea offshore-wind workflow three ways — with GeoKit, with pure
GDAL/OGR, and with GeoPandas + Rasterio — and count the lines reproducibly:

| Implementation | Data processing | Plotting | Total | Processing vs GeoKit | Total vs GeoKit |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ETHOS.GeoKit** | **28** | 54 | 82 | 1.00× | 1.00× |
| GeoPandas + Rasterio | 82 | 30 | 112 | 2.93× | 1.37× |
| Pure GDAL/OGR | 102 | 63 | 165 | 3.64× | 2.01× |

You can explore the examples yourself:

- [Offshore turbine depths — the GeoKit example](Examples/_06_application_examples/_1_determine_offshore_turbine_depths.ipynb): the worked workflow combining vector sea boundaries, a bathymetry raster, and turbine point locations.
- The same workflow implemented in [pure GDAL/OGR](Examples/_06_application_examples/_2_determine_offshore_turbine_depths_gdal.ipynb) and in [GeoPandas + Rasterio](Examples/_06_application_examples/_3_determine_offshore_turbine_depths_geopandas_rasterio.ipynb) for comparison.
- [Introduction to ETHOS.GeoKit](example_articles/_00_introduction.md) and the full set of [examples and guides](Examples/index.md).


## It also handles the individual data types

GeoKit fully supports the individual building blocks on their own, so you can use
it for pure raster work, pure vector work, or — where it shines — both at once:

- **Geometry / vector operations** — create, reproject, buffer, filter, intersect, and rasterize. [Learn more](example_articles/_02_vector_introduction.md)
- **Raster operations** — read, warp/resample, clip, polygonize, and sample at points. [Learn more](example_articles/_01_raster_overview.md)
- **Region-based integration** — combine many raster and vector sources within one region via `RegionMask`. [Learn more](example_articles/_05_combine_multiple_files.md)


However, if you know that you only need to deal with raster or vector data, the following single-purpose libraries are also worth considering:

| Tool | Scope |
| --- | --- |
| [GeoPandas](https://geopandas.org/) | Vector data with a pandas-style API |
| [Rasterio](https://rasterio.readthedocs.io/) | Raster I/O and array operations |





