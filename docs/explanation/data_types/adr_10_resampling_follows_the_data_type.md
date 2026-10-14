# ADR 10: The Default Resampling Follows the Data Type

**Status:** accepted on 2026-10-02, being implemented
([#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405))

## Context

`warp`, `RegionMask.warp`, `Extent.mutateRaster` and `RegionMask.mutateRaster` default to
`resampleAlg="bilinear"`, and `warpLike` and `Extent.warp` take the default of `warp`. Bilinear interpolation
mixes neighbouring values. That is right for continuous data such as elevation and wrong for categorical data
such as land cover, whose values are class codes:

- A `Byte` raster with the classes 10 and 30, warped from 100 m onto 50 m pixels, gets the values 15 and 25
  along the border between the classes, classes that do not exist. In v1.9.1 the result is `Int8`; under
  `"auto"` it is `Float32`, because bilinear gives fractional results ([ADR 3](adr_03_operation_rules.md);
  M17 in [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)).

GDAL's tools resample with nearest neighbour unless told otherwise. GDAL's COG driver chooses by the data for its
overviews: "For paletted images, NEAREST is used by default, otherwise it is CUBIC."

## Decision

`resampleAlg` takes `"auto"`, which is resolved from the data type of the source band:

- `"near"` for integer rasters;
- `"bilinear"` for float rasters.

`"auto"` is the default of `warp`, so `warpLike` and `Extent.warp` follow, and of `RegionMask.warp`,
`Extent.mutateRaster` and `RegionMask.mutateRaster`. The other defaults stay:

- `RegionMask.indicateValues` keeps `"bilinear"`. It warps a 0/1 indication, and its `threshold` compares the
  fractions that bilinear produces.
- `RegionMask.contoursFromRaster` keeps `"bilinear"`, because contours need a continuous surface. It takes
  `resampleAlg` as a parameter.
- `drawRaster` keeps `"med"`.
- `Extent.rasterMosaic` keeps `"near"`. An explicit `"auto"` is resolved from the promoted data type of all
  sources.

Under dtype `"auto"`, `"auto"` keeps the data type: `near` keeps integer types, and `bilinear` keeps `Float32`
and `Float64` ([ADR 3](adr_03_operation_rules.md)). A noData value that does not fit can still widen it
([ADR 4](adr_04_nodata_fill_and_burn_values.md)).

## Consequences

- Categorical integer rasters keep their classes and their data type in default calls.
- Continuous integer rasters, such as `Int16` elevation models, are no longer interpolated by default. Pass
  `resampleAlg="bilinear"`, which gives `Float32` or `Float64` under dtype `"auto"`.
- Float rasters are resampled as before.

## Alternatives considered

- **Keep `"bilinear"`.** Categorical rasters get classes that do not exist, and every default warp of an integer
  raster becomes `Float32` or `Float64`.
- **`"near"` for every raster, as GDAL's tools do.** This changes the results for float rasters, which hold
  continuous data in most cases and for which bilinear is right.
- **Choose from the values, for example from the number of distinct values.** That needs a read of the data,
  which no default call does ([ADR 3](adr_03_operation_rules.md)).
