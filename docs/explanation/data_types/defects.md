# Defect Catalogue

These are the data-type defects found while investigating issue
[#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396). Defects D1 to D29 come from a sweep of every
GeoKit function. Findings M1 to M11 come from the review of that sweep. The decisions that fix them are in the
[ADRs](index.md#decisions).

"Observed" is the behaviour of v1.9.1 with GDAL 3.10.3 and NumPy 2.5.1. Most cases were run in that
environment; a few were found by reading the code and are marked as such.

Most entries have a regression test in `test/test_13_dtypes_regression.py` whose name starts with the ID, for
example `test_D2_rasterize_constant_200`. Each test is marked `xfail(strict=True)` with the ID until the fix
lands.

Root causes R1 to R6 are explained on the [overview page](index.md#root-causes).

## A. Type representation

| ID | Call | Observed | Expected | Cause | ADR |
|---|---|---|---|---|---|
| D1 | `rasterize(value="field")` with an `Integer` field | `GeoKitCDataError("Unknown")` | an integer raster | R1 | [7](adr_07_numpy_dtype_inside.md), [9](adr_09_vector_field_types.md) |
| | the same with a `Real` field holding 7, 9.3, 1.392 (#396) | `Int16`; burns 7, 9, 1 | `Float64`; exact values | R1 | [7](adr_07_numpy_dtype_inside.md), [9](adr_09_vector_field_types.md) |
| D2 | `rasterize(value=200)` | `Int8`; burns 127 (every value from 128 to 255) | `Byte`; 200 | R2, R4 | [6](adr_06_resolve_once.md) |
| D3 | `createRaster` and `quickRaster` with `dtype="Byte"`, `"UInt16"` or `"UInt32"` | `Int8`, `Int16` or `Int32` | as requested | R2 | [2](adr_02_explicit_dtype.md), [6](adr_06_resolve_once.md) |
| D4 | `saveRasterAsTif` of a `Byte` raster | saved as `Int8`; scale and offset lost | an exact copy | R2, R4 | [6](adr_06_resolve_once.md) |
| D5 | `Extent.rasterMosaic` of a `Byte` source holding 200 | `Int8`; 127 | `Byte`; 200 | R4 | [3](adr_03_operation_rules.md), [6](adr_06_resolve_once.md) |
| D6 | `createRasterLike` of a `Float32` source, without `data` | `Int8`, although the docstring says the type is copied | `Float32` | R5 | [3](adr_03_operation_rules.md) |
| D7 | `createRaster`, `rasterize` and `quickRaster` with `dtype=np.float32`, `float` or `np.dtype(...)` | `createRaster` gives `Int8`; `rasterize` ignores it; `quickRaster` raises | honoured | R5 | [2](adr_02_explicit_dtype.md) |

## B. Promotion rules

| ID | Call | Observed | Expected | Cause | ADR |
|---|---|---|---|---|---|
| D8 | `rasterize` or `warp` of `Int32` or `Int64` data with `noData=np.nan` | `Float32`; 123 456 789 becomes 123 456 792 | `Float64`; exact | R2 | [4](adr_04_nodata_fill_and_burn_values.md) |
| D9 | a burn value of 0.1; integers above 2²⁴ mixed with floats | `Float32` | `Float64` | R2 | [4](adr_04_nodata_fill_and_burn_values.md) |
| D10 | `createRaster(dtype="UInt16", noData=-1)` | `Int16`, without a message | an error | R2, R5 | [2](adr_02_explicit_dtype.md) |
| D11 | `warp(dtype="Float32")` of a `Float64` source | `Float64` and a warning | `Float32` | R5 | [2](adr_02_explicit_dtype.md) |

## C. Operations that widen the value range

The expected values come from the same `gdal.Warp` call with `outputType=gdal.GDT_Float64`.

| ID | Call | Observed | Expected | Cause | ADR |
|---|---|---|---|---|---|
| D12 | `warp(average)` of a 0/1 `Byte` mask to a four times coarser grid | 0.75 stored as 1 | 0, 0.75 and 1 in a float type | R3 | [1](adr_01_dtype_modes.md), [3](adr_03_operation_rules.md) |
| D13 | `warp(cubic)` or `warp(lanczos)` upsampling a 0/255 `Byte` step | clipped to 0…255 | −18.9…272.4 (`cubic`), −29.7…284.6 (`lanczos`) | R3 | [1](adr_01_dtype_modes.md), [3](adr_03_operation_rules.md) |
| D14 | `warp(sum)` of 16 `Byte` pixels of 200 into one cell | 255 | 3200 | R3 | [3](adr_03_operation_rules.md) |
| D15 | `warp` of an `Int16` raster holding 1…400, without noData, from EPSG:3035 to EPSG:4326 | 202 of 484 pixels are 0 and not flagged | a warning | R3 | [4](adr_04_nodata_fill_and_burn_values.md) |
| D16 | `rasterize(value=100, add=True)` of two overlapping polygons | 127 | 200 | R3 | [3](adr_03_operation_rules.md) |
| D17 | `gradient` of a `UInt16` elevation model rising 1 m per 100 m pixel | 327.67 (wrap-around) | −0.01 | R6 | [8](adr_08_values_in_numpy.md) |
| D18 | `KernelProcessor(edgeValue=0)` on a float matrix | 0.5 becomes 0, 1.5 becomes 1 | unchanged values | R6 | [8](adr_08_values_in_numpy.md) |
| D19 | `combineSimilarRasters` of two `Byte` GeoTIFFs | `Int8` | `Byte` | R4 | [3](adr_03_operation_rules.md), [6](adr_06_resolve_once.md) |

## D. Crossing between raster and vector

| ID | Call | Observed | Expected | Cause | ADR |
|---|---|---|---|---|---|
| D20 | `createVector` with a `uint32` value of 3 000 000 000 | `Integer` field; 2 147 483 647 | `Integer64` field | R1 | [9](adr_09_vector_field_types.md) |
| | `createVector` with `uint64`, `float16` or pandas `boolean` columns | `String` fields, silently | numeric fields | R1 | [9](adr_09_vector_field_types.md) |
| D21 | `polygonizeRaster` of `UInt32` values of 3 000 000 000 | 2 147 483 647 | 3 000 000 000 | R1 | [9](adr_09_vector_field_types.md) |
| | `polygonizeRaster` of `Float32` values 1.7 and 2.4 | both become 2 and merge into one polygon | an error | R1 | [9](adr_09_vector_field_types.md) |
| D22 | `polygonizeMatrix` of a `uint32` matrix holding 3 000 000 000 | 2 147 483 647 | 3 000 000 000 | R1 | [9](adr_09_vector_field_types.md) |

## E. Values handled in NumPy

The raster for D23 and D24 is `Int16` with the values −9999, 100 and 200, noData −9999 and scale 0.1.

| ID | Call | Observed | Expected | Cause | ADR |
|---|---|---|---|---|---|
| D23 | `extractMatrix(autocorrect=True)`, `extractValues`, `interpolateValues` | −999.9 returned as data | NaN | R6 | [8](adr_08_values_in_numpy.md) |
| D24 | `rasterStats` | 3 observations, mean −323.3 | 2 observations, mean 15 | R6 | [8](adr_08_values_in_numpy.md) |
| D25 | `RegionMask.indicateValues(value="[0-10]", noData=-1)` or `noData=np.nan` on a float raster whose left half is NaN | all 100 pixels indicated | 50 indicated, 50 noData | R6 | [8](adr_08_values_in_numpy.md) |
| D26 | `RegionMask.applyMask` of a `uint8` array with `noData=np.int64(-1)` | 255, silently | −1 in an `int16` array | R6 | [8](adr_08_values_in_numpy.md) |

## F. Other

| ID | Call | Observed | Expected | Cause | ADR |
|---|---|---|---|---|---|
| D27 | `warp`, `checkSimilarRasters`, `combineSimilarRasters` (found by reading the code) | exact statistics over the whole source on every call; `.aux.xml` files next to the inputs | no read; only `"smallest"` reads its output | R3 | [3](adr_03_operation_rules.md) |
| D28 | `RegionMask.indicateValueToGeoms` and `mutateRaster(dtype="bool")` (found by reading the code) | `Int8` | `Byte` | R2 | [5](adr_05_signed_and_unsigned.md) |
| D29 | minor issues inside `MinimumCDataTypeHandler`, partly found by reading the code | none with the current GDAL and NumPy pins | removed with the handler | R2 | [7](adr_07_numpy_dtype_inside.md) |

## Findings of the review

| ID | Call | Observed | Expected | ADR |
|---|---|---|---|---|
| M1 | `warp(near)` of a `Byte` raster whose maximum is at most 127 | `Int8` | `Byte` | [5](adr_05_signed_and_unsigned.md), [6](adr_06_resolve_once.md) |
| M2 | `rasterize(value=40000)` and `rasterize(value=2**31)` | `Int16` with 32 767, `Int32` with 2 147 483 647 | `UInt16` and `UInt32` with the exact values | [6](adr_06_resolve_once.md) |
| M3 | `rasterize(value=1, dtype="Byte")` | `Int8` | `Byte` | [2](adr_02_explicit_dtype.md) |
| M4 | `RegionMask.createRaster()` and `RegionMask.rasterize()` | `Int8` raster, `int8` matrix | `Byte` raster, `uint8` matrix | [5](adr_05_signed_and_unsigned.md) |
| M5 | `rasterize(value="field")` with an `Integer64` field | correct by coincidence: reported as `UInt64`, then flipped back to `Int64` | `Int64` by design | [7](adr_07_numpy_dtype_inside.md) |
| M6 | `createRaster(noData=x)` without `fill` or `data` | filled with 0 (filled with noData, then overwritten) | filled with noData | – |
| M7 | the dtypes of the DataFrames returned by `extractFeatures` | not a defect: would change if they followed the field type | unchanged | [open question](index.md#open-questions) |
| M8 | `vectorInfo(...).attribute_data_types_str` | GDAL names of OGR constants: `"UInt16"` for `Real`, `"UInt64"` for `Integer64`, `"Unknown"` for `Integer` | OGR names | [7](adr_07_numpy_dtype_inside.md) |
| M9 | the statistics pass of D27 | not a defect: it never changes a result, so removing it is safe | – | [3](adr_03_operation_rules.md) |
| M10 | `RegionMask.indicateValues` | prints the memory usage on every call | no output | – |
| M11 | `mutateRaster` with a processor that returns `uint8` values of 200 or `uint16` values of 40 000 | `Int8` with 127, `Int16` with 32 767 | `Byte` and `UInt16` with the exact values | [6](adr_06_resolve_once.md) |

## Found on the way, not about data types

- `gradient(mode="ew")` raises `UnboundLocalError`; `mode="east-west"` works.
- `createRaster(output="name.tif")`, and so `saveRasterAsTif`, raises `PermissionError` for a bare file name.
  `"./name.tif"` works.
- `contours` prints every feature index.
- `Extent.clipRaster` never removes its `/vsimem/clip_*.tif` files.
- The docstring of `tileMosaic` documents `workingType` and `noData` parameters that do not exist.
