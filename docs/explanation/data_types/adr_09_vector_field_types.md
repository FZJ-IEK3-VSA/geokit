# ADR 9: Field Types Between Raster and Vector

**Status:** accepted on 2026-09-30, being implemented
([#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405))

## Context

OGR stores attributes in typed fields: `Integer` (32-bit signed), `Integer64`, `Real` (64-bit float),
`String` and a few others. OGR has no unsigned and no 8-bit or 16-bit field types; optional subtypes such as
`Boolean`, `Int16` and `Float32` narrow a field further. GeoKit crosses between these fields and raster or
NumPy types in several places, and loses values there (entries from the defect catalogue in
[#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)):

- `createVector` writes a `uint32` column into an `Integer` field, so 3 000 000 000 becomes 2 147 483 647.
  `uint64`, `float16` and pandas `boolean` columns silently become `String` fields (D20).
- `polygonizeRaster` always uses an `Integer` field. `UInt32` values above 2³¹ are clipped, and a float raster
  is truncated to integers, so pixels of 1.7 and 2.4 both become 2 and merge into one polygon (D21).
- `polygonizeMatrix` always uses an `Int32` raster, with the same clipping (D22).
- In the other direction, `rasterize(value="field")` reads the field type with the wrong vocabulary (D1,
  [ADR 7](adr_07_numpy_dtype_inside.md)).

## Decision

From a dtype to an OGR field (`to_ogr_field`), used by `createVector`, `mutateVector`, `polygonizeRaster` and
`polygonizeMatrix`:

| dtype | OGR field |
|---|---|
| `bool` | `Integer` |
| `int8`, `uint8`, `int16`, `uint16`, `int32` | `Integer` |
| `uint32`, `int64` | `Integer64` |
| `uint64` | `Integer64`, with an error for values above 2⁶³ − 1 |
| `float16`, `float32`, `float64` | `Real` |
| `str`, `object` | `String` |

Missing values are written as NULL.

From an OGR field to a dtype (`from_ogr_field`), used by `vectorInfo` and `rasterize`:

| OGR field | dtype |
|---|---|
| `Integer` | `int32` (`int16` with the `Int16` subtype, `uint8` with the `Boolean` subtype) |
| `Integer64` | `int64` |
| `Real` | `float64` (`float32` with the `Float32` subtype) |
| any other | `None`; `rasterize` raises a clear error for such a field |

`polygonizeRaster` takes the field type from the band type. Float rasters raise a `GeoKitDataTypeError`; the
docstring already says the function expects an integer raster. `polygonizeMatrix` creates its temporary band
in the dtype of the matrix.

The `dtype` modes of [ADR 1](adr_01_dtype_modes.md) do not apply on the vector side. The field type follows
from the dtype.

## Consequences

- Integer values up to 2⁶³ − 1 survive the way from a raster or a DataFrame into a vector.
- A float raster passed to `polygonizeRaster` now raises instead of returning merged polygons.
- Writing OGR subtypes for narrow dtypes (`Boolean`, `Int16`, `Float32`) is optional. GeoPackage and
  FlatGeobuf keep subtypes; shapefiles ignore them.
- The dtypes of the DataFrames returned by `extractFeatures` do not change; pandas still infers them. See the
  [open questions](index.md#open-questions).

## Alternatives considered

- **`uint64` columns as `Real` fields, or an error for every `uint64` column.** `Real` loses precision above
  2⁵³, and an error would reject columns whose values fit. `Integer64` with a range check keeps every value
  that fits and rejects only the rest.
- **Polygonize float rasters with `gdal.FPolygonize` and a `Real` field.** This gives exact polygons, but many
  more of them for continuous data. It is an [open question](index.md#open-questions).
