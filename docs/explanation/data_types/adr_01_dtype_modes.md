# ADR 1: Three Automatic Modes for `dtype`

**Status:** accepted on 2026-09-30, being implemented
([#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405))

## Context

When the caller passes no type, GeoKit has to choose one. Users want different things from that choice:

- Most users want results that are exactly right ([goal G1](index.md#goals)), even if the type gets wider.
- Users who follow the GDAL convention want the type of their input back. GDAL, rasterio and QGIS keep the
  source type when they resample and round fractional results, so `0.75` becomes `1`.
- Users who write many large files want the smallest type that still holds every value. This was the idea
  behind `MinimumCDataTypeHandler`.

Fractional resampling of integer rasters has two reasonable answers. Float output is exact, because rounded
averages are wrong values. Keeping the integer type is the ecosystem norm, uses no extra memory, and keeps
categorical data categorical. No single default serves every user.

## Decision

The `dtype` parameter of every function that writes a raster takes one of these forms:

| Form | Meaning |
|---|---|
| `None` or `"auto"` | The output type holds every value the operation can produce. It is chosen from the input types and the operation ([ADR 3](adr_03_operation_rules.md)), never from the data. |
| `"preserve_input"` | The output type is the input type. |
| `"smallest"` | `"auto"`, then one read of the finished output and a shrink to the smallest type that stores every value and the noData value exactly. |
| an explicit type | used as given |

`"preserve_input"` and an explicit type fix the type in advance. GeoKit uses a fixed type as given and does not
check the values against it ([ADR 2](adr_02_explicit_dtype.md)).

### `"auto"`

1. Take the input types and promote them to one type, as NumPy does for arrays.
2. Apply the operation's rule ([ADR 3](adr_03_operation_rules.md)). For example, averaging a `Byte` raster
   gives `Float32` and summing gives `Float64`.
3. Widen the type until every noData, fill and burn value fits
   ([ADR 4](adr_04_nodata_fill_and_burn_values.md)).

GeoKit reads no values to do this, so the same call gives the same type for every input of the same types.

### `"smallest"`

GeoKit chooses the type as under `"auto"` and runs the operation. It then reads the output once and finds
the exact minimum and maximum, whether every value is a whole number, and whether every value is exact in
`float32`:

- Whole numbers, including whole numbers stored as floats, go to the first integer type in the order of
  [ADR 5](adr_05_signed_and_unsigned.md) that holds every value and the noData value.
- Other values go to `Float32` if every value is exact in `float32`, and otherwise stay `Float64`.
- Whole numbers with a NaN noData stay in the float type that `"auto"` chose, because no integer type can
  store NaN.

The shrink is lossless, so `"smallest"` never warns about it. `"smallest"` is the only mode that reads values,
and the read is the price the caller opts into:

- A raster on disk is read block by block. Its minimum and maximum come from `ComputeRasterMinMax`, which
  writes no `.aux.xml` file. Statistics stored in the metadata of an input are not used: they may be
  approximate or out of date, and the result would depend on whether an `.aux.xml` file happens to exist.
- `createRaster(data=...)` and `mutateRaster` reuse the array that is already in memory.
- `combineSimilarRasters` writes its output piece by piece, so it reads the inputs instead of the output.

### Docstring

Every function with a `dtype` parameter shares one docstring block. It states the risk of the fixed types:

```text
dtype : str, numpy.dtype, type or None, optional
    The data type of the output raster. By default (None or "auto"), GeoKit chooses a type that holds
    every value the operation can produce.

    - "auto": a type that holds every possible result, chosen from the input types and the operation.
    - "preserve_input": the type of the input. GeoKit does not check the results. Results that this
      type cannot hold are clipped (overflow), fractional results are rounded, and precision can be
      lost, without a warning.
    - "smallest": as "auto", then the smallest type that stores every result exactly. Reads the
      output once.
    - an explicit type, such as "Byte", "Float32" or np.uint16: used as given. GeoKit does not check
      the results, so the same losses as under "preserve_input" can occur without a warning.

    A noData, fill or burn value that the type cannot store raises a GeoKitDataTypeError.
```

### Worked examples

The type of the output band, per mode. "Error" is a `GeoKitDataTypeError`.

| Call | `"auto"` (default) | `"preserve_input"` | `"smallest"` |
|---|---|---|---|
| `createRaster()` | `Byte` | `Byte` | `Byte` |
| `createRaster(data=int64 array holding 0…5)` | `Int64` | `Int64` | `Byte` |
| `createRaster(data=uint8 array holding 1, noData=-1)` | `Int16` | error | `Int8` |
| `rasterize(value=1)` | `Byte` | `Byte` | `Byte` |
| `rasterize(value=200)` | `Byte` | `Byte` | `Byte` |
| `rasterize(value=40000)` | `UInt16` | `UInt16` | `UInt16` |
| `rasterize(value=-1)` | `Int8` | `Int8` | `Int8` |
| `rasterize(value=0.1)` | `Float64` | `Float64` | `Float64` (0.1 is not exact in `float32`) |
| `rasterize(value="Integer field")` | `Int32` | `Int32` | from the values, e.g. `Byte` |
| `rasterize(value="Real field")` | `Float64` | `Float64` | `Float32` if every value is exact in `float32` |
| `rasterize(value="Integer64 field", noData=np.nan)` | `Float64` and a warning | error | `Float64` and a warning |
| `rasterize(value=100, add=True)`, three features, at most two overlapping | `Int16` (3 × 100 = 300) | `Byte`; sums above 255 would be clipped, no warning | `Byte` (the true maximum is 200) |
| `warp(Byte, near)` | `Byte` | `Byte` | `Byte` |
| `warp(Byte, bilinear)` of a 0/1 mask | `Float32` | `Byte`; rounded without a warning (0.75 → 1) | `Float32` |
| `warp(Byte, average)` of a 0/1 mask | `Float32` | `Byte`; rounded without a warning | `Float32` |
| `warp(Int16, bilinear)` | `Float32` | `Int16`; rounded without a warning | `Float32` or narrower |
| `warp(Int32, average)` | `Float64` | `Int32`; rounded without a warning | `Float32` if exact |
| `warp(Int32, near, noData=np.nan)` | `Float64` | error | `Float64` |
| `warp(Byte, cubic)` of a 0/255 step | `Float32`; the overshoot (−18.9…272.4) is kept | `Byte`; clipped without a warning | `Float32` |
| `warp(Byte, sum)`, 16 pixels of 200 per cell | `Float64` | `Byte`; clipped to 255 without a warning | `Int16` (3200) |
| `warp(Int16)` with reprojection and no noData | `Int16` and a warning (created pixels are 0) | the same | the same |
| `mutateRaster(Byte source, processor returns floats 0…1)` | `Float64` | `Byte`; rounded without a warning | `Float32` |
| `rasterMosaic([Byte, Int16])` | `Int16` | `Int16` | from the values, e.g. `Byte` |
| `saveRasterAsTif(Float64 raster holding whole numbers)` | `Float64` | `Float64` | the smallest integer type |

## Consequences

- Default results are exact. The cost is memory where the old result was wrong: a `bilinear` warp of a `Byte`
  raster now needs four times the memory (`Float32`), one of an `Int32` raster twice (`Float64`).
- A default warp keeps the type of an integer raster, because the default resampling follows the data type
  ([ADR 10](adr_10_resampling_follows_the_data_type.md)). The float types of the fractional rule come from an
  explicit interpolating `resampleAlg`, such as `"bilinear"`.
- `"preserve_input"` gives the GDAL convention, which is also how `warp` behaved up to v1.7.0.
- `"smallest"` is the only mode whose type depends on the data. Users opt into it.

## Alternatives considered

- **Float for fractional resampling as the only behaviour, without modes.** Exact, but it offers no way back to
  the GDAL convention except an explicit type per call.
- **Keep the integer type as the only behaviour, without modes.** No memory increase and the ecosystem norm,
  but the default result is rounded.
- **`"smallest"` as the default, or statistics under `"auto"` to choose a narrower type**, as the handler
  does. Both make the type depend on the data and cost a full read on every default call, although the rules
  of [ADR 3](adr_03_operation_rules.md) already give a correct type without one. Users who want a type
  chosen from the values pass `"smallest"`.
