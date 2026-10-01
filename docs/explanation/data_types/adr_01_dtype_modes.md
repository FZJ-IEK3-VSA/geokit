# ADR 1: Three Automatic Modes for `dtype`

**Status:** accepted on 2026-09-30, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

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
| `None` | the same as `"auto"` |
| `"auto"` | Accuracy is kept. The output type can hold every value the operation can produce. It is chosen from the input types and the operation ([ADR 3](adr_03_operation_rules.md)), never from the data. |
| `"preserve_input"` | The output type is the input type. GeoKit does not check the values against it. Values that this type cannot hold are clipped or rounded by GDAL, without a warning. |
| `"smallest"` | `"auto"`, then one read of the finished output and a shrink to the smallest type that stores every value and the noData value exactly. |
| an explicit type | used as given, without checks ([ADR 2](adr_02_explicit_dtype.md)) |

### `"auto"`

1. Take the input types and promote them to one type, as NumPy does for arrays.
2. Apply the operation's rule ([ADR 3](adr_03_operation_rules.md)). For example, averaging a `Byte` raster
   gives `Float32` and summing gives `Float64`.
3. Widen the type until every noData, fill and burn value fits
   ([ADR 4](adr_04_nodata_fill_and_burn_values.md)).

Where GeoKit chooses an integer width, it follows the order of [ADR 5](adr_05_signed_and_unsigned.md). If no
type can hold the values without loss, for example `Int64` data with a NaN noData, the result is `Float64`
with a warning.

### `"preserve_input"`

The output type is the promoted input type, so no input is narrowed. This is the same as passing that type
explicitly.

GeoKit does not check whether the values of the operation fit the type, reads no values and issues no
warning about them ([ADR 3](adr_03_operation_rules.md)). If the values do not fit, GDAL clips them
(overflow), rounds fractional results, or loses precision. The [docstring](#docstring) says so.

A noData, fill or burn value that the type cannot store at all, such as NaN in an integer type or −1 in
`Byte`, still raises a `GeoKitDataTypeError`. GeoKit could not write a valid raster with it.

### `"smallest"`

GeoKit chooses the type as under `"auto"` and runs the operation. It then reads the output once and finds
the exact minimum and maximum, whether every value is a whole number, and whether every value is exact in
`float32`:

- Whole numbers, including whole numbers stored as floats, go to the first integer type in `Byte`, `Int8`,
  `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64` that holds every value and the noData value.
- Other values go to `Float32` if every value is exact in `float32`, and otherwise stay `Float64`.
- Whole numbers with a NaN noData stay in the float type that `"auto"` chose, because no integer type can
  store NaN.

The shrink is lossless, so `"smallest"` never warns about it. The read is the price the caller opts into.
Large outputs on disk are read block by block. `createRaster(data=...)` and `mutateRaster` reuse the array
that is already in memory. `combineSimilarRasters` writes its output piece by piece, so it reads the inputs
instead of the output.

### Warnings and errors

| Situation | `"auto"` | `"preserve_input"` | `"smallest"` | explicit type |
|---|---|---|---|---|
| fractional results from integer input | widen to float | rounded, no warning | widen, then shrink | rounded if the type is an integer type, no warning |
| a sum can exceed the input type | `Float64` | clipped, no warning | `Float64`, then shrink | clipped if the type is too narrow, no warning |
| NaN noData on integer input | `Float32` or `Float64` | error | as `"auto"` | error if the type is an integer type |
| noData or fill outside the input type (−1 on `Byte`) | widen (`Int16`) | error | widen, then shrink | error |
| `Int64` or `UInt64` data with a NaN noData | `Float64` and a warning | error | `Float64` and a warning | as requested, no warning |
| explicit type narrower than the input | – | – | – | used as given, no warning |
| `warp` creates pixels and no noData is set | one warning; the pixels are 0 | the same | the same | the same |
| a bare integer, or a complex, string or object dtype | error | error | error | error |

`GeoKitDataTypeError` is the only data-type error. It replaces `GeoKitCDataError`
([ADR 7](adr_07_numpy_dtype_inside.md)) and, like every GeoKit error, is a subclass of `GeoKitError`.
`GeoKitDataTypeWarning` is a subclass of `UserWarning`, so existing warning filters keep working. Each
warning names the function, the input type, the chosen type and how to avoid the warning.

A user who turns the checks off with `geokit.dtypes.set_options(checks=False)` gets none of these warnings
([ADR 10](adr_10_turning_checks_off.md)). The types and the errors stay the same.

### Docstring

Every function with a `dtype` parameter shares one docstring block. It states the risk of the two modes
without checks:

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

The type of the output band, per mode:

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
| `warp(Byte, cubic)` of a 0/255 step | `Float32`; the overshoot (−18.9…272.4) is kept | `Byte`; clipped without a warning | `Float32` |
| `warp(Byte, sum)`, 16 pixels of 200 per cell | `Float64` | `Byte`; clipped to 255 without a warning | `Int16` (3200) |
| `warp(Int16)` with reprojection and no noData | `Int16` and a warning (created pixels are 0) | the same | the same |
| `mutateRaster(Byte source, processor returns floats 0…1)` | `Float64` | `Byte`; rounded without a warning | `Float32` |
| `rasterMosaic([Byte, Int16])` | `Int16` | `Int16` | from the values, e.g. `Byte` |
| `saveRasterAsTif(Float64 raster holding whole numbers)` | `Float64` | `Float64` | the smallest integer type |

An explicit type replaces the mode. `warp(Float64 raster, resampleAlg="near", dtype="Float32")` returns
`Float32` without a warning, although the requested type is narrower than the input and loses precision.

## Consequences

- Default results are exact. The cost is memory where the old result was wrong: a `bilinear` warp of a `Byte`
  raster now needs four times the memory (`Float32`), one of an `Int32` raster twice (`Float64`).
- `warp` and `RegionMask.warp` default to `resampleAlg="bilinear"`. Under `"auto"`, the default call on a
  categorical `Byte` raster therefore returns `Float32`. For categorical data, pass `resampleAlg="near"`
  (the type stays `Byte`) or `dtype="preserve_input"`. Whether the default resampler should change is an
  [open question](index.md#open-questions).
- `"preserve_input"` gives the GDAL convention, which is also how `warp` behaved up to v1.7.0. As then,
  GeoKit does not warn when values are lost.
- Under `"preserve_input"` and with an explicit type, results can be silently wrong. This is the exception to
  [goal G1](index.md#goals), and the docstring of `dtype` states it.
- `"smallest"` is the only mode that reads values and the only mode whose type depends on the data. Users opt
  into both.

## Alternatives considered

- **Float for fractional resampling as the only behaviour, without modes.** Exact, but it offers no way back to
  the GDAL convention except an explicit type per call.
- **Keep the integer type as the only behaviour, without modes.** No memory increase and the ecosystem norm,
  but the default result is rounded.
- **Float output together with `resampleAlg="near"` as the default for integer inputs.** Float would then
  appear only when the user asks for interpolation. This changes the values of default calls and is left
  open.
- **`"smallest"` as the default.** It makes types depend on the data ([goal G4](index.md#goals)) and costs a
  read on every call ([goal G5](index.md#goals)).
