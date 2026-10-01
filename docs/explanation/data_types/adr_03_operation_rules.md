# ADR 3: The Type Follows the Operation; Statistics Depend on the Mode

## Context

`MinimumCDataTypeHandler` chooses a type from a few sample numbers: noData, fill value, burn value, and the
minimum and maximum of the input. It never sees the operation. So it never widens the type when the operation
widens the values:

- A sum of 16 pixels of 200 is stored as 255 in a `Byte` raster (D14 in
  [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)).
- `rasterize(add=True)` of two overlapping features of 100 stores 127 instead of 200 (D16).
- Cubic resampling overshoots the input range, and the overshoot is clipped (D13).
- Averaging a 0/1 mask stores 0.75 as 1 (D12).

To get the minimum and maximum, `warp`, `checkSimilarRasters` and `combineSimilarRasters` compute exact
statistics over the whole source on every call. This is an extra full read, and GDAL writes `.aux.xml` files
next to the inputs (D27). Because `dtype` acted as a lower bound, statistics could only widen the type, and the
minimum and maximum of a band always fit its own type. So the read never changed a result (M9).

The input types alone give a safe bound on the values an operation can produce, but not a tight one. This
matters when the output type is fixed in advance, as under `"preserve_input"` or with an explicit type.
Warping a `Byte` raster to a coarser grid with `sum` adds up several source pixels per target pixel. From the
type alone, GeoKit can only say that the sums *may* exceed 255. Whether they do depends on the values: a 0/1
mask summed over 4 × 4 pixels fits `Byte`, a raster of 200s does not. Finding out costs a read of the input.

## Decision

### Rules

Every operation declares one rule that describes its effect on the value range. Under `"auto"`, the output
type follows from the input types and this rule. Then it is widened for the scalars GeoKit writes
([ADR 4](adr_04_nodata_fill_and_burn_values.md)).

| Rule | Operations | Type under `"auto"` |
|---|---|---|
| subset | `warp` with `near`, `mode`, `min`, `max`, `med`, `q1`, `q3` | the input type |
| identity | `createRaster(data=...)`, `createRasterLike`, `saveRasterAsTif`, `rasterize` | the input type |
| union | `Extent.rasterMosaic`, `Extent.tileMosaic`, `combineSimilarRasters` | the promotion of all input types |
| fractional | `warp` with `bilinear`, `average`, `cubic`, `cubicspline`, `lanczos`, `rms` | `Float32` if the input type is exact in `float32` (`Byte`, `Int16`, `UInt16`, `Float32`), otherwise `Float64` |
| sum | `warp` with `sum` | `Float64` |
| sum of burns | `rasterize(add=True)` | the narrowest integer type that holds the feature count times the burn value; for an attribute field, times the range of the field type |
| derivative | `gradient` | `Float64` |
| user function | `mutateRaster`, `combineSimilarRasters(combiningFunc=...)` | the dtype of the array the function returns |

The input type of `rasterize` is the type of the attribute field, or, for a constant burn value, the smallest
type that holds that constant. A constant is the one case where every value written is known in advance, so
it needs no read.

Under `"auto"`, these types hold every value the operation can produce, for any input of the given types.
Overflow is not possible: a sum goes to `Float64`, an average or an interpolation stays within the range of
its inputs, and the overshoot of `cubic` and `lanczos` is small compared with the range of `Float32`.

### When GeoKit reads values

Whether GeoKit reads values depends on the mode ([ADR 1](adr_01_dtype_modes.md)):

| Mode | What GeoKit reads | Why |
|---|---|---|
| `"auto"` (default) | nothing | The rules above already give a type that holds every possible value. Reading could only make the type narrower, and that is what `"smallest"` is for. |
| `"preserve_input"` | nothing | The input fixes the type. GeoKit does not check whether the values of the operation fit it. |
| an explicit type | nothing | The caller fixes the type. GeoKit does not check whether the values of the operation fit it. |
| `"smallest"` | the finished output, once | to choose the narrowest lossless type |

Under `"preserve_input"` and with an explicit type, GeoKit runs no checks of the values against the type and
issues no warning about them. If the operation produces values that the type cannot hold, GDAL clips them
(overflow), rounds fractional results, or loses precision. The docstring of `dtype` states these risks
([ADR 1](adr_01_dtype_modes.md#docstring)). Choosing a fixed type moves the responsibility for these losses to
the caller.

`ComputeStatistics` is no longer called. Where `"smallest"` needs the minimum and maximum of a raster on disk,
GeoKit computes them with `ComputeRasterMinMax`, which writes no `.aux.xml` file. Statistics stored in the
metadata of an input are not used. They may be approximate or out of date, and the result would depend on
whether an `.aux.xml` file happens to exist next to the input.

## Consequences

- The default mode never reads values. The same call gives the same type for every input of the same type
  ([goal G4](index.md#goals)).
- `warp` no longer computes statistics on every call and no longer writes `.aux.xml` files.
- Only `"smallest"` reads values. Under `"preserve_input"` and with an explicit type, sums can overflow and
  fractional results are rounded without a warning. The docstring says so.
- Type limits can be wider than needed. `bilinear` on an `Int32` raster gives `Float64` even if every value
  is small. Users who want the narrowest lossless type pass `dtype="smallest"`.
- A new GeoKit function that writes a raster has to declare its rule.

## Alternatives considered

- **Check sums against the input's minimum and maximum under `"preserve_input"` and explicit types.** GeoKit
  would read the input once and warn only if the sums can actually overflow. But a caller who fixes the type
  takes the responsibility for it, and the check would cost a read on calls that did not ask for one. The
  docstring states the risk instead.
- **Warn from the type limits alone under `"preserve_input"` and explicit types.** The type limits flag
  almost every sum into the input type as a possible overflow, even when the values are small. A warning that
  appears on every call gets ignored.
- **Read statistics under `"auto"` too, to choose a narrower type**, as the handler does. Reading is not
  needed for a correct result under `"auto"`. It would cost a full read on every default call
  ([goal G5](index.md#goals)) and make the type depend on the data ([goal G4](index.md#goals)). Users who want
  a type chosen from the values use `"smallest"`.
- **Full range propagation.** Each operation would transform a value range `[lo, hi]`, with measured
  overshoot factors for `cubic` and `lanczos`. For `sum`, GeoKit would compute the number of source pixels per
  target pixel from the transformed footprint of the output grid. That number depends on the coordinate
  systems and the latitude: summing a 0.00833° grid into 1 km UTM cells collects about 1.2 source pixels per
  cell at the equator, 1.8 at 50° N and 3.4 at 70° N. This is more machinery than the defects need. `sum` is
  rare and its results are fractional anyway, because GDAL weights partial overlaps. So `Float64` is never
  wrong under `"auto"`, and the rule table above gives the same guarantees with much less code.
