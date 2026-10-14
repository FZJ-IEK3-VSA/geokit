# ADR 3: The Type Follows the Operation

## Context

`MinimumCDataTypeHandler` chooses a type from a few sample numbers: noData, fill value, burn value, and the
minimum and maximum of the input. It never sees the operation. So it never widens the type when the operation
widens the values:

- A sum of 16 pixels of 200 is stored as 255 in a `Byte` raster (D14 in
  [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)).
- `rasterize(add=True)` of two overlapping features of 100 stores 127 instead of 200 (D16).
- Cubic resampling overshoots the input range, and the overshoot is clipped (D13).
- Averaging a 0/1 mask stores 0.75 as 1 (D12).
- `Extent.rasterMosaic` takes the type of its first source, so the values of a later `Int16` source are
  clipped (M13).

To get the minimum and maximum, `warp`, `checkSimilarRasters` and `combineSimilarRasters` compute exact
statistics over the whole source on every call. This is an extra full read, and GDAL writes `.aux.xml` files
next to the inputs (D27). Because `dtype` acted as a lower bound, statistics could only widen the type, and the
minimum and maximum of a band always fit its own type. So the read never changed a result (M9).

## Decision

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

These types hold every value the operation can produce, for any input of the given types. Overflow is not
possible: a sum goes to `Float64`, an average or an interpolation stays within the range of its inputs, and
the overshoot of `cubic` and `lanczos` is small compared with the range of `Float32`.

## Consequences

- No rule needs the values, so `ComputeStatistics` is no longer called. `warp`, `checkSimilarRasters` and
  `combineSimilarRasters` read no statistics and write no `.aux.xml` files.
- Type limits can be wider than needed. `bilinear` on an `Int32` raster gives `Float64` even if every value
  is small. Users who want the narrowest lossless type pass `dtype="smallest"`
  ([ADR 1](adr_01_dtype_modes.md)).
- A new GeoKit function that writes a raster has to declare its rule.

## Alternatives considered

- **Full range propagation.** Each operation would transform a value range `[lo, hi]`, with measured
  overshoot factors for `cubic` and `lanczos`. For `sum`, GeoKit would compute the number of source pixels per
  target pixel from the transformed footprint of the output grid. That number depends on the coordinate
  systems and the latitude: summing a 0.00833° grid into 1 km UTM cells collects about 1.2 source pixels per
  cell at the equator, 1.8 at 50° N and 3.4 at 70° N. This is more machinery than the defects need. `sum` is
  rare and its results are fractional anyway, because GDAL weights partial overlaps. So `Float64` is never
  wrong under `"auto"`, and the rule table above gives the same guarantees with much less code.
