# ADR 4: noData, Fill and Burn Values Must Fit

**Status:** accepted on 2026-09-30, being implemented
([#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405))

## Context

GeoKit writes some values itself: the noData value, the fill value of a new raster, and the burn value of
`rasterize`. The output type has to hold them. The handler gets this wrong in three ways:

- It widens an integer type with a NaN noData to `Float32`, whatever the integer width. `float32` holds whole
  numbers exactly only up to 2²⁴, so integer IDs change: 123 456 789 becomes 123 456 792 (D8 in
  [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)).
- It picks the float width by range, not by precision. `np.min_scalar_type(0.1)` is `float16`, so a burn value
  of 0.1 ends up in `Float32` (D9).
- An explicit type that cannot hold the noData value is widened without notice (D10).

A related case concerns pixels that GeoKit does not write on purpose. A reprojection, a cutline or bounds
larger than the source create pixels outside the source data. GDAL fills them with 0 and sets no noData
value, so they look like data (D15). This is standard GDAL behaviour.

## Decision

- **Under `"auto"`, the type widens until every scalar fits.** For each scalar that does not fit, the type is
  promoted with the type of the scalar: the smallest integer type that holds an integer scalar (in the order of
  [ADR 5](adr_05_signed_and_unsigned.md)), `Float32` for NaN, and `Float64` for any other non-integral float.
  NumPy's promotion then picks a type that holds both.
- **`Int64` or `UInt64` data with a NaN noData** has no lossless type. The result is `Float64` with a
  `GeoKitDataTypeWarning` that whole numbers are exact only up to 2⁵³. The same holds for `UInt64` data with
  a negative scalar.
- **Under `"preserve_input"` or with an explicit type**, a scalar that does not fit raises a
  `GeoKitDataTypeError` ([ADR 2](adr_02_explicit_dtype.md)).
- **Pixels created by `warp`** keep GDAL's value 0. If no noData value is set, GeoKit warns once, in every
  mode, and suggests passing `noData=`. GeoKit does not choose a noData value by itself.

Besides these two warnings, `polygonizeRaster` and `extractFeatures` warn ([ADR 9](adr_09_vector_field_types.md)).
Each warning names the function, the input type, the chosen type and how to avoid the warning.
`GeoKitDataTypeWarning` is a `UserWarning`, so Python's warning filters turn it off, for example
`warnings.filterwarnings("ignore", category=GeoKitDataTypeWarning)`.

Examples of the widening under `"auto"`:

| Input type | Scalar | Type under `"auto"` |
|---|---|---|
| `Byte` | 255 | `Byte` |
| `Byte` | −1 | `Int16` |
| `UInt16` | −1 | `Int32` |
| `Byte` or `Int16` | NaN | `Float32` |
| `Int32` | NaN | `Float64` |
| `Int64` | NaN | `Float64` and a warning |
| `Float32` | −9999 | `Float32` |
| none (constant burn value) | 0.1 | `Float64` |

## Consequences

- Integer data with a NaN noData keeps its values. `Int32` data takes twice the memory it took in `Float32`,
  but only where `Float32` gave wrong values.
- pandas uses `int64` for integer columns by default. So `createVector(dataframe)` followed by
  `rasterize(value="id", noData=np.nan)` is common. It keeps working, with a warning.
- A fractional burn value such as 0.1 gives `Float64`. Users who prefer the smaller type pass
  `dtype="Float32"` or `dtype="smallest"`.
- A reprojection without noData still produces zeros, but no longer silently.

## Alternatives considered

- **Raise an error for `Int64` data with a NaN noData.** This would break the common
  pandas path above, although the IDs are small in nearly every real case. NumPy makes the same choice:
  `np.promote_types(np.int64, np.float32)` is `float64`.
- **`Float32` for fractional scalars unless more precision is needed.** `float32` keeps about 7
  significant digits, which is what most GIS tools use and what is enough in most cases. It was not chosen
  for `"auto"`, whose promise is exact values: 0.1 is not exact in `float32`.
- **Set a noData value for created pixels automatically**: NaN for floats, and for
  integers either an unused value or a type widened by one step. This was rejected:
    - An unused value can only be found from the data or from stored statistics, so the result would depend
      on the data ([goal G1](index.md#goals)).
    - Widening by one step would turn every reprojection of an `Int16` raster into `Int32`, a type change of a
      correct result.
    - A noData value the user did not ask for changes later results, for example of
      `extractMatrix(autocorrect=True)`, `rasterStats` and the noData check in `checkSimilarRasters`.
    - Filling with 0 is what other tools do, and users know it.
