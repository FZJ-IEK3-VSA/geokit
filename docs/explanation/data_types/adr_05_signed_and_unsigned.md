# ADR 5: Signed and Unsigned Integer Types

**Status:** accepted on 2026-09-30, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

## Context

`MinimumCDataTypeHandler` prefers `Int8` over `Byte`, because `Int8` comes first in its list of candidates.
So these calls return `Int8` today:

- `createRaster()` without arguments, and `rasterize(value=1)`
- `RegionMask.createRaster()` and `RegionMask.rasterize()` (M4 in the [defect catalogue](defects.md))
- `mutateRaster(dtype="bool")` and `RegionMask.indicateValueToGeoms` (D28)
- `warp` with `near` of any `Byte` raster whose maximum is at most 127 (M1)

Values from 128 to 255 are then clipped to 127 (D2, D5, D19). Even where the values fit, `Int8` causes
trouble: `GDT_Int8` exists only since GDAL 3.7, and older builds of QGIS, ArcGIS and rasterio cannot read it.
Up to v1.7.0, all of these calls returned `Byte`.

`Int8` is still the smallest type for small negative values, such as a burn value of −1 or values from −100 to
100.

GeoKit chooses an integer width itself in three places: for a constant burn value of `rasterize`, for the
bound of `rasterize(add=True)` ([ADR 3](adr_03_operation_rules.md)), and when `"smallest"` shrinks its output
([ADR 1](adr_01_dtype_modes.md)). Often a signed and an unsigned type of the same width both hold the values,
for example `Int16` and `UInt16` for values from 0 to 1000. GeoKit needs a rule for that case.

At 16 bits and wider, every common tool reads both signed and unsigned types, so compatibility does not
decide. Three things favour the signed type:

- **Negative noData values fit.** The common sentinels are −9999 and −1. An `Int16` raster takes
  `noData=-9999` in a later call and stays `Int16`. A `UInt16` raster is widened to `Int32`, because NumPy
  promotes `UInt16` with `Int16` to `Int32`.
- **Arithmetic does not wrap around.** GeoKit's own arithmetic runs in `float64`
  ([ADR 8](adr_08_values_in_numpy.md)), but users compute on the arrays they get back. Unsigned subtraction
  wraps around: in `UInt16`, 100 − 102 is 65 534 (the cause of D17).
- **Differences of the data are signed**, for example gradients, anomalies or the change between two dates.

## Decision

- When GeoKit chooses or shrinks an integer type, it takes the first type in this order that holds every
  value: `Byte`, `Int8`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`.
- **At 8 bits, `Byte` comes first.** When both `Byte` and `Int8` can hold every value, GeoKit chooses `Byte`.
  `Int8` is chosen only when a value is negative and `Int8` is the smallest type that holds the values:
  `rasterize(value=-1)` gives `Int8`.
- **At 16, 32 and 64 bits, the signed type comes first.** The unsigned type is chosen only when the values do
  not fit the signed type: `rasterize(value=300)` gives `Int16`, `rasterize(value=40000)` gives `UInt16`.
- `bool` data and `dtype=bool` give `Byte`.
- Promoting an input type with a scalar follows NumPy: `noData=-1` on `Byte` data gives `Int16`, because the
  type must also hold 255.
- `createRaster()` without `dtype`, `data`, `fill` or `noData` gives `Byte`, as it did up to v1.7.0.

This order applies only where GeoKit chooses a width. Where the type comes from the input (the subset,
identity and union rules of [ADR 3](adr_03_operation_rules.md)), a `UInt16` input stays `UInt16`.

## Consequences

- A `Byte` input comes out as `Byte`.
- Masks, indicators and small counts are `Byte` and can be read by older GIS software.
- Small negative values give `Int8`, which GDAL before 3.7 and older QGIS, ArcGIS and rasterio builds cannot
  read. Users who need such a tool pass `dtype="Int16"`.
- Wider values that GeoKit sizes itself get a signed type unless only the unsigned type holds them.
- NumPy's `np.min_scalar_type` returns the unsigned type for non-negative values (`uint16` for 300), so
  GeoKit uses its own function to find the type of a scalar ([ADR 7](adr_07_numpy_dtype_inside.md)).
- Tests that expect `Int8` for non-negative values, such as the `RegionMask.createRaster()` test in
  `test/test_08_regionmask.py`, change to `Byte`.

## Alternatives considered

- **Never choose `Int8` automatically; use `Int16` for small negative values.** Every tool can read the
  result, but it takes twice the memory where `Int8` holds every value.
- **Prefer `Int8` over `Byte`**, as `MinimumCDataTypeHandler` does. A `Byte` input would come out as `Int8`,
  and masks would be unreadable for older software without any gain in size.
- **Unsigned first at every width** (`Byte`, `Int8`, `UInt16`, `Int16`, …). This matches NumPy's
  `np.min_scalar_type`. The extra positive range does not help: when both types are valid, the values fit
  either one. A later negative noData would force a wider type, and user arithmetic would wrap around.
