# ADR 2: A Fixed Type Is Used as Given

**Status:** accepted on 2026-09-30, being implemented
([#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405))

## Context

Since v1.7.1, an explicit `dtype` has acted as a lower bound, not as the type to use:

- `createRaster(dtype="UInt16", noData=-1)` silently returns `Int16` (D10 in
  [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)).
- `createRaster(dtype="Byte")` and `rasterize(value=1, dtype="Byte")` return `Int8`, and `dtype="UInt16"`
  returns `Int16` (D3, M3).
- `warp(dtype="Float32")` on a `Float64` source returns `Float64` with a warning (D11).
- Only strings are accepted. `rasterize` and `createRaster` ignore `np.float32`, `float` or `np.dtype(...)`
  without notice, and `quickRaster` raises (D7). A bare GDAL constant such as `gdal.GDT_Float32` is ignored
  as well (M12).

The [C data types example](../../Examples/_07_configuration_options/_02_c_datatypes.ipynb) documents the
parameter as a "minimum data type". A minimum defeats the main reason to pass a type: to get a file that
other software expects. Up to v1.7.0, `gdalType()` used an explicit type as given.

A type that is fixed in advance may not hold the values of the operation. The input types alone give a safe
bound on these values, but not a tight one. Warping a `Byte` raster to a coarser grid with `sum` adds up
several source pixels per target pixel. From the type alone, GeoKit can only say that the sums *may* exceed
255. Whether they do depends on the values: a 0/1 mask summed over 4 × 4 pixels fits `Byte`, a raster of 200s
does not. Finding out costs a read of the input.

## Decision

`"preserve_input"` and an explicit type fix the type in advance. `"preserve_input"` fixes the promoted input
type, so no input is narrowed. An explicit type can be spelled as:

- GDAL names, with or without the `GDT_` prefix and in any case: `"Byte"`, `"GDT_Byte"`, `"byte"`
- NumPy names and types: `"uint8"`, `np.uint8`, `np.dtype("uint8")`
- the Python types `bool`, `int` and `float`

A fixed type is used exactly as given. GeoKit does not check whether the values of the operation fit it,
reads no values and issues no warning about them. If they do not fit, GDAL clips them (overflow), rounds
fractional results, or loses precision. A type narrower than the input is used as well:
`warp(Float64 raster, resampleAlg="near", dtype="Float32")` returns `Float32` without a warning. The
[docstring](adr_01_dtype_modes.md#docstring) of `dtype` states these risks.

GeoKit raises a `GeoKitDataTypeError` only for a request it cannot carry out:

- **A noData, fill or burn value that the type cannot store**, such as −1 in `UInt16`
  ([ADR 4](adr_04_nodata_fill_and_burn_values.md)). GeoKit could not write a valid raster with it.
- **Bare integer constants** such as `gdal.GDT_Float32`. GDAL and OGR constants share the same integers
  (`ogr.OFTReal` and `gdal.GDT_UInt16` are both 2). Accepting them would bring back the class of bug behind
  #396. The error message names the string spelling to use instead.
- Complex, string and object dtypes, which GDAL bands cannot store.
- A type that the installed GDAL does not support, such as `Int8` before GDAL 3.7.

These checks look only at what GeoKit already holds in memory: the spelling of `dtype`, the noData, fill and
burn values, and the types the installed GDAL supports. They never read pixel or attribute values, so they
cost nothing.

Every `GeoKitDataTypeError` names the function, the requested type and its range, the value that does not
fit, and how to fix the call:

```text
GeoKitDataTypeError: createRaster: noData=-1 cannot be stored in the requested dtype UInt16
(range 0 to 65535). Pass a noData value in that range, a type that holds it such as
dtype="Int32", or dtype="auto" to let GeoKit choose.
```

## Consequences

- An explicit type behaves as it did up to v1.7.0. Most of the changes are restorations.
- Under a fixed type, results can be silently wrong: sums can overflow and fractional results are rounded.
  This is the price of fixing the type, and the docstring of `dtype` states it. The caller who fixes the type
  takes the responsibility for these losses.
- Two new errors appear where GeoKit used to continue silently: a type that cannot hold the noData value, and
  a bare GDAL constant such as `dtype=gdal.GDT_Float32`. The release notes list both.
- The C data types example, which describes `dtype` as a minimum, has to be rewritten around the modes.
- `Extent.fit(unit, dtype=None)` uses a parameter of the same name for something else: rounding the bounds.
  Its docstring has to say so.

## Alternatives considered

- **Keep `dtype` as a lower bound.** This is the behaviour since v1.7.1. It defeats the purpose of the
  parameter and caused defects D3 and D10.
- **Accept bare integers, as `gdalType()` did.** This is the one part of the old behaviour that is not
  restored, because the GDAL and OGR constants overlap.
- **Check sums against the input's minimum and maximum.** GeoKit would read the input once and warn only if
  the sums can actually overflow. But the check would cost a read on calls that did not ask for one, for a
  type the caller has already chosen.
- **Warn from the type limits alone.** The type limits flag almost every sum into the input type as a
  possible overflow, even when the values are small. A warning that appears on every call gets ignored.
