# ADR 2: An Explicit `dtype` Is an Exact Override

**Status:** accepted on 2026-09-30, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

## Context

Since v1.7.1, an explicit `dtype` has acted as a lower bound, not as the type to use:

- `createRaster(dtype="UInt16", noData=-1)` silently returns `Int16` (D10 in the [defect catalogue](defects.md)).
- `createRaster(dtype="Byte")` and `rasterize(value=1, dtype="Byte")` return `Int8`, and `dtype="UInt16"`
  returns `Int16` (D3, M3).
- `warp(dtype="Float32")` on a `Float64` source returns `Float64` with a warning (D11).
- Only strings are accepted. `rasterize` and `createRaster` ignore `np.float32`, `float` or `np.dtype(...)`
  without notice, and `quickRaster` raises (D7).

The [C data types example](../../Examples/_07_configuration_options/_02_c_datatypes.ipynb) documents the
parameter as a "minimum data type". A minimum defeats the main reason to pass a type: to get a file that
other software expects. Up to v1.7.0, `gdalType()` used an explicit type as given.

## Decision

`dtype` accepts the mode strings of [ADR 1](adr_01_dtype_modes.md) and these spellings of a type:

- GDAL names, with or without the `GDT_` prefix and in any case: `"Byte"`, `"GDT_Byte"`, `"byte"`
- NumPy names and types: `"uint8"`, `np.uint8`, `np.dtype("uint8")`
- the Python types `bool`, `int` and `float`

`bool` means `Byte` ([ADR 5](adr_05_signed_and_unsigned.md)).

An explicit type is used exactly as given. GeoKit does not check whether the values of the operation fit it,
reads no values and issues no warning ([ADR 3](adr_03_operation_rules.md)). A sum into `Byte` can overflow,
an interpolation into an integer type is rounded, and a type narrower than the input loses precision. The
[docstring](adr_01_dtype_modes.md#docstring) of `dtype` states these risks.

GeoKit raises a `GeoKitDataTypeError` only for a request it cannot carry out:

- **A noData, fill or burn value that the type cannot store**, such as −1 in `UInt16`. GeoKit could not write
  a valid raster with it.
- **Bare integer constants** such as `gdal.GDT_Float32`. GDAL and OGR constants share the same integers
  (`ogr.OFTReal` and `gdal.GDT_UInt16` are both 2). Accepting them would bring back the class of bug behind
  #396. The error message names the string spelling to use instead.
- Complex, string and object dtypes, which GDAL bands cannot store.
- A type that the installed GDAL does not support, such as `Int8` before GDAL 3.7.

These checks look only at what GeoKit already holds in memory: the spelling of `dtype`, the noData, fill and
burn values, and the types the installed GDAL supports. They never read pixel or attribute values and never
compute statistics, so they cost nothing.

Every `GeoKitDataTypeError` names the function, the requested type and its range, the value that does not
fit, and how to fix the call:

```text
GeoKitDataTypeError: createRaster: noData=-1 cannot be stored in the requested dtype UInt16
(range 0 to 65535). Pass a noData value in that range, a type that holds it such as
dtype="Int32", or dtype="auto" to let GeoKit choose.
```

`GeoKitDataTypeError` is the only error for data types; `GeoKitCDataError` is removed
([ADR 7](adr_07_numpy_dtype_inside.md)).

## Consequences

- An explicit type behaves as it did up to v1.7.0. Most of the changes are restorations.
- Two new errors appear where GeoKit used to continue silently: a type that cannot hold the noData value, and
  a bare GDAL constant such as `dtype=gdal.GDT_Float32`. Both are listed in the changelog when they ship.
- The C data types example describes `dtype` as a minimum. It is rewritten when the modes ship.
- `Extent.fit(unit, dtype=None)` uses a parameter of the same name for something else: rounding the bounds.
  Its docstring will point this out.

## Alternatives considered

- **Keep `dtype` as a lower bound.** This is the behaviour since v1.7.1. It defeats the purpose of the
  parameter and caused defects D3 and D10.
- **Accept bare integers, as `gdalType()` did.** This is the one part of the old behaviour that is not
  restored, because the GDAL and OGR constants overlap.
- **Warn when the operation may produce values that the explicit type cannot hold.** A caller who names a
  type has chosen it, and a check would cost a read on calls that did not ask for one
  ([ADR 3](adr_03_operation_rules.md)). The docstring states the risk instead.
