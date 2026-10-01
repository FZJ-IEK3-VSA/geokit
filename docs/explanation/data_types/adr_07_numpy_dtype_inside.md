# ADR 7: `numpy.dtype` Is the Only Type Inside GeoKit

**Status:** accepted on 2026-09-30, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

## Context

Types travel through GeoKit as bare strings and integers from four vocabularies: GDAL pixel types, OGR field
types, NumPy dtypes and pandas dtypes. Nothing records which vocabulary a value belongs to, and the integer
constants of GDAL and OGR overlap:

| Integer | OGR field type | GDAL pixel type |
|---|---|---|
| 0 | `OFTInteger` | `GDT_Unknown` |
| 2 | `OFTReal` | `GDT_UInt16` |
| 12 | `OFTInteger64` | `GDT_UInt64` |

`vectorInfo` passes OGR field constants to `gdal.GetDataTypeName`. It reports a `Real` field as `"UInt16"`, an
`Integer64` field as `"UInt64"` and an `Integer` field as `"Unknown"` (M8 in
[#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405)). `rasterize` then burns float attributes into
`Int16` and raises for integer attributes. This is issue #396 (D1). About ten places in GeoKit hold their
own mapping between these vocabularies.

## Decision

Inside GeoKit, a type is always a `numpy.dtype`. Conversions happen only at the edges, in one new module,
`geokit/dtypes.py`, which replaces `geokit/c_data_type_handler.py`:

| Function | Converts |
|---|---|
| `to_dtype(x)` | any accepted spelling of `dtype` ([ADR 2](adr_02_explicit_dtype.md)) to a `numpy.dtype`; `bool` becomes `uint8` and `float16` becomes `float32`, because GDAL has no type for either |
| `to_gdal(dt)` | a `numpy.dtype` to a GDAL pixel type |
| `from_band(band)` | the pixel type of a GDAL band to a `numpy.dtype` |
| `from_ogr_field(field_defn)` | an OGR field definition to a `numpy.dtype`, or `None` for a non-numeric field |
| `to_ogr_field(dt)` | a `numpy.dtype` to an OGR field type ([ADR 9](adr_09_vector_field_types.md)) |
| `can_hold(dt, value)` | whether a value can be stored exactly in a type |
| `dtype_for_value(value)` | the smallest type that holds a scalar, in the order of [ADR 5](adr_05_signed_and_unsigned.md) |

`from_ogr_field` takes the field definition object, not an integer, so OGR and GDAL constants cannot be mixed
up. `RasterInfo` gains a `numpy_dtype` field and the result of `vectorInfo` gains `attribute_dtypes`.

GeoKit promotes only dtypes (`np.promote_types`, `np.result_type` applied to dtypes), never Python scalars.
NumPy 2 changed how scalars are promoted (NEP 50), and GeoKit supports both NumPy 1.26 and NumPy 2. Promoting
dtypes gives the same result on both.

Every data-type error raises one class, `GeoKitDataTypeError`, and every data-type warning one class,
`GeoKitDataTypeWarning`. `GeoKitDataTypeError` is defined in `geokit.error` next to the other GeoKit errors
and is a subclass of `GeoKitError`. `GeoKitCDataError` is not part of the new design.

## Consequences

- One mapping exists, and it is tested in one place. The class of bug behind #396 cannot come back.
- `vectorInfo(...).attribute_data_types_str` returns OGR names (`"Integer"`, `"Integer64"`, `"Real"`,
  `"String"`) once its values are fixed. It is then deprecated in favour of `attribute_dtypes`.
- `MinimumCDataTypeHandler` and the `*_literal` type aliases in `geokit.data_types` are deprecated with a
  `FutureWarning` for one minor release and then removed.
- `GeoKitCDataError` is removed together with `MinimumCDataTypeHandler`. Until then only the deprecated
  handler raises it. Code that catches `GeoKitCDataError` around a GeoKit call has to catch
  `GeoKitDataTypeError` instead, or `GeoKitError` to cover both versions. The changelog lists this.

## Alternatives considered

- **Make `GeoKitDataTypeError` a subclass of `GeoKitCDataError`**, so that existing `except` clauses keep
  working. This would keep two names for one kind of error after the handler is gone, and the "C data" name
  describes the handler, not the error.
