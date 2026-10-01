# Data Types

GDAL stores every raster band in one fixed C type, such as `Byte` (8-bit unsigned) or `Float32`. OGR stores
every vector field as a typed column, such as `Integer` or `Real`. Many GeoKit functions create rasters or
fields for you, so GeoKit has to choose these types. If the type is too narrow, values are clipped, wrapped
around or truncated without notice. If it is too wide, rasters use more memory and disk space than needed.

The pages in this section record how GeoKit makes that choice and why. Each decision is written as an
architecture decision record (ADR): the context, the decision, its consequences, and the alternatives that
were considered.


## History

- **Up to v1.7.0**, types were chosen by `gdalType()`. An explicit `dtype` was used as given. Otherwise
  GeoKit used the dtype of the data, and otherwise `Byte`. `warp` kept the source type. `rasterize` used
  `Byte` for a burn value of 1 and `Float32` for everything else, including attributes, so integers above
  2²⁴ lost precision.
- **v1.7.1** introduced `MinimumCDataTypeHandler`. It infers a "minimum" type from a few sample numbers:
  noData, fill value, burn value, and the minimum and maximum of the input. It fixed some of the old problems
  but broke the meaning of an explicit `dtype`.
- **Issue #396** reported a float attribute that `rasterize` burned into an `Int16` raster, so 9.3 became 9.
  A sweep of every GeoKit function then found 29 data-type defects, and a review of that sweep added 11
  findings. They are listed in [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405), the umbrella issue of
  the fix. Each defect has a regression test in `test/test_13_dtypes_regression.py` whose name starts with
  the defect ID.

Several of the decisions below restore the behaviour before v1.7.1. They do not introduce it.

## Goals

- **G1** Users do not have to think about data types, and results are never silently wrong: no overflow,
  clipping, truncation, lost precision or lost noData. This holds for the modes that choose the type,
  `"auto"` and `"smallest"`, while the checks are on. A user who fixes the type with `"preserve_input"` or an
  explicit `dtype` gets no checks of the values against that type and no warnings about lost values, and the
  docstring says so. A user who turns the checks off
  ([ADR 10](adr_10_turning_checks_off.md)) gets no warnings in any mode.
- **G2** GeoKit widens a type automatically when an operation needs it. It never narrows without proof.
- **G3** An explicit `dtype` always wins. GeoKit uses it as given and does not check it against the values. It
  raises an error only for a request it cannot carry out.
- **G4** The choice is deterministic: the output type depends on the input types and the parameters, not on
  the data values. The only exception is value-based shrinking, which the user must ask for.
- **G5** GeoKit reads raster values to choose a type only under `"smallest"`, which the user must ask for.

Out of scope: complex types, GDAL's `Float16` (GDAL 3.11 and later), and multi-band rasters whose bands have
different types.

## Root causes

The sweep traced the defects to six causes:

| ID | Cause | Example | Addressed by |
|---|---|---|---|
| R1 | Types travel as bare strings and integers from four vocabularies: GDAL pixel types, OGR field types, NumPy dtypes and pandas dtypes. Nothing records which vocabulary a value belongs to. | An OGR field constant was passed to a GDAL function (#396). | [ADR 7](adr_07_numpy_dtype_inside.md) |
| R2 | The handler's promotion rules are wrong: signed and unsigned types collide, and the float width is chosen by range, not by precision. | `Byte` becomes `Int8`; `Int32` with a NaN noData becomes `Float32`. | [ADR 4](adr_04_nodata_fill_and_burn_values.md), [ADR 5](adr_05_signed_and_unsigned.md) |
| R3 | Operations are not modelled. The handler sees sample numbers but never what the operation does to the value range. | A sum of 16 pixels of 200 is stored as 255. | [ADR 3](adr_03_operation_rules.md) |
| R4 | A type that was already chosen is chosen a second time downstream, with less information. | `rasterize(value=200)` burns 127. | [ADR 6](adr_06_resolve_once.md) |
| R5 | An explicit `dtype` acts as a lower bound, and only strings are accepted. | `dtype="UInt16"` gives `Int16`. | [ADR 2](adr_02_explicit_dtype.md) |
| R6 | Values are handled in NumPy without care for their type. | The gradient of an unsigned elevation model wraps around. | [ADR 8](adr_08_values_in_numpy.md) |

## Decisions

| ADR | Decision |
|---|---|
| [1](adr_01_dtype_modes.md) | `dtype` takes three automatic modes: `"auto"` (the default; accuracy is kept), `"preserve_input"` (the input type is kept, without checks) and `"smallest"` (the smallest lossless type, found by reading the output once). |
| [2](adr_02_explicit_dtype.md) | An explicit `dtype` is used exactly as given, without checks against the values. Bare integer constants (as iternally used by GDAL) are rejected. |
| [3](adr_03_operation_rules.md) | The automatic type follows from the input types and the operation. Only `"smallest"` reads values, once from its output. `"preserve_input"` and explicit types run no checks. |
| [4](adr_04_nodata_fill_and_burn_values.md) | Every noData, fill and burn value that GeoKit writes must fit the output type. Integer data with a NaN noData becomes `Float64`. |
| [5](adr_05_signed_and_unsigned.md) | Where GeoKit chooses an integer width, `Byte` comes before `Int8`, and at 16 bits and wider the signed type comes before the unsigned one. |
| [6](adr_06_resolve_once.md) | The type is chosen once per public call. Internal helpers only convert it. |
| [7](adr_07_numpy_dtype_inside.md) | Inside GeoKit a type is always a `numpy.dtype`. GDAL, OGR and pandas types are converted at the edges, in one module. |
| [8](adr_08_values_in_numpy.md) | A value is never written into a NumPy array that cannot hold it. GeoKit's own arithmetic runs in `float64`. |
| [9](adr_09_vector_field_types.md) | OGR field types follow from the dtype, including 64-bit and unsigned integers. `polygonizeRaster` rejects float rasters. |
| [10](adr_10_turning_checks_off.md) | A central option, `geokit.dtypes.set_options(checks=False)`, turns the remaining data-type warnings off. The type and the errors stay the same. |

## Open questions

These are not decided and are not part of the current work:

- **Default resampling for integer rasters.** `warp` and `RegionMask.warp` default to `resampleAlg="bilinear"`,
  which is wrong for categorical rasters such as land cover. Under the `"auto"` mode, such a call returns
  `Float32`. Changing the default to `"near"` for integer inputs would change the values of default calls,
  so it needs its own decision.
- **An automatic noData for created pixels.** An opt-in `noData="auto"` could flag the pixels that a
  reprojection creates outside the source. See [ADR 4](adr_04_nodata_fill_and_burn_values.md).
- **Typed columns from `extractFeatures`.** Columns could take their dtype from the OGR field type (`int32` for
  an `Integer` field) instead of pandas' inference (`int64`). This would change the dtypes of every
  DataFrame users get back, so it would be optional at most.
- **Polygonizing float rasters.** `gdal.FPolygonize` could polygonize float rasters exactly, at the cost of
  many more polygons for continuous data. See [ADR 9](adr_09_vector_field_types.md).
