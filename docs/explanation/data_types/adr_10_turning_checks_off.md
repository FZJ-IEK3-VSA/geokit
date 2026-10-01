# ADR 10: A Central Option Turns the Checks Off

**Status:** accepted on 2026-10-01, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

## Context

`"preserve_input"` and explicit types run no checks ([ADR 3](adr_03_operation_rules.md)). GeoKit still warns
in two cases ([ADR 4](adr_04_nodata_fill_and_burn_values.md)):

- `Int64` or `UInt64` data with a NaN noData becomes `Float64`, which holds whole numbers exactly only up to
  2⁵³. This warning appears under `"auto"` and `"smallest"`.
- `warp` creates pixels outside the source, and no noData value is set. This warning appears in every mode.

Users who process many rasters in a batch and know their data may not want these warnings.

## Decision

One option, `checks`, turns the remaining data-type checks off. It is `True` by default and is set centrally
in `geokit.dtypes`:

```python
geokit.dtypes.set_options(checks=False)        # for the whole process

with geokit.dtypes.options(checks=False):      # only inside this block
    geokit.raster.warp(source, srs=4326, resampleAlg="near")
```

With `checks=False`:

- GeoKit issues no `GeoKitDataTypeWarning`, in any mode.
- `"smallest"` still reads its output, because that read chooses the type. It is what the mode does, not a
  check.
- The type is chosen exactly as with the checks on. The option never changes the type or the values.
- The errors stay. GeoKit still raises a `GeoKitDataTypeError` for a request it cannot carry out: a scalar the
  type cannot store (NaN in an integer type, −1 in `Byte`), a bare integer as `dtype`, or an unsupported dtype.
  These are not checks for a possible loss. Going on would write an invalid raster.

`set_options` changes the setting for the whole process. `options` changes it only inside the `with` block
and restores the previous setting when the block ends. It is implemented with a `contextvars.ContextVar`, so
a block in one thread does not affect other threads.

## Consequences

- Goal G1 holds only while the checks are on, which is the default. A user who turns them off takes over the
  responsibility for precision and unflagged pixels ([goals](index.md#goals)).
- No function signature changes.
- The setting is global state: a call in one module can change how calls in another module behave. This is
  acceptable because the option never changes a result, only whether GeoKit warns.
- Tests that change the option must restore it, for example with a fixture.
- The same mechanism could later carry a library-wide default mode.

## Alternatives considered

- **A further mode string**, such as `"auto_unchecked"`. Every mode would need an unchecked twin.
- **A flag on every function**, such as `checks=False`. About 15 raster-writing functions would need the
  parameter, plus the methods of `Extent` and `RegionMask` that forward to them. Every signature would grow
  for a setting that changes no result.
- **Only filtering the warnings** with `warnings.filterwarnings("ignore", category=GeoKitDataTypeWarning)`.
  Since no check reads values, this has the same effect. It remains possible for users who prefer Python's
  warning filters.
