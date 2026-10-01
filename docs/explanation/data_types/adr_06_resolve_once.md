# ADR 6: Choose the Type Once per Call

**Status:** accepted on 2026-09-30, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

## Context

Public functions such as `rasterize`, `Extent.rasterMosaic` and `combineSimilarRasters` choose a type and pass
it to `quickRaster` or `createRaster`. Those run the handler again, now without the values that led to the
first choice. The handler's lookup is keyed by bit width, so signed and unsigned types of the same width
collide. The second pass therefore turns `Byte` into `Int8`, `UInt16` into `Int16` and `UInt32` into `Int32`.

This is the mechanism behind many entries of the [defect catalogue](defects.md):

- `rasterize(value=200)` burns 127; `rasterize(value=40000)` burns 32 767 (D2, M2).
- `Extent.rasterMosaic` and `combineSimilarRasters` turn `Byte` inputs into `Int8` (D5, D19).
- `mutateRaster` clips a processor's `uint8` or `uint16` output (M11).

## Decision

- The type is chosen once, by the public function that knows the inputs, the operation and the scalars
  ([ADR 1](adr_01_dtype_modes.md), [ADR 3](adr_03_operation_rules.md)).
- Internal helpers receive the chosen type and only convert it with `to_dtype`
  ([ADR 7](adr_07_numpy_dtype_inside.md)). Converting is idempotent: a type that is already chosen comes out
  unchanged.
- `createRaster` is split into the public function, which chooses the type, and an internal
  `_create_raster`, which only converts it. `quickRaster`, `Extent._quickRaster`, `rasterize`,
  `Extent.rasterMosaic` and `combineSimilarRasters` call the internal one.
- `quickRaster` stays public. It keeps accepting strings and NumPy types, because code outside GeoKit uses it.

## Consequences

- A type that was chosen once can no longer flip from unsigned to signed. This alone fixes D2, D3, D5 and D19.
- Each public function has exactly one place where its type is chosen.

## Alternatives considered

- **Let internal helpers accept only `np.dtype`**, checked with `isinstance`, and make `quickRaster` internal
  as well. This is not needed: the defects come from choosing twice, not from converting twice. Since
  conversion is idempotent, the check adds nothing, and making `quickRaster` internal would break code outside
  GeoKit.
