# ADR 8: Values Handled in NumPy Keep Their Meaning

**Status:** accepted on 2026-09-30, being implemented
([#396](https://github.com/FZJ-IEK3-VSA/geokit/issues/396))

## Context

Several GeoKit functions work on NumPy arrays and lose values because of the array's dtype. The entries are in
the defect catalogue in [#405](https://github.com/FZJ-IEK3-VSA/geokit/issues/405):

- `extractMatrix(autocorrect=True)`, `extractValues`, `interpolateValues` and `rasterStats` apply scale and
  offset before they compare with the noData value. The comparison then never matches, so noData pixels of
  scaled rasters are returned as data, for example −999.9 (D23, D24).
- `RegionMask.indicateValues` writes the noData value into a `bool` array, where it becomes `True`. Every
  noData pixel is then indicated (D25).
- `gradient` subtracts in the source dtype. On an unsigned elevation model, `100 - 102` wraps around to 65 534
  (D17).
- `KernelProcessor` pads the matrix with an array whose dtype comes from the integer `edgeValue`. Float values
  copied into it are truncated (D18).
- `RegionMask.applyMask` writes a NumPy integer noData of −1 into a `uint8` array, where it silently becomes
  255 (D26).

## Decision

- **A value is never written into an array that cannot hold it.** Before GeoKit writes a scalar into an array,
  it promotes the array to `np.promote_types(array.dtype, type of the scalar)`. `bool` arrays become `uint8` or
  float first. This applies to `RegionMask.applyMask`, `RegionMask.indicateValues`, the padding of
  `KernelProcessor` and the noData masking in the `extract*` functions.
- **GeoKit's own arithmetic on raster values runs in `float64`**, for example in `gradient` and in statistics.
- **With scale and offset, noData is compared on the raw values.** GeoKit masks the noData pixels first, then
  applies scale and offset, then sets the masked pixels to NaN. The result is `float64`.

These rules do not depend on the `dtype` modes. They only correct values, so they can ship before the modes.

## Consequences

- `applyMask` of a `uint8` array with `noData=-1` returns an `int16` array instead of a `uint8` array.
- The processor of `indicateValues` returns `uint8` without noData and `float32` with noData, instead of `bool`.
- `rasterStats` of a scaled raster with noData reports fewer observations and a different mean. The old
  numbers included the noData pixels.
