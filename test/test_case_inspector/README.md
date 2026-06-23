# Test-case inspector

Visual debugging aids for raster tests. When a golden-regression test in
`test/test_04_raster.py` fails, open the matching notebook here to *see* what changed instead of
reading raw arrays.

## Notebooks

- **`inspect_warp_resampling.ipynb`** — inspects `test_warp_resampling_golden`. Builds the same
  deterministic source (`test.helpers.make_resampling_test_raster`), warps it with a chosen
  algorithm, and shows the source, the produced output, the committed golden, and their difference
  side by side. Also renders all 14 algorithms at a glance.

## Golden references

The goldens live in `geokit/data/raster_results/warp_resampling_<alg>.tif` and are committed to the
repo. They are generated automatically the first time `test_warp_resampling_golden` runs. After a
deliberate GDAL/PROJ upgrade that legitimately changes the numerics, regenerate and commit them:

```bash
GEOKIT_REGEN_GOLDEN=1 python -m pytest test/test_04_raster.py -k resampling_golden
```

These notebooks are developer tools; they are not collected by pytest and not part of the docs site.
