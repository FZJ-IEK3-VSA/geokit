# Test-case inspector

Visual debugging aids for raster tests. When a golden-regression test in
`test/test_04_raster.py` fails, open the matching notebook here to *see* what changed instead of
reading raw arrays.

## Notebooks

- **`inspect_warp_resampling.ipynb`** — inspects `test_warp_resampling_golden`. Loads the same
  committed input raster (`test.test_case_creator.load_test_raster`), warps it with a chosen
  algorithm, and shows the source, the produced output, the committed golden, and their difference
  side by side. Also renders all 14 algorithms at a glance. Set `CASE = "int"` or `CASE = "float"`
  to switch test case; both carry a nodata region, drawn blank.

## Committed data

Input rasters live in `geokit/data/raster_data/input_data/resampling_input_<case>.tif` and golden
references in `geokit/data/raster_data/golden_regression_results/warp_resampling_<case>_<alg>.tif`;
both are committed to the repo. The goldens are generated automatically the first time
`test_warp_resampling_golden` runs. After editing the input construction, regenerate the inputs
first; after a deliberate GDAL/PROJ upgrade that legitimately changes the numerics, regenerate the
goldens. Commit the updated `.tif` files:

```bash
python -m test.test_case_creator.create_rasters                                 # inputs
GEOKIT_REGEN_GOLDEN=1 python -m pytest test/test_04_raster.py -k resampling_golden  # goldens
```

These notebooks are developer tools. They are not collected by pytest and not part of the docs site.
