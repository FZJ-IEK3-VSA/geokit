# Test-case creator

Single source of truth for the deterministic rasters used by the warp resampling tests.

There are two cases, both on a 32×32 grid and **both always carrying a nodata region** (baked into
the data and registered as the band's noData value):

- **`"int"`** — an `Int16` raster whose every feature value is a whole number, and
- **`"float"`** — a `Float32` raster whose every feature value carries a fraction.

Having both is deliberate. They share the same spatial layout (smooth diagonal ramp, constant
block, sharp step edge, two discrete class patches, a single bright spike), but the integer case
cannot hold fractional results, so it exercises the rounding/casting an integer band performs;
the float case uses non-integer inputs so that even value-selecting algorithms (`near`, `mode`,
`min`/`max`, `med`, `q1`/`q3`) prove the float code path preserves sub-integer precision.

## Files on disk (not rebuilt in memory)

The input rasters are **written to disk and committed**, then read back by the tests — they are not
rebuilt in memory on every run. Committing them means a change to the construction in
`create_rasters.py` only takes effect once the files are regenerated, and that regeneration shows
up as a reviewable git diff, so the inputs can never drift silently from the golden references.

- Inputs: `geokit/data/raster_data/input_data/resampling_input_<case>.tif`
- Golden warp outputs: `geokit/data/raster_data/golden_regression_results/warp_resampling_<case>_<alg>.tif`

## Regenerating

After editing the feature construction in `create_rasters.py`:

```bash
# 1. rewrite the committed input rasters
python -m test.test_case_creator.create_rasters

# 2. rewrite the golden warp outputs from those inputs
GEOKIT_REGEN_GOLDEN=1 pytest test/test_04_raster.py -k warp_resampling_golden
```

Commit both sets of `.tif` files.

## Public API (`from test.test_case_creator import ...`)

- `TEST_CASE_NAMES` — `("int", "float")`
- `load_test_raster(case)` — load the committed input raster for a case (what the tests use)
- `build_test_raster(case)` — build the input raster in memory (used to write the committed file)
- `input_raster_path(case)` / `golden_raster_path(case, alg)` — committed file paths
- `write_input_rasters()` — (re)write all committed input rasters

## Consumers

- `test/test_04_raster.py` — `test_warp_resampling_golden` and `test_warp_resampling_nodata_respected`
  load each case once per session (a `scope="session"` fixture) and parametrize over
  `(case × algorithm)`.
- `test/test_case_inspector/inspect_warp_resampling.ipynb` — visual debugging of a failing case.
