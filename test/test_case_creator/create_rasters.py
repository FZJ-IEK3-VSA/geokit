"""Single source of truth for the deterministic warp resampling test rasters.

The input rasters are *built here and written to disk* under
``geokit/data/raster_data/input_data`` (rather than rebuilt in memory on every test run). The tests
then read those committed files. This is deliberate: committing the inputs means any change to the
construction below only takes effect once the files are regenerated, and that regeneration shows up
as a reviewable diff in git -- so the input data can never change silently underneath the golden
references.

To (re)generate the committed input rasters after editing the construction below::

    python -m test.test_case_creator.create_rasters

Then regenerate the golden warp outputs (see test/test_04_raster.py) with::

    GEOKIT_REGEN_GOLDEN=1 pytest test/test_04_raster.py -k warp_resampling_golden

and commit both sets of files.
"""

import pathlib

import numpy as np

from geokit import raster, srs

# 32x32 grid; the nodata block sits on odd-aligned edges so a 2:1 downsample yields both
# fully-nodata output pixels (its interior) and mixed nodata/valid output pixels (its border).
GRID_SIZE = 32
PIXEL = 100
_NODATA_REGION = (slice(5, 11), slice(15, 23))

# Committed data locations (under the geokit package, alongside the other test/example data).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
INPUT_DATA_DIR = _REPO_ROOT / "geokit" / "data" / "raster_data" / "input_data"
GOLDEN_DIR = _REPO_ROOT / "geokit" / "data" / "raster_data" / "golden_regression_results"


def _int_feature_array():
    """Integer-valued feature layout: every value is a whole number.

    Algorithms that pick or aggregate to whole numbers (near/mode/min/max/...) stay integral here,
    and the interpolating algorithms expose the rounding/casting an integer band must perform.
    """
    rows, cols = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
    data = (rows + cols).astype(np.float64)  # smooth diagonal ramp (0..62)
    data[2:10, 2:10] = 10  # constant block
    data[12:20, :16] = 5  # step edge: low ...
    data[12:20, 16:] = 90  # ... to high at column 16
    data[22:26, 2:10] = 120  # discrete class patch A
    data[22:26, 18:26] = 210  # discrete class patch B
    data[29, 29] = 300  # single bright spike
    return data


def _float_feature_array():
    """Float-valued feature layout: *no value is a whole number*.

    This is the whole point of having a separate float case. The integer case above can never tell
    you whether the float code path preserves sub-integer precision, because all of its inputs and
    most of its outputs are integral anyway. Here every feature carries a fraction, so a
    value-selecting algorithm (near/mode/min/max/med/q...) that returns a source value unchanged
    still proves the Float32 band kept the fraction -- behaviour an integer raster physically
    cannot represent.
    """
    rows, cols = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
    data = (rows + cols) * 0.5 + 0.25  # fractional diagonal ramp (0.25 .. 31.25, step 0.5)
    data[2:10, 2:10] = 10.5  # constant block
    data[12:20, :16] = 1.25  # step edge: low ...
    data[12:20, 16:] = 88.6  # ... to high at column 16
    data[22:26, 2:10] = 120.75  # discrete class patch A
    data[22:26, 18:26] = 210.4  # discrete class patch B
    data[29, 29] = 305.9  # single bright spike
    return data


# name -> spec. nodata is negative so it forces a signed dtype and never collides with the
# (non-negative) feature values.
TEST_CASES = {
    "int": dict(dtype="Int16", noData=-9999, cast=np.int16, features=_int_feature_array),
    "float": dict(dtype="Float32", noData=-9999.0, cast=np.float32, features=_float_feature_array),
}
TEST_CASE_NAMES = tuple(TEST_CASES)


def input_raster_path(case):
    """Path to the committed input raster for ``case``: input_data/resampling_input_<case>.tif."""
    if case not in TEST_CASES:
        raise KeyError(f"unknown test case {case!r}; choose from {sorted(TEST_CASES)}")
    return str(INPUT_DATA_DIR / f"resampling_input_{case}.tif")


def golden_raster_path(case, resample_alg):
    """Path to the committed golden warp output for ``(case, resample_alg)``."""
    if case not in TEST_CASES:
        raise KeyError(f"unknown test case {case!r}; choose from {sorted(TEST_CASES)}")
    return str(GOLDEN_DIR / f"warp_resampling_{case}_{resample_alg}.tif")


def build_test_raster(case, pixel=PIXEL):
    """Build the deterministic resampling test raster for ``case`` ('int' or 'float') in memory.

    The raster always contains a nodata region (registered as the band's noData value), so callers
    do not need to opt into nodata handling -- it is part of the test data by construction.

    Returns an in-memory gdal.Dataset on a 32x32 grid with bounds (0, 0, 32*pixel, 32*pixel). This
    is used to *write* the committed input file; tests read that file via ``load_test_raster``.
    """
    if case not in TEST_CASES:
        raise KeyError(f"unknown test case {case!r}; choose from {sorted(TEST_CASES)}")
    spec = TEST_CASES[case]

    data = spec["features"]()
    data[_NODATA_REGION] = spec["noData"]
    data = spec["cast"](data)

    return raster.createRaster(
        bounds=(0, 0, GRID_SIZE * pixel, GRID_SIZE * pixel),
        data=data,
        pixelWidth=pixel,
        pixelHeight=pixel,
        srs=srs.EPSG3035,
        dtype=spec["dtype"],
        noData=spec["noData"],
    )


def load_test_raster(case):
    """Load the committed input raster for ``case`` from disk.

    Raises a clear error (pointing at the regeneration command) if the file is missing, rather than
    silently rebuilding it -- the committed file is the source of truth for the tests.
    """
    path = input_raster_path(case)
    if not pathlib.Path(path).is_file():
        raise FileNotFoundError(
            f"missing committed input raster for case {case!r}: {path}\n"
            "Generate it with: python -m test.test_case_creator.create_rasters"
        )
    return raster.loadRaster(path)


def write_input_rasters(overwrite=True):
    """Write every test-case input raster to its committed path; return the list of paths."""
    INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for case in TEST_CASE_NAMES:
        source = build_test_raster(case)
        path = input_raster_path(case)
        raster.createRaster(
            bounds=raster.rasterInfo(source).bounds,
            data=raster.extractMatrix(source),
            pixelWidth=PIXEL,
            pixelHeight=PIXEL,
            srs=srs.EPSG3035,
            dtype=TEST_CASES[case]["dtype"],
            noData=TEST_CASES[case]["noData"],
            output=path,
            overwrite=overwrite,
        )
        written.append(path)
    return written


def main():
    for path in write_input_rasters():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
