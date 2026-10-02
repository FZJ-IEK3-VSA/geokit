"""The data-type contract of GeoKit: every entry of the defect catalogue in #405 and the decisions in
docs/explanation/data_types.

- MODE_CASES (ADR 1, 3, 4, 5, 10): one row per call, with the band type per mode and the values the call keeps.
  Each row runs with dtype="auto", "preserve_input" and "smallest".
- EXPLICIT_CASES (ADR 2): a fixed dtype gives exactly that band type, or raises.
- Standalone tests for the entries where no raster type is chosen: values handled in NumPy (ADR 8), field types of
  vectors (ADR 9), the default resampling of the wrappers (ADR 10), warnings and reads (ADR 3, 4).

A case whose name starts with a catalogue ID, such as "D2_rasterize_200", pins that entry of #405, so
``pytest -k D2_`` runs every case of D2. A row of MODE_CASES with a catalogue ID also runs as the default call,
without dtype, because that is the call the entry is about. The other rows pin decisions of the ADRs.

What this branch does not fix yet is marked xfail(strict=True):

- PENDING lists the catalogue cases that are not fixed, with the pull request that fixes them;
- WITHOUT_MODES lists the functions that do not take the dtype modes, with the pull request that adds them.

Each pull request deletes its lines from the two lists, so its test diff shows what it fixes. A strict mark fails the
run as soon as its case passes, so the lists cannot go stale.
"""

import inspect
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, ogr
from typeguard import suppress_type_checks

import geokit
from geokit.error import GeoKitDataTypeError, GeoKitDataTypeWarning
from test.gdal_builders import (
    band_type_name,
    create_gdal_raster,
    spatial_reference_from_epsg,
    square_polygon,
    write_gdal_geopackage,
    write_gdal_geotiff,
)

# ----------------------------------------------------------------------------------------------
# sources: rasters of 100 m pixels in EPSG:3035, built with plain GDAL


def gdal_raster(values, numpy_type, gdal_type, **raster_kwargs):
    pixel_values = np.array(values, numpy_type)
    return create_gdal_raster(pixel_values, gdal_type, **raster_kwargs)


def byte_mask():
    """20 x 20 pixels; the left seven columns are 1, the others 0."""
    mask_values = np.zeros((20, 20), np.uint8)
    mask_values[:, :7] = 1
    return create_gdal_raster(mask_values, gdal.GDT_Byte)


def byte_step():
    """20 x 20 pixels; the left half is 0, the right half 255."""
    step_values = np.zeros((20, 20), np.uint8)
    step_values[:, 10:] = 255
    return create_gdal_raster(step_values, gdal.GDT_Byte)


def byte_10_and_30():
    """20 x 20 pixels; the left ten columns are 10, the others 30."""
    class_values = np.full((20, 20), 30, np.uint8)
    class_values[:, :10] = 10
    return create_gdal_raster(class_values, gdal.GDT_Byte)


def byte_ones():
    return gdal_raster(np.ones((4, 4)), np.uint8, gdal.GDT_Byte)


def byte_200s():
    return gdal_raster(np.full((20, 20), 200), np.uint8, gdal.GDT_Byte)


def byte_0_to_127():
    return gdal_raster([[0, 50], [100, 127]], np.uint8, gdal.GDT_Byte)


def int32_0_to_15():
    return gdal_raster(np.arange(16).reshape(4, 4), np.int32, gdal.GDT_Int32)


def int32_large_ids():
    return gdal_raster([[123456789, 987654321]], np.int32, gdal.GDT_Int32)


def float32_fractions():
    return gdal_raster([[0.5, 2.5]], np.float32, gdal.GDT_Float32)


def float64_quarters():
    return gdal_raster([[0.5, 0.25]], np.float64, gdal.GDT_Float64)


def float64_whole_numbers():
    return gdal_raster([[0, 1], [2, 3]], np.float64, gdal.GDT_Float64)


def tile(tmp_path, name, value, numpy_type, gdal_type, x_min=0):
    """A GeoTIFF of 4 x 4 pixels, all holding value."""
    pixel_values = np.full((4, 4), value, numpy_type)
    return write_gdal_geotiff(tmp_path / name, pixel_values, gdal_type, x_min=x_min)


def squares(*x_mins):
    """A vector of 1000 m squares in EPSG:3035 whose lower left corners are (x_min, 0)."""
    polygons = [square_polygon(x_min) for x_min in x_mins]
    return geokit.vector.createVector(polygons)


def three_points(values, numpy_type=None):
    """A vector of three points in EPSG:4326 whose attribute "v" holds values."""
    coordinates = [(6.1, 50.1), (7.5, 52.0), (6.8, 51.4)]
    points = [geokit.geom.point(x, y, srs=4326) for x, y in coordinates]
    if numpy_type is not None:
        values = np.array(values, numpy_type)
    attributes = pd.DataFrame({"geom": points, "v": values})
    return geokit.vector.createVector(attributes)


# ----------------------------------------------------------------------------------------------
# calls: each returns build(tmp_path, **dtype), which makes the call and returns the raster it writes


def create(**kwargs):
    """A createRaster call on a 10 x 10 grid of 1 degree."""
    return lambda _, **dtype: geokit.raster.createRaster(
        bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326, **kwargs, **dtype
    )


def quick(**kwargs):
    """A quickRaster call on a 10 x 10 grid of 1 degree."""
    wgs84 = spatial_reference_from_epsg(4326)
    return lambda _, **dtype: geokit.util.quickRaster(bounds=(0, 0, 10, 10), srs=wgs84, dx=1, dy=1, **kwargs, **dtype)


def create_like(source, **kwargs):
    return lambda _, **dtype: geokit.raster.createRasterLike(source(), **kwargs, **dtype)


def save_as_tif(source):
    return lambda tmp_path, **dtype: geokit.raster.saveRasterAsTif(source(), str(tmp_path / "saved.tif"), **dtype)


def burn(value, *x_mins, **kwargs):
    """Rasterize 1000 m squares onto 100 m pixels with a constant burn value."""
    return lambda _, **dtype: geokit.vector.rasterize(
        squares(*x_mins), pixelWidth=100, pixelHeight=100, value=value, **kwargs, **dtype
    )


def burn_field(values, numpy_type=None, **kwargs):
    """Rasterize three points by their attribute onto 0.5 degree pixels."""
    point_grid = dict(pixelWidth=0.5, pixelHeight=0.5, srs=4326, bounds=(5, 50, 8, 53))
    return lambda _, **dtype: geokit.vector.rasterize(
        three_points(values, numpy_type), value="v", **point_grid, **kwargs, **dtype
    )


def warp(source, resampleAlg=None, pixel_size=None, **kwargs):
    """A warp call; without resampleAlg, warp uses its default resampling."""
    if resampleAlg is not None:
        kwargs.update(resampleAlg=resampleAlg)
    if pixel_size is not None:
        kwargs.update(pixelWidth=pixel_size, pixelHeight=pixel_size)
    return lambda _, **dtype: geokit.raster.warp(source(), **kwargs, **dtype)


def warp_like(source):
    """A warpLike call onto the grid of a second copy of the source."""
    return lambda _, **dtype: geokit.raster.warpLike(source(), source(), **dtype)


def mutate(processor, source=byte_ones):
    return lambda _, **dtype: geokit.raster.mutateRaster(source(), processor=processor, **dtype)


def extent_mutate(source, processor=None):
    """Extent.mutateRaster over the extent of the source itself."""

    def build(_, **dtype):
        source_raster = source()
        source_extent = geokit.Extent.fromRaster(source_raster)
        return source_extent.mutateRaster(source_raster, processor=processor, **dtype)

    return build


def mosaic(*tiles):
    """Extent.rasterMosaic of GeoTIFF tiles, each given as (value, numpy type, GDAL type) and placed side by side."""

    def build(tmp_path, **dtype):
        tile_paths = []
        for index, (value, numpy_type, gdal_type) in enumerate(tiles):
            tile_path = tile(tmp_path, f"tile_{index}.tif", value, numpy_type, gdal_type, x_min=400 * index)
            tile_paths.append(tile_path)
        mosaic_extent = geokit.Extent(0, 0, 400 * len(tiles), 400, srs=3035)
        return mosaic_extent.rasterMosaic(tile_paths, _skipFiltering=True, **dtype)

    return build


def combine(*tiles):
    """A combineSimilarRasters call on GeoTIFF tiles, each given as (value, numpy type, GDAL type), side by side."""

    def build(tmp_path, **dtype):
        from geokit._algorithms.combineSimilarRasters import combineSimilarRasters

        tile_paths = []
        for index, (value, numpy_type, gdal_type) in enumerate(tiles):
            tile_path = tile(tmp_path, f"tile_{index}.tif", value, numpy_type, gdal_type, x_min=400 * index)
            tile_paths.append(tile_path)
        combined_path = str(tmp_path / "combined.tif")
        combineSimilarRasters(tile_paths, output=combined_path, verbose=False, **dtype)
        return combined_path

    return build


def halve(matrix):
    return matrix * 0.5


def double(matrix):
    return matrix * 2


def times_200_in_uint8(matrix):
    return matrix.astype(np.uint8) * np.uint8(200)


def times_40000_in_uint16(matrix):
    return matrix.astype(np.uint16) * np.uint16(40000)


def greater_than_five(matrix):
    return matrix > 5


# ----------------------------------------------------------------------------------------------
# value checks


def distinct(*expected_values):
    """The pixels hold exactly these values, apart from NaN."""

    def check(matrix):
        values = matrix[~np.isnan(matrix)] if matrix.dtype.kind == "f" else matrix
        assert set(np.unique(values).tolist()) == set(expected_values)

    return check


def spans(below, above):
    """The pixels reach below and above the given values, so nothing was clipped to that range."""

    def check(matrix):
        assert matrix.min() < below
        assert matrix.max() > above

    return check


# ----------------------------------------------------------------------------------------------
# the two tables

INT64_0_TO_5 = np.arange(100, dtype=np.int64).reshape(10, 10) % 6
UINT8_ONES = np.ones((10, 10), np.uint8)
INT32_ABOVE_2_24 = np.full((10, 10), 2**24 + 1, np.int32)
BOOL_HALVES = np.arange(100).reshape(10, 10) < 50
FLOAT64_HALVES = np.full((4, 4), 0.5)
BYTE = (np.uint8, gdal.GDT_Byte)
INT16 = (np.int16, gdal.GDT_Int16)
INT32 = (np.int32, gdal.GDT_Int32)

# The band type per mode, as in the worked examples of ADR 1: "auto" (also the default call), "preserve_input",
# "smallest". "Float64+warning" is Float64 with a GeoKitDataTypeWarning, "error" a GeoKitDataTypeError. The values
# are checked in every mode but "preserve_input", which may round or clip by design.
# fmt: off
MODE_CASES = {
    # name                                       call                                            auto preserve_input smallest             values
    "M4_createRaster_empty":                    (create(),                                      "Byte Byte Byte",                        None),
    "createRaster_int64_0_to_5":                (create(data=INT64_0_TO_5),                     "Int64 Int64 Byte",                      distinct(0, 1, 2, 3, 4, 5)),
    "createRaster_uint8_nodata_-1":             (create(data=UINT8_ONES, noData=-1),            "Int16 error Int8",                      distinct(1)),
    "D9_createRaster_above_2**24_nodata_0.5":   (create(data=INT32_ABOVE_2_24, noData=0.5),     "Float64 error Float64",                 distinct(2**24 + 1)),
    "D9_createRaster_fill_12.34":               (create(fill=12.34),                            "Float64 error Float64",                 distinct(12.34)),
    "D28_createRaster_bool_data":               (create(data=BOOL_HALVES),                      "Byte Byte Byte",                        distinct(0, 1)),
    "M6_createRaster_nodata_without_data":      (create(noData=5),                              "Byte Byte Byte",                        distinct(5)),
    "quickRaster_nodata_-9999":                 (quick(noData=-9999),                           "Int16 error Int16",                     distinct(-9999)),
    "D6_createRasterLike_float32":              (create_like(float32_fractions),                "Float32 Float32 Byte",                  None),
    "createRasterLike_float64_data":            (create_like(byte_ones, data=FLOAT64_HALVES),   "Float64 Float64 Float32",               distinct(0.5)),
    "D4_saveRasterAsTif_byte":                  (save_as_tif(byte_0_to_127),                    "Byte Byte Byte",                        distinct(0, 50, 100, 127)),
    "saveRasterAsTif_float64_whole":            (save_as_tif(float64_whole_numbers),            "Float64 Float64 Byte",                  distinct(0, 1, 2, 3)),
    "mutateRaster_halve":                       (mutate(halve),                                 "Float64 Byte Float32",                  distinct(0.5)),
    "M11_mutateRaster_uint8_200":               (mutate(times_200_in_uint8),                    "Byte Byte Byte",                        distinct(200)),
    "M11_mutateRaster_uint16_40000":            (mutate(times_40000_in_uint16),                 "UInt16 Byte UInt16",                    distinct(40000)),
    "M4_rasterize_1":                           (burn(1, 0),                                    "Byte Byte Byte",                        distinct(1)),
    "D2_rasterize_200":                         (burn(200, 0),                                  "Byte Byte Byte",                        distinct(200)),
    "M2_rasterize_40000":                       (burn(40000, 0),                                "UInt16 UInt16 UInt16",                  distinct(40000)),
    "M2_rasterize_2**31":                       (burn(2**31, 0),                                "UInt32 UInt32 UInt32",                  distinct(2**31)),
    "rasterize_-1":                             (burn(-1, 0),                                   "Int8 Int8 Int8",                        distinct(-1)),
    "D9_rasterize_0.1":                         (burn(0.1, 0),                                  "Float64 Float64 Float64",               distinct(0.1)),
    # three features of 100 can add up to 300, which needs Int16; at most two overlap, so 200 is the maximum
    "D16_rasterize_add_100_three_squares":      (burn(100, 0, 500, 1000, add=True),             "Int16 Byte Byte",                       distinct(100, 200)),
    "D1_rasterize_int32_field":                 (burn_field([1, 2, 3], np.int32),               "Int32 Int32 Byte",                      distinct(0, 1, 2, 3)),
    "rasterize_int64_field":                    (burn_field([1, 2, 3], np.int64),               "Int64 Int64 Byte",                      distinct(0, 1, 2, 3)),
    "D1_rasterize_real_field_of_396":           (burn_field([7.0, 9.3, 1.392]),                 "Float64 Float64 Float64",               distinct(0, 1.392, 7, 9.3)),
    "D1_rasterize_real_field_float32_exact":    (burn_field([0.5, 1.5, 2.5]),                   "Float64 Float64 Float32",               distinct(0, 0.5, 1.5, 2.5)),
    "D8_rasterize_int64_field_nan_nodata":      (burn_field([123456789, 987654321, 5], np.int64, noData=np.nan),
                                                                                                "Float64+warning error Float64+warning", distinct(5, 123456789, 987654321)),
    "M1_warp_byte_near":                        (warp(byte_mask, "near"),                       "Byte Byte Byte",                        distinct(0, 1)),
    "D12_warp_byte_average":                    (warp(byte_mask, "average", 400),               "Float32 Byte Float32",                  distinct(0, 0.75, 1)),
    "D12_warp_byte_bilinear":                   (warp(byte_mask, "bilinear", 400),              "Float32 Byte Float32",                  None),
    "D12_warp_int32_average":                   (warp(int32_0_to_15, "average", 200),           "Float64 Int32 Float32",                 None),
    "M17_warp_byte_classes_default":            (warp(byte_10_and_30, pixel_size=50),           "Byte Byte Byte",                        distinct(10, 30)),
    "D13_warp_byte_cubic_overshoot":            (warp(byte_step, "cubic", 25),                  "Float32 Byte Float32",                  spans(0, 255)),
    "D14_warp_byte_sum_16_times_200":           (warp(byte_200s, "sum", 400),                   "Float64 Byte Int16",                    distinct(3200)),
    "warp_byte_nan_nodata":                     (warp(byte_mask, "near", noData=np.nan),        "Float32 error Float32",                 distinct(0, 1)),
    "D8_warp_int32_nan_nodata":                 (warp(int32_large_ids, "near", noData=np.nan),  "Float64 error Float64",                 distinct(123456789, 987654321)),
    "D5_rasterMosaic_byte_200":                 (mosaic((200, *BYTE)),                          "Byte Byte Byte",                        distinct(200)),
    "M13_rasterMosaic_byte_then_int16_1000":    (mosaic((100, *BYTE), (1000, *INT16)),          "Int16 Int16 Int16",                     distinct(100, 1000)),
    "D19_combineSimilarRasters_byte":           (combine((100, *BYTE), (200, *BYTE)),           "Byte Byte Byte",                        distinct(100, 200)),
    "combineSimilarRasters_int32_3000_7":       (combine((3000, *INT32), (7, *INT32)),          "Int32 Int32 Int16",                     distinct(7, 3000)),
}

# A fixed dtype is used exactly as given and gives no warning (ADR 2). "error" is a GeoKitDataTypeError. The M15
# rows pass "preserve_input", because that entry is about passing dtype on.
EXPLICIT_CASES = {
    # name                                                    call                                        dtype               band type  values
    "D3_createRaster_Byte":                                  (create(),                                   "Byte",             "Byte",    None),
    "D3_createRaster_UInt16":                                (create(),                                   "UInt16",           "UInt16",  None),
    "D3_createRaster_UInt32":                                (create(),                                   "UInt32",           "UInt32",  None),
    "D3_quickRaster_Byte":                                   (quick(),                                    "Byte",             "Byte",    None),
    "D3_quickRaster_UInt16":                                 (quick(),                                    "UInt16",           "UInt16",  None),
    "D3_quickRaster_UInt32":                                 (quick(),                                    "UInt32",           "UInt32",  None),
    "D7_quickRaster_np.float32":                             (quick(),                                    np.float32,         "Float32", None),
    "D7_createRaster_np.float32":                            (create(),                                   np.float32,         "Float32", None),
    "D7_createRaster_float":                                 (create(),                                   float,              "Float64", None),
    "D7_createRaster_np.dtype_uint16":                       (create(),                                   np.dtype("uint16"), "UInt16",  None),
    "createRaster_Float32":                                  (create(),                                   "Float32",          "Float32", None),
    "createRaster_Int16":                                    (create(),                                   "Int16",            "Int16",   None),
    "D28_createRaster_bool":                                 (create(),                                   bool,               "Byte",    None),
    "D28_quickRaster_bool":                                  (quick(),                                    "bool",             "Byte",    None),
    "createRaster_scalars_that_fit":                         (create(noData=65535, fill=1),               "UInt16",           "UInt16",  distinct(1)),
    "D10_createRaster_noData_out_of_range":                  (create(noData=-1),                          "UInt16",           "error",   None),
    "M3_rasterize_Byte":                                     (burn(1, 0),                                 "Byte",             "Byte",    distinct(1)),
    "M3_rasterize_Int16_of_an_Int64_field":                  (burn_field([1, 2, 3], np.int64),            "Int16",            "Int16",   distinct(0, 1, 2, 3)),
    "D7_rasterize_np.float32":                               (burn(1, 0),                                 np.float32,         "Float32", distinct(1)),
    "D11_warp_Float32_of_a_Float64_source":                  (warp(float64_quarters, "near"),             "Float32",          "Float32", distinct(0.25, 0.5)),
    "D11_warpLike_Float32_of_a_Float64_source":              (warp_like(float64_quarters),                "Float32",          "Float32", distinct(0.25, 0.5)),
    "D28_mutateRaster_bool":                                 (mutate(greater_than_five, byte_0_to_127),   "bool",             "Byte",    distinct(0, 1)),
    "M14_combineSimilarRasters_Float32":                     (combine((100, *BYTE), (200, *BYTE)),        "Float32",          "Float32", distinct(100, 200)),
    "M14_combineSimilarRasters_np.int16":                    (combine((100, *BYTE), (200, *BYTE)),        np.int16,           "Int16",   distinct(100, 200)),
    "M15_Extent.mutateRaster_Int16":                         (extent_mutate(byte_0_to_127),               "Int16",            "Int16",   distinct(0, 50, 100, 127)),
    "M15_Extent.mutateRaster_preserve_input":                (extent_mutate(byte_0_to_127),               "preserve_input",   "Byte",    distinct(0, 50, 100, 127)),
    "M15_Extent.mutateRaster_preserve_input_with_processor": (extent_mutate(byte_0_to_127, double),       "preserve_input",   "Byte",    distinct(0, 100, 200, 254)),
}

# ----------------------------------------------------------------------------------------------
# the two pending lists

# Catalogue cases of #405 that this branch does not fix, with the pull request that fixes them. That pull request
# deletes the lines of its cases.
PENDING = {
    "D17_gradient_unsigned_dem":                              "PR 2",
    "D18_KernelProcessor_float_matrix":                       "PR 2",
    "D23_extractMatrix":                                      "PR 2",
    "D23_extractValues":                                      "PR 2",
    "D23_interpolateValues":                                  "PR 2",
    "D24_rasterStats":                                        "PR 2",
    "D25_indicateValues_nodata_-1":                           "PR 2",
    "D25_indicateValues_nodata_nan":                          "PR 2",
    "D26_applyMask":                                          "PR 2",
    "gradient_mode_ew":                                       "PR 2",
    "D27_checkSimilarRasters":                                "PR 7",
    "D27_combineSimilarRasters":                              "PR 7",
    "M13_rasterMosaic_byte_then_int16_1000":                  "PR 7",
    "M14_combineSimilarRasters_Float32":                      "PR 7",
    "M14_combineSimilarRasters_np.int16":                     "PR 7",
    "M15_Extent.mutateRaster_Int16":                          "PR 8",
    "M15_Extent.mutateRaster_preserve_input":                 "PR 8",
    "M15_Extent.mutateRaster_preserve_input_with_processor":  "PR 8",
    "M17_Extent.mutateRaster_default":                        "PR 8",
    "M17_RegionMask.mutateRaster_default":                    "PR 8",
    "M17_RegionMask.warp_default":                            "PR 8",
    "D20_createVector_float16":                               "PR 9",
    "D20_createVector_pandas_boolean":                        "PR 9",
    "D20_createVector_uint32":                                "PR 9",
    "D20_createVector_uint64":                                "PR 9",
    "D21_polygonizeRaster_float":                             "PR 9",
    "D21_polygonizeRaster_uint32":                            "PR 9",
    "D22_polygonizeMatrix_uint32":                            "PR 9",
    "M16_extractFeatures_integer64_with_null":                "PR 9",
}

# Functions that do not take the dtype modes on this branch, with the pull request that adds them. That pull
# request deletes the lines of its functions.
WITHOUT_MODES = {
    "combineSimilarRasters":                                  "PR 7",
    "rasterMosaic":                                           "PR 7",
    "RegionMask.indicateValues":                              "PR 8",
}
# fmt: on


def catalogue_id(case_name):
    """The catalogue ID that a case name starts with, such as "D2" for "D2_rasterize_200", or None."""
    match = re.match(r"([DM]\d+)_", case_name)
    return match.group(1) if match else None


def function_of(case_name):
    """The GeoKit function that a case name names after its catalogue ID, such as "rasterize"."""
    name_parts = case_name.split("_")
    if catalogue_id(case_name) is not None:
        name_parts = name_parts[1:]
    return name_parts[0]


def xfail_until(pull_request, case_name):
    """xfail(strict=True) while a pull request is still needed, otherwise no effect."""
    return pytest.mark.xfail(pull_request is not None, reason=f"{case_name}: fixed by {pull_request}", strict=True)


STANDALONE_CASES = set()
FUNCTIONS_WITH_MODE_TESTS = set()


def case(case_name):
    """Register a case of a standalone test; xfail(strict=True) while it is listed in PENDING."""
    STANDALONE_CASES.add(case_name)
    return xfail_until(PENDING.get(case_name), case_name)


def case_param(case_name, *values):
    """A pytest.param of a standalone case, with the case name as its id."""
    return pytest.param(*values, id=case_name, marks=case(case_name))


def modes_of(function_name):
    """xfail(strict=True) while the function is listed in WITHOUT_MODES."""
    FUNCTIONS_WITH_MODE_TESTS.add(function_name)
    return xfail_until(WITHOUT_MODES.get(function_name), f"the dtype modes of {function_name}")


# ----------------------------------------------------------------------------------------------
# the tests of the two tables


def mode_case_params():
    for name in MODE_CASES:
        modes = ["auto", "preserve_input", "smallest"]
        if catalogue_id(name) is not None:
            modes = [None, *modes]
        for mode in modes:
            if mode is None:
                pull_request = PENDING.get(name)
            else:
                pull_request = WITHOUT_MODES.get(function_of(name)) or PENDING.get(name)
            yield pytest.param(name, mode, id=f"{name}-{mode or 'default'}", marks=xfail_until(pull_request, name))


@pytest.mark.parametrize("name, mode", list(mode_case_params()))
def test_mode_case(name, mode, tmp_path):
    """The call gives the band type of its row in this mode, and keeps the values of its row."""
    build, types, check_values = MODE_CASES[name]
    auto, preserve_input, smallest = types.split()
    expected = {None: auto, "auto": auto, "preserve_input": preserve_input, "smallest": smallest}[mode]
    dtype_argument = {} if mode is None else {"dtype": mode}

    if expected == "error":
        with pytest.raises(GeoKitDataTypeError):
            build(tmp_path, **dtype_argument)
        return

    expected_type, _, expected_warning = expected.partition("+")
    if expected_warning:
        with pytest.warns(GeoKitDataTypeWarning):
            raster = build(tmp_path, **dtype_argument)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            raster = build(tmp_path, **dtype_argument)

    assert band_type_name(raster) == expected_type
    if check_values is not None and mode != "preserve_input":
        check_values(geokit.raster.extractMatrix(raster))


def explicit_case_params():
    for name in EXPLICIT_CASES:
        yield pytest.param(name, id=name, marks=xfail_until(PENDING.get(name), name))


@pytest.mark.parametrize("name", list(explicit_case_params()))
def test_explicit_case(name, tmp_path):
    """A fixed dtype gives exactly that band type, without a warning, or raises a GeoKitDataTypeError."""
    build, dtype, expected, check_values = EXPLICIT_CASES[name]

    if expected == "error":
        with pytest.raises(GeoKitDataTypeError):
            build(tmp_path, dtype=dtype)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        raster = build(tmp_path, dtype=dtype)

    assert band_type_name(raster) == expected
    if check_values is not None:
        check_values(geokit.raster.extractMatrix(raster))


# ----------------------------------------------------------------------------------------------
# raster types outside the two tables


@case("M12_createRaster_bare_integer")
def test_M12_bare_integer_dtype_is_rejected(tmp_path):
    """A bare integer such as gdal.GDT_Float32 raises an error that names the string spelling (ADR 2, #396)."""
    # typeguard, which the test suite runs, would reject the integer at the annotation before GeoKit's own check
    with suppress_type_checks(), pytest.raises(GeoKitDataTypeError, match='dtype="Float32"'):
        create()(tmp_path, dtype=gdal.GDT_Float32)


@case("D15_warp_reprojection_without_nodata")
def test_D15_warp_warns_about_created_pixels():
    """A reprojection without noData warns that the created pixels hold 0 and are not flagged."""
    int16_raster = gdal_raster(np.arange(1, 401).reshape(20, 20), np.int16, gdal.GDT_Int16, x_min=4e6, y_max=3e6)

    with pytest.warns(GeoKitDataTypeWarning, match="not flagged"):
        warped = geokit.raster.warp(int16_raster, srs=4326, resampleAlg="near")

    assert (geokit.raster.extractMatrix(warped) == 0).any()


def warp_the_tiles(tile_paths):
    geokit.raster.warp(tile_paths[0], resampleAlg="near")


def check_the_tiles(tile_paths):
    from geokit._algorithms.combineSimilarRasters import checkSimilarRasters

    checkSimilarRasters(tile_paths)


def combine_the_tiles(tile_paths):
    from geokit._algorithms.combineSimilarRasters import combineSimilarRasters

    combined_path = pathlib.Path(tile_paths[0]).with_name("combined.tif")
    combineSimilarRasters(tile_paths, output=str(combined_path), verbose=False)


@pytest.mark.parametrize(
    "read_the_tiles",
    [
        case_param("D27_warp", warp_the_tiles),
        case_param("D27_checkSimilarRasters", check_the_tiles),
        case_param("D27_combineSimilarRasters", combine_the_tiles),
    ],
)
def test_D27_no_statistics_are_read(read_the_tiles, tmp_path):
    """No .aux.xml file appears next to the input rasters, because no statistics are computed."""
    left_tile = tile(tmp_path, "left.tif", 100, np.uint8, gdal.GDT_Byte)
    right_tile = tile(tmp_path, "right.tif", 200, np.uint8, gdal.GDT_Byte, x_min=400)

    read_the_tiles([left_tile, right_tile])

    assert not (tmp_path / "left.tif.aux.xml").exists()
    assert not (tmp_path / "right.tif.aux.xml").exists()


@case("D4_saveRasterAsTif_scale_offset_nodata")
def test_D4_saveRasterAsTif_keeps_scale_offset_and_nodata(tmp_path):
    """The copy written by saveRasterAsTif keeps the stored values, scale, offset and noData of the source."""
    source = gdal_raster([[-9999, 100, 200]], np.int16, gdal.GDT_Int16, noData=-9999, scale=0.1, offset=5.0)

    saved = geokit.raster.saveRasterAsTif(source, str(tmp_path / "saved.tif"))

    saved_dataset = geokit.raster.loadRaster(saved)
    saved_band = saved_dataset.GetRasterBand(1)
    assert band_type_name(saved) == "Int16"
    assert (saved_band.GetScale(), saved_band.GetOffset(), saved_band.GetNoDataValue()) == (0.1, 5.0, -9999)
    np.testing.assert_array_equal(saved_band.ReadAsArray(), [[-9999, 100, 200]])


def square_region_mask():
    return geokit.RegionMask.fromGeom(square_polygon(0), pixelRes=100, srs=3035)


@case("M4_RegionMask.createRaster")
def test_M4_RegionMask_createRaster_gives_byte():
    """RegionMask.createRaster without data gives a Byte band."""
    created_raster = square_region_mask().createRaster()

    assert band_type_name(created_raster) == "Byte"


@case("M4_RegionMask.rasterize")
def test_M4_RegionMask_rasterize_gives_uint8():
    """RegionMask.rasterize with a burn value of 1 gives a uint8 matrix."""
    rasterized_matrix = square_region_mask().rasterize(squares(0), value=1)

    assert rasterized_matrix.dtype == np.uint8
    assert rasterized_matrix.max() == 1


def region_mask_with_50_m_pixels():
    """A region mask over the 2000 m square of the 20 x 20 sources, with pixels half as wide as theirs."""
    return geokit.RegionMask.fromGeom(geokit.geom.box(0, 0, 2000, 2000, srs=3035), pixelRes=50, srs=3035)


def warp_onto_50_m_pixels(source_raster):
    return region_mask_with_50_m_pixels().warp(source_raster, applyMask=False)


def extent_mutate_onto_50_m_pixels(source_raster):
    source_extent = geokit.Extent.fromRaster(source_raster)
    mutated_raster = source_extent.mutateRaster(source_raster, pixelWidth=50, pixelHeight=50, matchContext=True)
    return geokit.raster.extractMatrix(mutated_raster)


def region_mask_mutate_onto_50_m_pixels(source_raster):
    mutated_raster = region_mask_with_50_m_pixels().mutateRaster(source_raster, applyMask=False)
    return geokit.raster.extractMatrix(mutated_raster)


@pytest.mark.parametrize(
    "resample_onto_50_m_pixels",
    [
        case_param("M17_RegionMask.warp_default", warp_onto_50_m_pixels),
        case_param("M17_Extent.mutateRaster_default", extent_mutate_onto_50_m_pixels),
        case_param("M17_RegionMask.mutateRaster_default", region_mask_mutate_onto_50_m_pixels),
    ],
)
def test_M17_default_resampling_keeps_the_classes(resample_onto_50_m_pixels):
    """The default resampling of the warping wrappers keeps the classes of a categorical raster."""
    resampled_values = resample_onto_50_m_pixels(byte_10_and_30())

    assert set(np.unique(resampled_values).tolist()) == {10, 30}


@pytest.mark.parametrize(
    "mode, expected_dtype",
    [
        pytest.param("auto", np.float32, id="auto", marks=modes_of("RegionMask.indicateValues")),
        pytest.param("preserve_input", np.uint8, id="preserve_input", marks=modes_of("RegionMask.indicateValues")),
        pytest.param("smallest", np.float32, id="smallest", marks=modes_of("RegionMask.indicateValues")),
    ],
)
def test_indicateValues_takes_the_dtype_modes(mode, expected_dtype):
    """The indication keeps its fractions under auto and smallest, and its Byte type under preserve_input."""
    # bilinear resampling of the 0/1 indication onto 400 m pixels gives fractions where it crosses the edge
    region_mask = geokit.RegionMask.fromGeom(geokit.geom.box(0, 0, 2000, 2000, srs=3035), pixelRes=400, srs=3035)

    indicated = region_mask.indicateValues(
        byte_mask(), value=(1, None), dtype=mode, applyMask=False, multiProcess=False
    )

    assert indicated.dtype == expected_dtype


# ----------------------------------------------------------------------------------------------
# values handled in NumPy (ADR 8)


def scaled_int16_raster_with_nodata():
    """Int16 pixels -9999, 100 and 200 with noData -9999 and scale 0.1, so the data pixels read 10 and 20."""
    return gdal_raster([[-9999, 100, 200]], np.int16, gdal.GDT_Int16, noData=-9999, scale=0.1)


def first_two_values_of_extractMatrix(raster):
    return geokit.raster.extractMatrix(raster, autocorrect=True)[0, :2]


def first_two_values_of_extractValues(raster):
    extracted = geokit.raster.extractValues(raster, [(50, 50), (150, 50)], pointSRS=3035)
    return np.asarray(extracted.data)


def first_two_values_of_interpolateValues(raster):
    interpolated = geokit.raster.interpolateValues(raster, [(50, 50), (150, 50)], pointSRS=3035, mode="near")
    return np.asarray(interpolated)


@pytest.mark.parametrize(
    "first_two_values",
    [
        case_param("D23_extractMatrix", first_two_values_of_extractMatrix),
        case_param("D23_extractValues", first_two_values_of_extractValues),
        case_param("D23_interpolateValues", first_two_values_of_interpolateValues),
    ],
)
def test_D23_nodata_is_masked_before_the_scale(first_two_values):
    """The noData pixel of a scaled raster comes back as NaN, because the mask is built on the stored values."""
    nodata_value, data_value = first_two_values(scaled_int16_raster_with_nodata())

    assert np.isnan(nodata_value)
    assert np.isclose(data_value, 10.0)


@case("D24_rasterStats")
def test_D24_rasterStats_leaves_out_scaled_nodata():
    """The statistics of rasterStats leave out the noData pixels of a scaled raster."""
    stats = geokit.raster.rasterStats(scaled_int16_raster_with_nodata())

    assert stats.nobs == 2
    assert np.isclose(stats.mean, 15.0)


@case("D17_gradient_unsigned_dem")
def test_D17_gradient_of_unsigned_dem():
    """The gradient of a UInt16 elevation model is computed in float and does not wrap around."""
    # the terrain rises 1 m per 100 m pixel towards the east: (100 - 102) m / (2 * 100 m) = -0.01, while in uint16
    # 100 - 102 wraps around to 65534. The edge columns have no neighbour on one side and are left out.
    dem_raster = gdal_raster([[100, 101, 102, 103]] * 4, np.uint16, gdal.GDT_UInt16)

    east_west_gradient = geokit.raster.gradient(dem_raster, mode="east-west", asMatrix=True)

    np.testing.assert_allclose(east_west_gradient[:, 1:-1], -0.01)


@case("gradient_mode_ew")
def test_gradient_ew_is_east_west():
    """The gradient with mode='ew' is the same as with mode='east-west' (found on the way, no catalogue entry)."""
    dem_raster = gdal_raster([[100, 101, 102, 103]] * 4, np.float64, gdal.GDT_Float64)

    short_mode_gradient = geokit.raster.gradient(dem_raster, mode="ew", asMatrix=True)
    long_mode_gradient = geokit.raster.gradient(dem_raster, mode="east-west", asMatrix=True)

    np.testing.assert_array_equal(short_mode_gradient, long_mode_gradient)


@case("D18_KernelProcessor_float_matrix")
def test_D18_kernel_processor_keeps_floats():
    """KernelProcessor pads in a type that holds the matrix, so an integer edgeValue does not truncate floats."""
    float_matrix = np.array([[0.5, 1.5], [2.5, 3.5]])

    # returning the centre pixel makes the processor an identity, so only the padding can change the values
    def centre_pixel(window):
        return window[1, 1]

    process_every_window = geokit.util.KernelProcessor(1, edgeValue=0)(centre_pixel)
    output = process_every_window(float_matrix)

    assert output.dtype == np.float64
    np.testing.assert_array_equal(output, float_matrix)


@pytest.mark.parametrize(
    "noData",
    [case_param("D25_indicateValues_nodata_-1", -1), case_param("D25_indicateValues_nodata_nan", np.nan)],
)
def test_D25_indicateValues_does_not_indicate_nodata(noData):
    """NoData pixels of the source are not indicated by indicateValues, for an integer and a NaN noData."""
    left_half_nan_values = np.full((10, 10), 5.0, np.float32)
    left_half_nan_values[:, :5] = np.nan
    source_raster = geokit.raster.createRaster(
        bounds=(0, 0, 1000, 1000), pixelWidth=100, pixelHeight=100, srs=3035, data=left_half_nan_values, noData=np.nan
    )
    region_mask = geokit.RegionMask.fromGeom(geokit.geom.box(0, 0, 1000, 1000, srs=3035), pixelRes=100, srs=3035)

    indicated = region_mask.indicateValues(
        source_raster, value="[0-10]", noData=noData, applyMask=False, multiProcess=False
    )

    is_nodata = np.isnan(indicated) if np.isnan(noData) else indicated == noData
    assert is_nodata.sum() == 50
    assert (indicated[~is_nodata] == 1).sum() == 50


@case("D26_applyMask")
def test_D26_applyMask_widens_for_nodata():
    """A uint8 matrix is widened by applyMask, so a NumPy-integer noData of -1 is stored and not wrapped to 255."""
    triangle = geokit.geom.polygon([(0, 0), (1000, 0), (0, 1000), (0, 0)], srs=3035)
    triangle_mask = geokit.RegionMask.fromGeom(triangle, pixelRes=100, srs=3035)
    uint8_ones = np.ones(triangle_mask.mask.shape, np.uint8)

    masked = triangle_mask.applyMask(uint8_ones, noData=np.int64(-1))

    assert masked.min() == -1
    assert masked[triangle_mask.mask].min() == 1


# ----------------------------------------------------------------------------------------------
# field types of vectors (ADR 9)


def first_field_value(column):
    """Write a one-row column "v" with createVector and read the value back from the OGR feature."""
    attributes = pd.DataFrame({"geom": [geokit.geom.point(6.1, 50.1, srs=4326)], "v": column})
    vector = geokit.vector.createVector(attributes)
    return vector.GetLayer().GetNextFeature().GetField("v")


def through_createVector(value):
    return [first_field_value(np.array([value], np.uint32))]


def through_polygonizeRaster(value):
    uint32_raster = gdal_raster(np.full((4, 4), value), np.uint32, gdal.GDT_UInt32)
    return list(geokit.raster.polygonizeRaster(uint32_raster)["value"])


def through_polygonizeMatrix(value):
    uint32_matrix = np.full((2, 2), value, np.uint32)
    return list(geokit.geom.polygonizeMatrix(uint32_matrix)["value"])


@pytest.mark.parametrize(
    "into_a_vector",
    [
        case_param("D20_createVector_uint32", through_createVector),
        case_param("D21_polygonizeRaster_uint32", through_polygonizeRaster),
        case_param("D22_polygonizeMatrix_uint32", through_polygonizeMatrix),
    ],
)
def test_uint32_above_2_31_survives_the_way_into_a_vector(into_a_vector):
    """A uint32 value of 3 000 000 000, above 2**31 - 1, keeps its value in the field of the vector."""
    assert into_a_vector(3_000_000_000) == [3_000_000_000]


@pytest.mark.parametrize(
    "column, expected",
    [
        case_param("D20_createVector_uint64", np.array([5], np.uint64), 5),
        case_param("D20_createVector_float16", np.array([1.5], np.float16), 1.5),
        case_param("D20_createVector_pandas_boolean", pd.array([True], dtype="boolean"), 1),
    ],
)
def test_D20_createVector_writes_numeric_columns_as_numbers(column, expected):
    """Columns of uint64, float16 and pandas boolean become numeric fields in createVector, not strings."""
    field_value = first_field_value(column)

    assert not isinstance(field_value, str)
    assert field_value == expected


@case("D21_polygonizeRaster_float")
def test_D21_polygonizeRaster_warns_for_float_rasters():
    """A float raster is polygonized with rounded values and a warning, because 1.7 and 2.4 both become 2 and merge."""
    float_values = np.full((4, 4), 1.7, np.float32)
    float_values[:, 2:] = 2.4
    float_raster = create_gdal_raster(float_values, gdal.GDT_Float32)

    with pytest.warns(GeoKitDataTypeWarning, match="rounds"):
        polygons = geokit.raster.polygonizeRaster(float_raster)

    assert sorted(polygons["value"]) == [2]


@case("M16_extractFeatures_integer64_with_null")
def test_M16_extractFeatures_warns_for_integer64_with_null(tmp_path):
    """An Integer64 field with NULLs and a value beyond 2**53 warns, because pandas stores it as float64."""
    vector_path = write_gdal_geopackage(tmp_path / "ids.gpkg", "id", ogr.OFTInteger64, [2**60 + 1, None])

    with pytest.warns(GeoKitDataTypeWarning, match="asPandas=False"):
        geokit.vector.extractFeatures(vector_path)


@case("M8_vectorInfo_field_types")
def test_M8_vectorInfo_reports_ogr_names_and_dtypes():
    """The field type names of vectorInfo are OGR names, deprecated in favour of the dtypes of attribute_dtypes."""
    point = geokit.geom.point(6.1, 50.1, srs=4326)
    attributes = pd.DataFrame(
        {"geom": [point] * 2, "i32": np.array([1, 2], np.int32), "i64": [1, 2], "f": [0.5, 1.5], "s": ["a", "b"]}
    )
    info = geokit.vector.vectorInfo(geokit.vector.createVector(attributes))

    with pytest.warns(FutureWarning, match="attribute_dtypes"):
        field_type_names = info.attribute_data_types_str

    assert field_type_names == {"i32": "Integer", "i64": "Integer64", "f": "Real", "s": "String"}
    assert info.attribute_data_types_constant == {
        "i32": ogr.OFTInteger,
        "i64": ogr.OFTInteger64,
        "f": ogr.OFTReal,
        "s": ogr.OFTString,
    }
    assert info.attribute_dtypes == {
        "i32": np.dtype("int32"),
        "i64": np.dtype("int64"),
        "f": np.dtype("float64"),
        "s": None,
    }


# ----------------------------------------------------------------------------------------------
# shared docstring blocks and deprecated names


def test_raster_writers_share_the_dtype_docstring_block():
    """The dtype parameter of every raster writer is documented by the shared block of ADR 1, word for word."""
    raster_writers = [
        geokit.raster.createRaster,
        geokit.raster.createRasterLike,
        geokit.raster.saveRasterAsTif,
        geokit.raster.mutateRaster,
    ]
    for writer in raster_writers:
        assert geokit.dtypes.DTYPE_PARAMETER_DOCSTRING in inspect.getdoc(writer), writer.__name__


def test_createRasterLike_data_type_as_string_is_deprecated():
    """The data_type_as_string parameter of createRasterLike still works and warns in favour of dtype."""
    with pytest.warns(FutureWarning, match="dtype"):
        copied_raster = geokit.raster.createRasterLike(byte_ones(), data_type_as_string="Int16")

    assert band_type_name(copied_raster) == "Int16"


# ----------------------------------------------------------------------------------------------
# coverage of the catalogue

CATALOGUE = {f"D{number}" for number in range(1, 30)} | {f"M{number}" for number in range(1, 18)}

# entries of the catalogue that no case can pin
NOT_PINNED = {
    "D29": "minor issues inside the replaced handler, without effect with the supported GDAL and NumPy",
    "M5": "v1.9.1 already gives Int64 for an Integer64 field, by coincidence; rasterize_int64_field pins the type",
    "M7": "not a defect: the dtypes of the DataFrames from extractFeatures stay as they are",
    "M9": "not a defect: the statistics pass of D27 never changed a result",
    "M10": "not a defect: the memory print of indicateValues is wanted",
}


def all_case_names():
    return [*MODE_CASES, *EXPLICIT_CASES, *STANDALONE_CASES]


def test_every_catalogue_entry_is_pinned():
    """Every entry of the defect catalogue in #405 has a case in this module, or is listed as not pinnable."""
    pinned = {catalogue_id(name) for name in all_case_names()} - {None}

    assert pinned.isdisjoint(NOT_PINNED)
    assert pinned | set(NOT_PINNED) == CATALOGUE


def test_the_pending_lists_name_existing_cases_and_functions():
    """Every line of PENDING names a case of this module, and every line of WITHOUT_MODES a function with modes."""
    functions_with_modes = {function_of(name) for name in MODE_CASES} | FUNCTIONS_WITH_MODE_TESTS

    assert set(PENDING) <= set(all_case_names())
    assert set(WITHOUT_MODES) <= functions_with_modes
