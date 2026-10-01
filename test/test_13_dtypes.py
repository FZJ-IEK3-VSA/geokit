"""The data-type contract of GeoKit: every entry of the defect catalogue in #405 and the decisions in
docs/explanation/data_types.

- MODE_CASES (ADR 1, 3, 4, 5): one row per call, with the band type per mode and the values the call keeps. Each
  row runs as the default call and with dtype="auto", "preserve_input" and "smallest".
- EXPLICIT_CASES (ADR 2): an explicit dtype gives exactly that band type, or raises.
- One test per entry where no raster type is chosen: values handled in NumPy (ADR 8), field types of vectors
  (ADR 9), warnings and reads (ADR 3, 4).
- test_every_catalogue_entry_is_pinned: every ID of the catalogue is pinned by a row or a test of this module.

A row or test that is not fixed on this branch is pending: it is marked xfail(strict=True) with its catalogue ID.
The PR that fixes it deletes its line in PENDING, or its @pending mark. A strict mark fails the run as soon as its
case passes, so the marks cannot go stale.
"""

import inspect
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


def quick():
    """A quickRaster call on a 10 x 10 grid of 1 degree."""
    wgs84 = spatial_reference_from_epsg(4326)
    return lambda _, **dtype: geokit.util.quickRaster(bounds=(0, 0, 10, 10), srs=wgs84, dx=1, dy=1, **dtype)


def create_like(source):
    return lambda _, **dtype: geokit.raster.createRasterLike(source(), **dtype)


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


def warp(source, resampleAlg, pixel_size=None, **kwargs):
    if pixel_size is not None:
        kwargs.update(pixelWidth=pixel_size, pixelHeight=pixel_size)
    return lambda _, **dtype: geokit.raster.warp(source(), resampleAlg=resampleAlg, **kwargs, **dtype)


def mutate(processor, source=byte_ones):
    return lambda _, **dtype: geokit.raster.mutateRaster(source(), processor=processor, **dtype)


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
# marks of the tests outside the two tables

PINNED_BY_TESTS = set()


def pins(*catalogue_ids):
    """Record the catalogue IDs of #405 that a test pins."""
    PINNED_BY_TESTS.update(catalogue_ids)
    return lambda test: test


def pending(reason):
    """Mark a test xfail(strict=True) until the PR that fixes it deletes this mark."""
    return pytest.mark.xfail(strict=True, reason=reason)


# ----------------------------------------------------------------------------------------------
# the two tables

INT64_0_TO_5 = np.arange(100, dtype=np.int64).reshape(10, 10) % 6
UINT8_ONES = np.ones((10, 10), np.uint8)
INT32_ABOVE_2_24 = np.full((10, 10), 2**24 + 1, np.int32)
BYTE = (np.uint8, gdal.GDT_Byte)
INT16 = (np.int16, gdal.GDT_Int16)
INT32 = (np.int32, gdal.GDT_Int32)

# The band type per mode, as in the worked examples of ADR 1: "auto" (also the default call), "preserve_input",
# "smallest". "Float64+warning" is Float64 with a GeoKitDataTypeWarning, "error" a GeoKitDataTypeError. The values
# are checked in every mode but "preserve_input", which may round or clip by design.
# fmt: off
MODE_CASES = {
    # name                                  call                                          auto preserve_input smallest            values                       catalogue
    "createRaster_empty":                  (create(),                                     "Byte Byte Byte",                       None,                        ()),
    "createRaster_int64_0_to_5":           (create(data=INT64_0_TO_5),                    "Int64 Int64 Byte",                     distinct(0, 1, 2, 3, 4, 5),  ()),
    "createRaster_uint8_nodata_-1":        (create(data=UINT8_ONES, noData=-1),           "Int16 error Int8",                     distinct(1),                 ()),
    "createRaster_above_2**24_nodata_0.5": (create(data=INT32_ABOVE_2_24, noData=0.5),    "Float64 error Float64",                distinct(2**24 + 1),         ("D9",)),
    "createRaster_nodata_without_data":    (create(noData=5),                             "Byte Byte Byte",                       distinct(5),                 ("M6",)),
    "createRasterLike_float32":            (create_like(float32_fractions),               "Float32 Float32 Byte",                 None,                        ("D6",)),
    "saveRasterAsTif_byte":                (save_as_tif(byte_0_to_127),                   "Byte Byte Byte",                       distinct(0, 50, 100, 127),   ("D4",)),
    "saveRasterAsTif_float64_whole":       (save_as_tif(float64_whole_numbers),           "Float64 Float64 Byte",                 distinct(0, 1, 2, 3),        ()),
    "rasterize_1":                         (burn(1, 0),                                   "Byte Byte Byte",                       distinct(1),                 ()),
    "rasterize_200":                       (burn(200, 0),                                 "Byte Byte Byte",                       distinct(200),               ("D2",)),
    "rasterize_40000":                     (burn(40000, 0),                               "UInt16 UInt16 UInt16",                 distinct(40000),             ("M2",)),
    "rasterize_2**31":                     (burn(2**31, 0),                               "UInt32 UInt32 UInt32",                 distinct(2**31),             ("M2",)),
    "rasterize_-1":                        (burn(-1, 0),                                  "Int8 Int8 Int8",                       distinct(-1),                ()),
    "rasterize_0.1":                       (burn(0.1, 0),                                 "Float64 Float64 Float64",              distinct(0.1),               ("D9",)),
    # three features of 100 can add up to 300, which needs Int16; at most two overlap, so 200 is the maximum
    "rasterize_add_100_three_squares":     (burn(100, 0, 500, 1000, add=True),            "Int16 Byte Byte",                      distinct(100, 200),          ("D16",)),
    "rasterize_int32_field":               (burn_field([1, 2, 3], np.int32),              "Int32 Int32 Byte",                     distinct(0, 1, 2, 3),        ("D1",)),
    "rasterize_int64_field":               (burn_field([1, 2, 3], np.int64),              "Int64 Int64 Byte",                     distinct(0, 1, 2, 3),        ("M5",)),
    "rasterize_real_field_of_396":         (burn_field([7.0, 9.3, 1.392]),                "Float64 Float64 Float64",              distinct(0, 1.392, 7, 9.3),  ("D1",)),
    "rasterize_real_field_float32_exact":  (burn_field([0.5, 1.5, 2.5]),                  "Float64 Float64 Float32",              distinct(0, 0.5, 1.5, 2.5),  ()),
    "rasterize_int64_field_nan_nodata":    (burn_field([123456789, 987654321, 5], np.int64, noData=np.nan),
                                                                                          "Float64+warning error Float64+warning", distinct(5, 123456789, 987654321), ("D8",)),
    "warp_byte_near":                      (warp(byte_mask, "near"),                      "Byte Byte Byte",                       distinct(0, 1),              ("M1",)),
    "warp_byte_average":                   (warp(byte_mask, "average", 400),              "Float32 Byte Float32",                 distinct(0, 0.75, 1),        ("D12",)),
    "warp_byte_default_bilinear":          (warp(byte_mask, "bilinear", 400),             "Float32 Byte Float32",                 None,                        ()),
    "warp_byte_cubic_overshoot":           (warp(byte_step, "cubic", 25),                 "Float32 Byte Float32",                 spans(0, 255),               ("D13",)),
    "warp_byte_sum_16_times_200":          (warp(byte_200s, "sum", 400),                  "Float64 Byte Int16",                   distinct(3200),              ("D14",)),
    "warp_int32_average":                  (warp(int32_0_to_15, "average", 200),          "Float64 Int32 Float32",                None,                        ()),
    "warp_int32_nan_nodata":               (warp(int32_large_ids, "near", noData=np.nan), "Float64 error Float64",                distinct(123456789, 987654321), ("D8",)),
    "mutateRaster_halve":                  (mutate(halve),                                "Float64 Byte Float32",                 distinct(0.5),               ()),
    "mutateRaster_uint8_200":              (mutate(times_200_in_uint8),                   "Byte Byte Byte",                       distinct(200),               ("M11",)),
    "mutateRaster_uint16_40000":           (mutate(times_40000_in_uint16),                "UInt16 Byte UInt16",                   distinct(40000),             ("M11",)),
    "rasterMosaic_byte_200":               (mosaic((200, *BYTE)),                         "Byte Byte Byte",                       distinct(200),               ("D5",)),
    "rasterMosaic_byte_and_int16":         (mosaic((100, *BYTE), (50, *INT16)),           "Int16 Int16 Byte",                     distinct(50, 100),           ()),
    "combineSimilarRasters_byte":          (combine((100, *BYTE), (200, *BYTE)),          "Byte Byte Byte",                       distinct(100, 200),          ("D19",)),
    "combineSimilarRasters_int32_3000_7":  (combine((3000, *INT32), (7, *INT32)),         "Int32 Int32 Int16",                    distinct(7, 3000),           ()),
}

# An explicit dtype is used exactly as given and gives no warning (ADR 2). "error" is a GeoKitDataTypeError.
EXPLICIT_CASES = {
    # name                                 call                             dtype                band type  values               catalogue
    "createRaster-Byte":                  (create(),                        "Byte",              "Byte",    None,                ("D3",)),
    "createRaster-UInt16":                (create(),                        "UInt16",            "UInt16",  None,                ("D3",)),
    "createRaster-UInt32":                (create(),                        "UInt32",            "UInt32",  None,                ("D3",)),
    "quickRaster-Byte":                   (quick(),                         "Byte",              "Byte",    None,                ("D3",)),
    "quickRaster-UInt16":                 (quick(),                         "UInt16",            "UInt16",  None,                ("D3",)),
    "quickRaster-UInt32":                 (quick(),                         "UInt32",            "UInt32",  None,                ("D3",)),
    "quickRaster-np.float32":             (quick(),                         np.float32,          "Float32", None,                ("D7",)),
    "createRaster-np.float32":            (create(),                        np.float32,          "Float32", None,                ("D7",)),
    "createRaster-float":                 (create(),                        float,               "Float64", None,                ("D7",)),
    "createRaster-np.dtype-uint16":       (create(),                        np.dtype("uint16"),  "UInt16",  None,                ("D7",)),
    "createRaster-Float32":               (create(),                        "Float32",           "Float32", None,                ()),
    "createRaster-Int16":                 (create(),                        "Int16",             "Int16",   None,                ()),
    "createRaster-bool":                  (create(),                        bool,                "Byte",    None,                ()),
    "quickRaster-bool":                   (quick(),                         "bool",              "Byte",    None,                ()),
    "createRaster-scalars-that-fit":      (create(noData=65535, fill=1),    "UInt16",            "UInt16",  distinct(1),         ()),
    "createRaster-noData-out-of-range":   (create(noData=-1),               "UInt16",            "error",   None,                ("D10",)),
    "rasterize-Byte":                     (burn(1, 0),                      "Byte",              "Byte",    distinct(1),         ("M3",)),
    "rasterize-np.float32":               (burn(1, 0),                      np.float32,          "Float32", distinct(1),         ("D7",)),
    "warp-Float32-of-a-Float64-source":   (warp(float64_quarters, "near"),  "Float32",           "Float32", distinct(0.25, 0.5), ("D11",)),
    "mutateRaster-bool":                  (mutate(greater_than_five, byte_0_to_127), "bool",     "Byte",    distinct(0, 1),      ("D28",)),
    "combineSimilarRasters-Float32":      (combine((100, *BYTE), (200, *BYTE)), "Float32",   "Float32", distinct(100, 200),  ()),
    "combineSimilarRasters-np.int16":     (combine((100, *BYTE), (200, *BYTE)), np.int16,    "Int16",   distinct(100, 200),  ()),
}
# fmt: on

# Rows of the two tables that are not fixed on this branch, with the reason of their xfail(strict=True) mark. The PR
# that fixes a row deletes its line.
PENDING = {
    # fixed by #416
    "rasterMosaic_byte_200": "D5: rasterMosaic of a Byte source clips 200 to 127",
    "rasterMosaic_byte_and_int16": "rasterMosaic takes the type of the first source only",
    "combineSimilarRasters_byte": "D19: combineSimilarRasters turns Byte into Int8",
    "combineSimilarRasters_int32_3000_7": "combineSimilarRasters has no dtype parameter",
    "combineSimilarRasters-Float32": "combineSimilarRasters has no dtype parameter",
    "combineSimilarRasters-np.int16": "combineSimilarRasters has no dtype parameter",
}

# Pending rows of MODE_CASES whose default call already gives the expected type and values
DEFAULT_CALL_HOLDS = {
    "rasterMosaic_byte_200",
    "combineSimilarRasters_byte",
    "combineSimilarRasters_int32_3000_7",
}


def pending_mark(name, mode=None):
    if name not in PENDING or (mode is None and name in DEFAULT_CALL_HOLDS):
        return ()
    return pytest.mark.xfail(strict=True, reason=PENDING[name])


def mode_case_params():
    for name in MODE_CASES:
        for mode in [None, "auto", "preserve_input", "smallest"]:
            yield pytest.param(name, mode, id=f"{name}-{mode or 'default'}", marks=pending_mark(name, mode))


@pytest.mark.parametrize("name, mode", list(mode_case_params()))
def test_mode_case(name, mode, tmp_path):
    """The call gives the band type of its row in this mode, and keeps the values of its row."""
    build, types, check_values, _ = MODE_CASES[name]
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


@pytest.mark.parametrize("name", [pytest.param(name, marks=pending_mark(name)) for name in EXPLICIT_CASES])
def test_explicit_case(name, tmp_path):
    """An explicit dtype gives exactly that band type, without a warning, or raises a GeoKitDataTypeError."""
    build, dtype, expected, check_values, _ = EXPLICIT_CASES[name]

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


def test_bare_integer_dtype_is_rejected(tmp_path):
    """A bare integer such as gdal.GDT_Float32 raises an error that names the string spelling (ADR 2, #396)."""
    # typeguard, which the test suite runs, would reject the integer at the annotation before GeoKit's own check
    with suppress_type_checks(), pytest.raises(GeoKitDataTypeError, match='dtype="Float32"'):
        create()(tmp_path, dtype=gdal.GDT_Float32)


# ----------------------------------------------------------------------------------------------
# tests outside the two tables


@pins("D15")
def test_D15_warp_warns_about_created_pixels():
    """A reprojection without noData warns that the created pixels hold 0 and are not flagged."""
    int16_raster = gdal_raster(np.arange(1, 401).reshape(20, 20), np.int16, gdal.GDT_Int16, x_min=4e6, y_max=3e6)

    with pytest.warns(GeoKitDataTypeWarning, match="not flagged"):
        warped = geokit.raster.warp(int16_raster, srs=4326, resampleAlg="near")

    assert (geokit.raster.extractMatrix(warped) == 0).any()


@pins("D27")
def test_D27_warp_reads_no_statistics(tmp_path):
    """No .aux.xml file appears next to the source of warp, because it computes no statistics."""
    source_path = tile(tmp_path, "source.tif", 7, np.uint8, gdal.GDT_Byte)

    geokit.raster.warp(source_path, resampleAlg="near")

    assert list(tmp_path.glob("*.aux.xml")) == []


@pending("D27: checkSimilarRasters and combineSimilarRasters compute statistics of their inputs")
@pins("D27")
def test_D27_checkSimilarRasters_and_combineSimilarRasters_read_no_statistics(tmp_path):
    """No .aux.xml file appears next to the inputs of checkSimilarRasters and combineSimilarRasters."""
    from geokit._algorithms.combineSimilarRasters import checkSimilarRasters

    left_tile = tile(tmp_path, "left.tif", 100, np.uint8, gdal.GDT_Byte)
    right_tile = tile(tmp_path, "right.tif", 200, np.uint8, gdal.GDT_Byte, x_min=400)

    checkSimilarRasters([left_tile, right_tile])
    combine((100, *BYTE), (200, *BYTE))(tmp_path)

    assert list(tmp_path.glob("left.tif.aux.xml")) == []
    assert list(tmp_path.glob("right.tif.aux.xml")) == []


@pins("D4")
def test_D4_saveRasterAsTif_keeps_scale_offset_and_nodata(tmp_path):
    """The copy written by saveRasterAsTif keeps the stored values, scale, offset and noData of the source."""
    source = gdal_raster([[-9999, 100, 200]], np.int16, gdal.GDT_Int16, noData=-9999, scale=0.1)
    source.GetRasterBand(1).SetOffset(5.0)

    saved = geokit.raster.saveRasterAsTif(source, str(tmp_path / "saved.tif"))

    saved_dataset = geokit.raster.loadRaster(saved)
    saved_band = saved_dataset.GetRasterBand(1)
    assert band_type_name(saved) == "Int16"
    assert (saved_band.GetScale(), saved_band.GetOffset(), saved_band.GetNoDataValue()) == (0.1, 5.0, -9999)
    np.testing.assert_array_equal(saved_band.ReadAsArray(), [[-9999, 100, 200]])


@pins("M4")
def test_M4_regionmask_gives_byte():
    """RegionMask.createRaster gives a Byte band and RegionMask.rasterize a uint8 matrix."""
    region_mask = geokit.RegionMask.fromGeom(square_polygon(0), pixelRes=100, srs=3035)

    created_raster = region_mask.createRaster()
    rasterized_matrix = region_mask.rasterize(squares(0), value=1)

    assert band_type_name(created_raster) == "Byte"
    assert rasterized_matrix.dtype == np.uint8
    assert rasterized_matrix.max() == 1


# ----------------------------------------------------------------------------------------------
# values handled in NumPy (ADR 8)


def scaled_int16_raster_with_nodata():
    """Int16 pixels -9999, 100 and 200 with noData -9999 and scale 0.1, so the data pixels read 10 and 20."""
    return gdal_raster([[-9999, 100, 200]], np.int16, gdal.GDT_Int16, noData=-9999, scale=0.1)


@pending("D23: extractMatrix(autocorrect=True) compares noData after scaling")
@pins("D23")
def test_D23_extractMatrix_masks_nodata_before_scaling():
    """The noData mask of extractMatrix(autocorrect=True) is built on the stored values, before the scale."""
    matrix = geokit.raster.extractMatrix(scaled_int16_raster_with_nodata(), autocorrect=True)

    assert np.isnan(matrix[0, 0])
    np.testing.assert_allclose(matrix[0, 1:], [10.0, 20.0])


@pending("D23: extractValues compares noData after scaling")
@pins("D23")
def test_D23_extractValues_masks_nodata_before_scaling():
    """The noData mask of extractValues is built on the stored values, before the scale."""
    extracted = geokit.raster.extractValues(scaled_int16_raster_with_nodata(), [(50, 50), (150, 50)], pointSRS=3035)

    assert np.isnan(extracted.data[0])
    assert np.isclose(extracted.data[1], 10.0)


@pending("D23: interpolateValues compares noData after scaling")
@pins("D23")
def test_D23_interpolateValues_masks_nodata_before_scaling():
    """The noData mask of interpolateValues is built on the stored values, before the scale."""
    interpolated = geokit.raster.interpolateValues(
        scaled_int16_raster_with_nodata(), [(50, 50), (150, 50)], pointSRS=3035, mode="near"
    )

    assert np.isnan(interpolated[0])
    assert np.isclose(interpolated[1], 10.0)


@pending("D24: rasterStats treats scaled noData as data")
@pins("D24")
def test_D24_rasterStats_leaves_out_scaled_nodata():
    """The statistics of rasterStats leave out the noData pixels of a scaled raster."""
    stats = geokit.raster.rasterStats(scaled_int16_raster_with_nodata())

    assert stats.nobs == 2
    assert np.isclose(stats.mean, 15.0)


@pending("D17: gradient of a UInt16 DEM wraps around")
@pins("D17")
def test_D17_gradient_of_unsigned_dem():
    """The gradient of a UInt16 elevation model is computed in float and does not wrap around."""
    # the terrain rises 1 m per 100 m pixel towards the east: (100 - 102) m / (2 * 100 m) = -0.01, while in uint16
    # 100 - 102 wraps around to 65534. The edge columns have no neighbour on one side and are left out.
    dem_raster = gdal_raster([[100, 101, 102, 103]] * 4, np.uint16, gdal.GDT_UInt16)

    east_west_gradient = geokit.raster.gradient(dem_raster, mode="east-west", asMatrix=True)

    np.testing.assert_allclose(east_west_gradient[:, 1:-1], -0.01)


@pending("found on the way: gradient(mode='ew') raises UnboundLocalError")
def test_gradient_ew_is_east_west():
    """The gradient with mode='ew' is the same as with mode='east-west'."""
    dem_raster = gdal_raster([[100, 101, 102, 103]] * 4, np.float64, gdal.GDT_Float64)

    short_mode_gradient = geokit.raster.gradient(dem_raster, mode="ew", asMatrix=True)
    long_mode_gradient = geokit.raster.gradient(dem_raster, mode="east-west", asMatrix=True)

    np.testing.assert_array_equal(short_mode_gradient, long_mode_gradient)


@pending("D18: KernelProcessor pads with an integer array and truncates floats")
@pins("D18")
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


@pytest.mark.parametrize("noData", [-1, np.nan], ids=["-1", "nan"])
@pending("D25: indicateValues writes noData into a bool array and indicates every noData pixel")
@pins("D25")
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


@pending("D26: applyMask wraps a NumPy-integer noData into uint8")
@pins("D26")
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


@pending("D20: createVector clips a uint32 above 2**31 - 1")
@pins("D20")
def test_D20_createVector_keeps_uint32_above_2_31():
    """A uint32 value above 2**31 - 1 survives createVector."""
    assert first_field_value(np.array([3_000_000_000], np.uint32)) == 3_000_000_000


@pytest.mark.parametrize(
    "column, expected",
    [(np.array([5], np.uint64), 5), (np.array([1.5], np.float16), 1.5), (pd.array([True], dtype="boolean"), 1)],
    ids=["uint64", "float16", "pandas-boolean"],
)
@pending("D20: createVector writes uint64, float16 and pandas boolean columns as strings")
@pins("D20")
def test_D20_createVector_writes_numeric_columns_as_numbers(column, expected):
    """Columns of uint64, float16 and pandas boolean become numeric fields in createVector, not strings."""
    field_value = first_field_value(column)

    assert not isinstance(field_value, str)
    assert field_value == expected


@pending("D21: polygonizeRaster clips UInt32 above 2**31 - 1")
@pins("D21")
def test_D21_polygonizeRaster_keeps_uint32_above_2_31():
    """UInt32 values above 2**31 - 1 survive polygonizeRaster."""
    uint32_raster = gdal_raster(np.full((4, 4), 3_000_000_000), np.uint32, gdal.GDT_UInt32)

    assert list(geokit.raster.polygonizeRaster(uint32_raster)["value"]) == [3_000_000_000]


@pending("D21: polygonizeRaster truncates float rasters and merges their areas")
@pins("D21")
def test_D21_polygonizeRaster_rejects_float_rasters():
    """A float raster is rejected by polygonizeRaster instead of truncating 1.7 and 2.4 and merging their areas."""
    float_values = np.full((4, 4), 1.7, np.float32)
    float_values[:, 2:] = 2.4
    float_raster = create_gdal_raster(float_values, gdal.GDT_Float32)

    with pytest.raises(GeoKitDataTypeError):
        geokit.raster.polygonizeRaster(float_raster)


@pending("D22: polygonizeMatrix always uses an Int32 band")
@pins("D22")
def test_D22_polygonizeMatrix_keeps_uint32_above_2_31():
    """Values above 2**31 - 1 in a uint32 matrix survive polygonizeMatrix."""
    uint32_matrix = np.full((2, 2), 3_000_000_000, np.uint32)

    assert list(geokit.geom.polygonizeMatrix(uint32_matrix)["value"]) == [3_000_000_000]


@pins("M8")
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
# shared docstring blocks


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


# ----------------------------------------------------------------------------------------------
# coverage of the catalogue

CATALOGUE = {f"D{number}" for number in range(1, 30)} | {f"M{number}" for number in range(1, 12)}

# entries of the catalogue that no test can pin
NOT_PINNED = {
    "D29": "minor issues inside the replaced handler, without effect with the supported GDAL and NumPy",
    "M7": "not a defect: the dtypes of the DataFrames from extractFeatures stay as they are",
    "M9": "not a defect: the statistics pass of D27 never changed a result",
    "M10": "withdrawn: the memory print of indicateValues is wanted",
}


def test_every_catalogue_entry_is_pinned():
    """Every entry of the defect catalogue in #405 is pinned by this module, or listed as not pinnable."""
    pinned = set(PINNED_BY_TESTS)
    for *_, catalogue_ids in [*MODE_CASES.values(), *EXPLICIT_CASES.values()]:
        pinned.update(catalogue_ids)

    assert pinned.isdisjoint(NOT_PINNED)
    assert pinned | set(NOT_PINNED) == CATALOGUE
