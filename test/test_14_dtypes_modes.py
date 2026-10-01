"""The ``dtype`` modes ``"auto"``, ``"preserve_input"`` and ``"smallest"``.

Encodes the worked examples of ADR 1 in ``docs/explanation/data_types/adr_01_dtype_modes.md``. Every
case is run for ``dtype=None`` (which must behave like ``"auto"``) and the three mode strings.

All tests are ``xfail(strict=True)`` until the ``geokit.dtypes`` module and its call sites land.
A case whose expectation is a tuple ``(type, "warning")`` must emit a
``GeoKitDataTypeWarning`` (a ``UserWarning``); ``"error"`` must raise ``GeoKitDataTypeError``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

import geokit
from test.test_13_dtypes_regression import (
    GeoKitDataTypeError,
    band_type_name,
    create_gdal_raster,
    spatial_reference_from_epsg,
    square_polygon,
    write_gdal_geotiff,
)

DTYPE_MODES = [None, "auto", "preserve_input", "smallest"]

XFAIL_UNTIL_DTYPE_MODES_LAND = pytest.mark.xfail(
    strict=True,
    reason="dtype modes of ADR 1 are not implemented yet (#396)",
)

# (case, mode) combinations whose expectation already holds today; they carry no xfail mark so
# that the marks stay strict for the real defects.
CASE_MODE_COMBINATIONS_PASSING_TODAY = {
    ("createRaster_int64_small_values", None),
    ("mutateRaster_float_processor", None),
    ("rasterize_-1", None),
}

POINT_RASTERIZE_GRID = dict(
    pixelWidth=0.5,
    pixelHeight=0.5,
    srs=4326,
    bounds=(5, 50, 8, 53),
)
TEN_BY_TEN_RASTER_GEOMETRY = dict(
    bounds=(0, 0, 10, 10),
    pixelWidth=1,
    pixelHeight=1,
    srs=4326,
)


# ----------------------------------------------------------------------------------------------
# input builders


def _three_points_vector_with_values(values, values_dtype=None):
    point_coordinates = [(6.1, 50.1), (7.5, 52.0), (6.8, 51.4)]
    point_geometries = []
    for x, y in point_coordinates:
        point_geometries.append(geokit.geom.point(x, y, srs=4326))

    if values_dtype is None:
        value_column = values
    else:
        value_column = np.array(values, values_dtype)

    points_table = pd.DataFrame({"geom": point_geometries, "v": value_column})
    return geokit.vector.createVector(points_table)


def _squares_vector(*square_offsets):
    polygons = []
    for square_offset in square_offsets:
        polygons.append(square_polygon(square_offset))
    return geokit.vector.createVector(polygons)


def _byte_mask_raster_left_seven_columns_set():
    mask_values = np.zeros((20, 20), np.uint8)
    mask_values[:, :7] = 1
    return create_gdal_raster(mask_values, gdal.GDT_Byte)


def _byte_raster_filled_with_200():
    pixel_values = np.full((20, 20), 200, np.uint8)
    return create_gdal_raster(pixel_values, gdal.GDT_Byte)


def _byte_raster_filled_with_one():
    pixel_values = np.ones((4, 4), np.uint8)
    return create_gdal_raster(pixel_values, gdal.GDT_Byte)


def _int32_ascending_values_raster():
    pixel_values = np.arange(16, dtype=np.int32).reshape(4, 4)
    return create_gdal_raster(pixel_values, gdal.GDT_Int32)


def _float64_integral_values_raster():
    pixel_values = np.array([[0.0, 1.0], [2.0, 3.0]])
    return create_gdal_raster(pixel_values, gdal.GDT_Float64)


def _halve(pixel_values):
    return pixel_values * 0.5


# ----------------------------------------------------------------------------------------------
# cases: each builder takes (tmp_dir, dtype_argument) and returns the resulting raster


def _create_raster_without_data(_tmp_dir, dtype_argument):
    return geokit.raster.createRaster(
        dtype=dtype_argument,
        **TEN_BY_TEN_RASTER_GEOMETRY,
    )


def _create_raster_from_int64_data_with_small_values(_tmp_dir, dtype_argument):
    ascending_values = np.arange(100, dtype=np.int64).reshape(10, 10)
    small_values = ascending_values % 6
    return geokit.raster.createRaster(
        data=small_values,
        dtype=dtype_argument,
        **TEN_BY_TEN_RASTER_GEOMETRY,
    )


def _create_raster_from_uint8_data_with_nodata_minus_one(_tmp_dir, dtype_argument):
    uint8_ones = np.ones((10, 10), np.uint8)
    return geokit.raster.createRaster(
        data=uint8_ones,
        noData=-1,
        dtype=dtype_argument,
        **TEN_BY_TEN_RASTER_GEOMETRY,
    )


def _rasterize_square_with_value(burn_value, dtype_argument):
    square_vector = _squares_vector(0)
    return geokit.vector.rasterize(
        square_vector,
        100,
        100,
        value=burn_value,
        dtype=dtype_argument,
    )


def _rasterize_square_with_value_1(_tmp_dir, dtype_argument):
    return _rasterize_square_with_value(1, dtype_argument)


def _rasterize_square_with_value_200(_tmp_dir, dtype_argument):
    return _rasterize_square_with_value(200, dtype_argument)


def _rasterize_square_with_value_40000(_tmp_dir, dtype_argument):
    return _rasterize_square_with_value(40000, dtype_argument)


def _rasterize_square_with_value_minus_one(_tmp_dir, dtype_argument):
    return _rasterize_square_with_value(-1, dtype_argument)


def _rasterize_square_with_value_zero_point_one(_tmp_dir, dtype_argument):
    return _rasterize_square_with_value(0.1, dtype_argument)


def _rasterize_int32_field(_tmp_dir, dtype_argument):
    points_vector = _three_points_vector_with_values([1, 2, 3], np.int32)
    return geokit.vector.rasterize(
        points_vector,
        value="v",
        dtype=dtype_argument,
        **POINT_RASTERIZE_GRID,
    )


def _rasterize_real_field(_tmp_dir, dtype_argument):
    points_vector = _three_points_vector_with_values([0.5, 1.5, 2.5])
    return geokit.vector.rasterize(
        points_vector,
        value="v",
        dtype=dtype_argument,
        **POINT_RASTERIZE_GRID,
    )


def _rasterize_int64_field_with_nan_nodata(_tmp_dir, dtype_argument):
    points_vector = _three_points_vector_with_values([1, 2, 3], np.int64)
    return geokit.vector.rasterize(
        points_vector,
        value="v",
        noData=np.nan,
        dtype=dtype_argument,
        **POINT_RASTERIZE_GRID,
    )


def _rasterize_three_squares_adding_100_each(_tmp_dir, dtype_argument):
    # Three features of 100 can add up to 300, which needs Int16. At most two of the squares overlap, so the
    # true maximum is 200, which fits Byte.
    three_squares_vector = _squares_vector(0, 500, 1000)
    return geokit.vector.rasterize(
        three_squares_vector,
        100,
        100,
        value=100,
        add=True,
        dtype=dtype_argument,
    )


def _warp_byte_mask_with_nearest_neighbour(_tmp_dir, dtype_argument):
    byte_mask_raster = _byte_mask_raster_left_seven_columns_set()
    return geokit.raster.warp(
        byte_mask_raster,
        resampleAlg="near",
        dtype=dtype_argument,
    )


def _warp_byte_mask_with_average(_tmp_dir, dtype_argument):
    byte_mask_raster = _byte_mask_raster_left_seven_columns_set()
    return geokit.raster.warp(
        byte_mask_raster,
        resampleAlg="average",
        pixelWidth=400,
        pixelHeight=400,
        dtype=dtype_argument,
    )


def _warp_byte_mask_with_default_resampling(_tmp_dir, dtype_argument):
    byte_mask_raster = _byte_mask_raster_left_seven_columns_set()
    return geokit.raster.warp(
        byte_mask_raster,
        pixelWidth=400,
        pixelHeight=400,
        dtype=dtype_argument,
    )


def _warp_int32_raster_with_average(_tmp_dir, dtype_argument):
    int32_raster = _int32_ascending_values_raster()
    return geokit.raster.warp(
        int32_raster,
        resampleAlg="average",
        pixelWidth=200,
        pixelHeight=200,
        dtype=dtype_argument,
    )


def _warp_byte_raster_of_200s_with_sum(_tmp_dir, dtype_argument):
    byte_raster = _byte_raster_filled_with_200()
    return geokit.raster.warp(
        byte_raster,
        resampleAlg="sum",
        pixelWidth=400,
        pixelHeight=400,
        dtype=dtype_argument,
    )


def _mutate_byte_raster_with_float_processor(_tmp_dir, dtype_argument):
    byte_raster = _byte_raster_filled_with_one()
    return geokit.raster.mutateRaster(
        byte_raster,
        processor=_halve,
        dtype=dtype_argument,
    )


def _mosaic_byte_and_int16_tiles(tmp_dir, dtype_argument):
    byte_tile_path = tmp_dir / "byte_tile.tif"
    byte_tile_values = np.full((4, 4), 100, np.uint8)
    byte_tile = write_gdal_geotiff(byte_tile_path, byte_tile_values, gdal.GDT_Byte)

    int16_tile_path = tmp_dir / "int16_tile.tif"
    int16_tile_values = np.full((4, 4), 50, np.int16)
    int16_tile = write_gdal_geotiff(int16_tile_path, int16_tile_values, gdal.GDT_Int16, x_min=400)

    mosaic_extent = geokit.Extent(0, 0, 800, 400, srs=3035)
    return mosaic_extent.rasterMosaic(
        [byte_tile, int16_tile],
        _skipFiltering=True,
        dtype=dtype_argument,
    )


def _save_float64_raster_with_integral_values_as_tif(tmp_dir, dtype_argument):
    float64_raster = _float64_integral_values_raster()
    output_path = str(tmp_dir / "saved_raster.tif")
    return geokit.raster.saveRasterAsTif(
        float64_raster,
        output_path,
        dtype=dtype_argument,
    )


# case name -> (builder, {mode: expected})
#   expected is a GDAL type name, (type name, "warning") or "error"
EXPECTED_BAND_TYPE_PER_MODE_BY_CASE = {
    "createRaster_empty": (
        _create_raster_without_data,
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "createRaster_int64_small_values": (
        _create_raster_from_int64_data_with_small_values,
        {"auto": "Int64", "preserve_input": "Int64", "smallest": "Byte"},
    ),
    "createRaster_uint8_nodata_-1": (
        _create_raster_from_uint8_data_with_nodata_minus_one,
        {"auto": "Int16", "preserve_input": "error", "smallest": "Int8"},
    ),
    "rasterize_1": (
        _rasterize_square_with_value_1,
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "rasterize_200": (
        _rasterize_square_with_value_200,
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "rasterize_40000": (
        _rasterize_square_with_value_40000,
        {"auto": "UInt16", "preserve_input": "UInt16", "smallest": "UInt16"},
    ),
    "rasterize_-1": (
        _rasterize_square_with_value_minus_one,
        {"auto": "Int8", "preserve_input": "Int8", "smallest": "Int8"},
    ),
    "rasterize_0.1": (
        _rasterize_square_with_value_zero_point_one,
        {"auto": "Float64", "preserve_input": "Float64", "smallest": "Float64"},
    ),
    "rasterize_integer_field": (
        _rasterize_int32_field,
        {"auto": "Int32", "preserve_input": "Int32", "smallest": "Byte"},
    ),
    "rasterize_real_field": (
        _rasterize_real_field,
        {"auto": "Float64", "preserve_input": "Float64", "smallest": "Float32"},
    ),
    "rasterize_int64_field_nan_nodata": (
        _rasterize_int64_field_with_nan_nodata,
        {
            "auto": ("Float64", "warning"),
            "preserve_input": "error",
            "smallest": ("Float64", "warning"),
        },
    ),
    "rasterize_add_100_x3": (
        _rasterize_three_squares_adding_100_each,
        {"auto": "Int16", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "warp_byte_near": (
        _warp_byte_mask_with_nearest_neighbour,
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "warp_byte_average": (
        _warp_byte_mask_with_average,
        {"auto": "Float32", "preserve_input": "Byte", "smallest": "Float32"},
    ),
    "warp_byte_bilinear_default": (
        _warp_byte_mask_with_default_resampling,
        {"auto": "Float32", "preserve_input": "Byte", "smallest": "Float32"},
    ),
    "warp_int32_average": (
        _warp_int32_raster_with_average,
        {"auto": "Float64", "preserve_input": "Int32", "smallest": "Float32"},
    ),
    "warp_byte_sum_16x200": (
        _warp_byte_raster_of_200s_with_sum,
        {"auto": "Float64", "preserve_input": "Byte", "smallest": "Int16"},
    ),
    "mutateRaster_float_processor": (
        _mutate_byte_raster_with_float_processor,
        {"auto": "Float64", "preserve_input": "Byte", "smallest": "Float32"},
    ),
    "rasterMosaic_byte_int16": (
        _mosaic_byte_and_int16_tiles,
        {"auto": "Int16", "preserve_input": "Int16", "smallest": "Byte"},
    ),
    "saveRasterAsTif_float64_integral": (
        _save_float64_raster_with_integral_values_as_tif,
        {"auto": "Float64", "preserve_input": "Float64", "smallest": "Byte"},
    ),
}


def _case_and_mode_params():
    for case_name in EXPECTED_BAND_TYPE_PER_MODE_BY_CASE:
        for dtype_mode in DTYPE_MODES:
            case_and_mode = (case_name, dtype_mode)
            if case_and_mode in CASE_MODE_COMBINATIONS_PASSING_TODAY:
                marks = ()
            else:
                marks = (XFAIL_UNTIL_DTYPE_MODES_LAND,)

            yield pytest.param(
                case_name,
                dtype_mode,
                id=f"{case_name}-{dtype_mode}",
                marks=marks,
            )


ALL_CASE_AND_MODE_PARAMS = list(_case_and_mode_params())


@pytest.mark.parametrize("case_name, dtype_mode", ALL_CASE_AND_MODE_PARAMS)
def test_band_type_matches_expectation_for_each_dtype_mode(case_name, dtype_mode, tmp_path):
    create_raster, expected_by_mode = EXPECTED_BAND_TYPE_PER_MODE_BY_CASE[case_name]

    mode_key = "auto" if dtype_mode is None else dtype_mode
    expected = expected_by_mode[mode_key]

    if expected == "error":
        with pytest.raises(GeoKitDataTypeError):
            create_raster(tmp_path, dtype_mode)
        return

    if isinstance(expected, tuple):
        expected_band_type, _ = expected
        with pytest.warns(UserWarning, match="dtype"):
            result_raster = create_raster(tmp_path, dtype_mode)
    else:
        expected_band_type = expected
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            result_raster = create_raster(tmp_path, dtype_mode)

    assert band_type_name(result_raster) == expected_band_type


# ----------------------------------------------------------------------------------------------
# explicit dtype next to the modes


@XFAIL_UNTIL_DTYPE_MODES_LAND
def test_explicit_narrower_float_is_used_without_warning():
    """An explicit dtype narrower than the source is used as given, without a warning: explicit types are not checked."""
    float64_values = np.array([[0.1, 0.2]], np.float64)
    float64_source_raster = create_gdal_raster(float64_values, gdal.GDT_Float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        warped_raster = geokit.raster.warp(
            float64_source_raster,
            dtype="Float32",
            resampleAlg="near",
        )

    assert band_type_name(warped_raster) == "Float32"


def test_explicit_dtype_with_scalars_that_fit_is_silent():
    """An explicit dtype that holds every scalar is used as is, silently; this holds today and must keep holding."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        uint16_raster = geokit.raster.createRaster(
            dtype="UInt16",
            noData=65535,
            fill=1,
            **TEN_BY_TEN_RASTER_GEOMETRY,
        )

    assert band_type_name(uint16_raster) == "UInt16"

    pixel_values = geokit.raster.extractMatrix(uint16_raster)
    distinct_pixel_values = set(np.unique(pixel_values))
    assert distinct_pixel_values == {1}


@XFAIL_UNTIL_DTYPE_MODES_LAND
def test_smallest_keeps_nodata_representable():
    """'smallest' shrinks to a type that still holds noData: the values fit Byte, but noData=-1 needs Int8."""
    int32_ones = np.ones((10, 10), np.int32)
    shrunk_raster = geokit.raster.createRaster(
        data=int32_ones,
        noData=-1,
        dtype="smallest",
        **TEN_BY_TEN_RASTER_GEOMETRY,
    )

    assert band_type_name(shrunk_raster) == "Int8"


@XFAIL_UNTIL_DTYPE_MODES_LAND
def test_bool_dtype_is_byte():
    """dtype=bool and dtype='bool' give Byte in createRaster and quickRaster."""
    created_raster = geokit.raster.createRaster(
        dtype=bool,
        **TEN_BY_TEN_RASTER_GEOMETRY,
    )
    assert band_type_name(created_raster) == "Byte"

    wgs84 = spatial_reference_from_epsg(4326)
    quick_raster = geokit.util.quickRaster(
        bounds=(0, 0, 10, 10),
        srs=wgs84,
        dx=1,
        dy=1,
        dtype="bool",
    )
    assert band_type_name(quick_raster) == "Byte"
