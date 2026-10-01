"""Regression tests for the data-type defects of issue #396.

The defects are listed in the defect catalogue of GitHub issue #405. Test names
and xfail reasons start with the catalogue ID (D1-D28, M1-M11). Every test asserts *values*, not only
types.

Each test carries ``xfail(strict=True)`` with the defect ID. The PR that fixes a defect removes
the mark; because the marks are strict, a test that starts passing unexpectedly fails the run,
so the marks cannot go stale.

The rasters used here are built with plain GDAL so that GeoKit's type logic is not involved in
the inputs.
"""

import os
import pathlib
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, gdal_array, ogr, osr

import geokit.vector

try:
    from geokit.error import GeoKitDataTypeError
except ImportError:  # GeoKitDataTypeError lands with the dtype modes; the tests that expect it are xfail until then

    class GeoKitDataTypeError(Exception):
        """Stand-in for geokit.error.GeoKitDataTypeError until it exists; GeoKit never raises it."""


# ----------------------------------------------------------------------------------------------
# helpers


def xfail_until_fixed(defect: str):
    return pytest.mark.xfail(strict=True, reason=defect)


def spatial_reference_from_epsg(epsg: int) -> osr.SpatialReference:
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(epsg)
    if gdal.__version__ >= "3.0.0":
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_reference


def create_gdal_raster(
    matrix,
    gdal_type,
    noData=None,
    scale=None,
    epsg=3035,
    pixel_size=100,
    x_min=0.0,
    y_max=None,
    driver="MEM",
    path="",
):
    """A raster built with plain GDAL. ``y_max`` is the top edge (defaults to rows * pixel_size)."""
    rows, columns = matrix.shape
    if y_max is None:
        y_max = rows * pixel_size

    dataset = gdal.GetDriverByName(driver).Create(path, columns, rows, 1, gdal_type)
    dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    dataset.SetProjection(spatial_reference_from_epsg(epsg).ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.WriteArray(matrix)
    if noData is not None:
        band.SetNoDataValue(noData)
    if scale is not None:
        band.SetScale(scale)
    band.FlushCache()
    return dataset


def write_gdal_geotiff(path, matrix, gdal_type, **raster_kwargs):
    """Write ``matrix`` to a GeoTIFF with plain GDAL and return its path as a string."""
    dataset = create_gdal_raster(matrix, gdal_type, driver="GTiff", path=str(path), **raster_kwargs)
    dataset = None  # closing the dataset flushes it to disk
    return str(path)


def band_type_name(raster) -> str:
    dataset = geokit.raster.loadRaster(raster)
    gdal_type = dataset.GetRasterBand(1).DataType
    return gdal.GetDataTypeName(gdal_type)


def band_numpy_dtype(raster) -> np.dtype:
    dataset = geokit.raster.loadRaster(raster)
    gdal_type = dataset.GetRasterBand(1).DataType
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def gdal_warp_to_float64_array(source, resample_alg, pixel_size, **warp_kwargs):
    """Reference: the same warp done by plain GDAL into Float64, returned as an array."""
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        xRes=pixel_size,
        yRes=pixel_size,
        resampleAlg=resample_alg,
        outputType=gdal.GDT_Float64,
        **warp_kwargs,
    )
    return warped.ReadAsArray()


GEOM_THREE_POINTS = [geokit.geom.point(x, y, srs=4326) for x, y in [(6.1, 50.1), (7.5, 52.0), (6.8, 51.4)]]
GRID_KWARGS = dict(pixelWidth=0.5, pixelHeight=0.5, srs=4326, bounds=(5, 50, 8, 53))
TEN_BY_TEN_DEGREE_GRID_KWARGS = dict(bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326)


def square_polygon(x_min: float, size: float = 1000.0) -> ogr.Geometry:
    corners = [(x_min, 0), (x_min + size, 0), (x_min + size, size), (x_min, size), (x_min, 0)]
    return geokit.geom.polygon(corners, srs=3035)


# ----------------------------------------------------------------------------------------------
# A. Type representation


@xfail_until_fixed("D1: float attribute rasterized as Int16, decimals truncated (#396)")
def test_D1_rasterize_float_attribute_exact():
    """A vector with three points as geometries are rasterized into a raster with intact Float64 pixels."""
    float_values = np.array([7.0, 9.3, 1.392], np.float64)
    attributes = pd.DataFrame({"geom": GEOM_THREE_POINTS, "v": float_values})
    point_vector = geokit.vector.createVector(attributes)

    raster = geokit.vector.rasterize(point_vector, value="v", **GRID_KWARGS)

    matrix = geokit.raster.extractMatrix(raster)
    assert band_type_name(raster) == "Float64"
    assert set(np.unique(matrix)) == {0.0, 1.392, 7.0, 9.3}


@xfail_until_fixed("D1: integer attribute raises GeoKitCDataError('Unknown')")
def test_D1_rasterize_int16_attribute():
    """An int16 column rasterizes without error into an integer band that holds the values exactly.

    Only "integer" is asserted: the width depends on how the column is stored as an OGR field.
    """
    int16_values = np.array([7, 9, 1], np.int16)
    attributes = pd.DataFrame({"geom": GEOM_THREE_POINTS, "v": int16_values})
    point_vector = geokit.vector.createVector(attributes)

    raster = geokit.vector.rasterize(point_vector, value="v", **GRID_KWARGS)

    matrix = geokit.raster.extractMatrix(raster)
    assert np.issubdtype(band_numpy_dtype(raster), np.integer)
    assert set(np.unique(matrix)) == {0, 1, 7, 9}


@xfail_until_fixed("D2: rasterize(value=200) burns 127 into an Int8 raster")
def test_D2_rasterize_constant_200():
    """A constant burn value of 200 gives a Byte band holding 200, not an Int8 band holding 127 when using rasterize."""
    square_vector = geokit.vector.createVector([square_polygon(0)])

    raster = geokit.vector.rasterize(square_vector, pixelWidth=100, pixelHeight=100, value=200)

    matrix = geokit.raster.extractMatrix(raster)
    assert band_type_name(raster) == "Byte"
    assert matrix.max() == 200


@pytest.mark.parametrize(
    "value, expected_type",
    [(40000, "UInt16"), (2**31, "UInt32")],
    ids=["40000", "2**31"],
)
@xfail_until_fixed("M2: every unsigned-only constant is clipped to the signed maximum")
def test_M2_rasterize_constant_unsigned_ranges(value, expected_type):
    """A constant above the signed maximum gets the unsigned type that holds it."""
    square_vector = geokit.vector.createVector([square_polygon(0)])

    raster = geokit.vector.rasterize(square_vector, pixelWidth=100, pixelHeight=100, value=value)

    matrix = geokit.raster.extractMatrix(raster)
    assert band_type_name(raster) == expected_type
    assert matrix.max() == value


@xfail_until_fixed("M3: rasterize(value=1, dtype='Byte') returns Int8")
def test_M3_rasterize_explicit_byte():
    """The explicit dtype='Byte' is honoured by rasterize instead of becoming Int8."""
    square_vector = geokit.vector.createVector([square_polygon(0)])

    raster = geokit.vector.rasterize(square_vector, pixelWidth=100, pixelHeight=100, value=1, dtype="Byte")

    matrix = geokit.raster.extractMatrix(raster)
    assert band_type_name(raster) == "Byte"
    assert matrix.max() == 1


@pytest.mark.parametrize("dtype", ["Byte", "UInt16", "UInt32"])
@xfail_until_fixed("D3: explicit unsigned types are turned into signed ones")
def test_D3_createRaster_explicit_unsigned(dtype):
    """Explicit unsigned dtypes are honoured by createRaster and quickRaster."""
    created_raster = geokit.raster.createRaster(**TEN_BY_TEN_DEGREE_GRID_KWARGS, dtype=dtype)

    quick_raster = geokit.util.quickRaster(
        bounds=(0, 0, 10, 10),
        srs=spatial_reference_from_epsg(4326),
        dx=1,
        dy=1,
        dtype=dtype,
    )

    assert band_type_name(created_raster) == dtype
    assert band_type_name(quick_raster) == dtype


@xfail_until_fixed("D4: saveRasterAsTif writes a Byte raster as Int8 when every value fits Int8")
def test_D4_saveRasterAsTif_keeps_type(tmp_path):
    """A Byte source stays Byte: saveRasterAsTif writes an exact copy."""
    uint8_values = np.array([[0, 50], [100, 127]], np.uint8)
    source_raster = create_gdal_raster(uint8_values, gdal.GDT_Byte)

    saved_raster = geokit.raster.saveRasterAsTif(source_raster, str(tmp_path / "saved.tif"))

    matrix = geokit.raster.extractMatrix(saved_raster)
    assert band_type_name(saved_raster) == "Byte"
    assert matrix.max() == 127


@xfail_until_fixed("D5: rasterMosaic clips 200 to 127")
def test_D5_rasterMosaic_keeps_byte():
    """The Byte type of the source survives rasterMosaic, so 200 is not clipped to 127."""
    uint8_values_of_200 = np.full((4, 4), 200, np.uint8)
    source_raster = create_gdal_raster(uint8_values_of_200, gdal.GDT_Byte)
    extent = geokit.Extent(0, 0, 400, 400, srs=3035)

    mosaic = extent.rasterMosaic([source_raster], _skipFiltering=True)

    matrix = geokit.raster.extractMatrix(mosaic)
    assert band_type_name(mosaic) == "Byte"
    assert matrix.max() == 200


@xfail_until_fixed("D6: createRasterLike does not copy the dtype")
def test_D6_createRasterLike_inherits_dtype():
    """Without data, createRasterLike copies the source dtype."""
    float32_values = np.array([[0.5, 2.5]], np.float32)
    source_raster = create_gdal_raster(float32_values, gdal.GDT_Float32)

    like_raster = geokit.raster.createRasterLike(source_raster)

    assert band_type_name(like_raster) == "Float32"


@pytest.mark.parametrize(
    "dtype, expected_type",
    [(np.float32, "Float32"), (float, "Float64"), (np.dtype("uint16"), "UInt16")],
)
@xfail_until_fixed("D7: non-string dtype is ignored")
def test_D7_createRaster_numpy_dtype(dtype, expected_type):
    """NumPy dtypes and Python float are accepted as dtype by createRaster."""
    raster = geokit.raster.createRaster(**TEN_BY_TEN_DEGREE_GRID_KWARGS, dtype=dtype)

    assert band_type_name(raster) == expected_type


@xfail_until_fixed("ADR 2: a bare integer dtype must be rejected (GDAL and OGR constants overlap)")
def test_bare_int_dtype_rejected():
    """A bare integer dtype, like gdal.GDT_Float32=6, is rejected: GDAL and OGR constants overlap, which caused #396."""
    with pytest.raises(GeoKitDataTypeError):
        geokit.raster.createRaster(**TEN_BY_TEN_DEGREE_GRID_KWARGS, dtype=gdal.GDT_Float32)


@pytest.mark.parametrize("dtype", ["Float32", "Float64", "Int16", "Int32"])
def test_string_dtype_accepted(dtype):
    """The string spelling of a GDAL type, e.g. dtype='Float32', is accepted and honoured."""
    raster = geokit.raster.createRaster(**TEN_BY_TEN_DEGREE_GRID_KWARGS, dtype=dtype)

    assert band_type_name(raster) == dtype


# ----------------------------------------------------------------------------------------------
# B. Promotion rules


@xfail_until_fixed("D8: integer IDs with noData=nan are stored as Float32 and lose precision")
def test_D8_rasterize_int_ids_nan_nodata_exact():
    """Integer IDs rasterized with noData=NaN go to Float64 and stay exact."""
    int64_ids = np.array([123456789, 987654321, 5], np.int64)
    attributes = pd.DataFrame({"geom": GEOM_THREE_POINTS, "id": int64_ids})
    point_vector = geokit.vector.createVector(attributes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Int64 + NaN may warn (exact up to 2**53)
        raster = geokit.vector.rasterize(point_vector, value="id", noData=np.nan, **GRID_KWARGS)

    matrix = geokit.raster.extractMatrix(raster)
    data_values = matrix[~np.isnan(matrix)]
    assert band_type_name(raster) == "Float64"
    assert {123456789.0, 987654321.0, 5.0} <= set(np.unique(data_values))


@xfail_until_fixed("D8: warp of Int32 with noData=nan gives Float32")
def test_D8_warp_int32_nan_nodata_exact():
    """Warping Int32 with noData=NaN gives Float64, not a Float32 that rounds the values."""
    int32_values = np.array([[123456789, 987654321]], np.int32)
    source_raster = create_gdal_raster(int32_values, gdal.GDT_Int32)

    warped = geokit.raster.warp(source_raster, noData=np.nan, resampleAlg="near")

    matrix = geokit.raster.extractMatrix(warped)
    assert band_type_name(warped) == "Float64"
    assert set(np.unique(matrix)) == {123456789.0, 987654321.0}


@xfail_until_fixed("D10: an explicit dtype that cannot hold noData is silently widened")
def test_D10_explicit_dtype_cannot_hold_nodata_raises():
    """An explicit dtype that cannot hold noData raises instead of being widened silently."""
    with pytest.raises(GeoKitDataTypeError):
        geokit.raster.createRaster(**TEN_BY_TEN_DEGREE_GRID_KWARGS, dtype="UInt16", noData=-1)


@xfail_until_fixed("D11: warp(dtype='Float32') on a Float64 source returns Float64 with a warning")
def test_D11_warp_explicit_narrower_float():
    """warp(dtype='Float32') on a Float64 source returns Float32."""
    float64_values = np.array([[0.5, 0.25]], np.float64)
    source_raster = create_gdal_raster(float64_values, gdal.GDT_Float64)

    warped = geokit.raster.warp(source_raster, dtype="Float32", resampleAlg="near")

    matrix = geokit.raster.extractMatrix(warped)
    assert band_type_name(warped) == "Float32"
    assert set(np.unique(matrix)) == {0.25, 0.5}


# ----------------------------------------------------------------------------------------------
# C. Operations that widen the value range


@xfail_until_fixed("D12: averaging a 0/1 mask stores 0.75 as 1")
def test_D12_warp_average_fractional():
    """Averaging a 0/1 mask keeps the fractions; the reference is gdal.Warp into Float64."""
    mask_values = np.zeros((20, 20), np.uint8)
    mask_values[:, :7] = 1
    source_raster = create_gdal_raster(mask_values, gdal.GDT_Byte)

    warped = geokit.raster.warp(source_raster, resampleAlg="average", pixelWidth=400, pixelHeight=400)

    actual = geokit.raster.extractMatrix(warped)
    expected = gdal_warp_to_float64_array(source_raster, "average", 400)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@xfail_until_fixed("D13: cubic overshoot is clipped to the Byte range")
def test_D13_warp_cubic_overshoot_kept():
    """Cubic overshoot outside the Byte range is kept; the reference is gdal.Warp into Float64."""
    step_values = np.zeros((20, 20), np.uint8)
    step_values[:, 10:] = 255
    source_raster = create_gdal_raster(step_values, gdal.GDT_Byte)

    warped = geokit.raster.warp(source_raster, resampleAlg="cubic", pixelWidth=30, pixelHeight=30)

    actual = geokit.raster.extractMatrix(warped)
    expected = gdal_warp_to_float64_array(source_raster, "cubic", 30)
    assert expected.min() < 0 < 255 < expected.max()  # the reference really overshoots
    np.testing.assert_allclose(actual, expected, atol=1e-3)


@xfail_until_fixed("D14: warp(sum) into Byte clips 3200 to 255")
def test_D14_warp_sum_not_clipped():
    """warp(sum) of 16 Byte pixels of 200 gives 3200 instead of clipping at 255."""
    uint8_values_of_200 = np.full((20, 20), 200, np.uint8)
    source_raster = create_gdal_raster(uint8_values_of_200, gdal.GDT_Byte, pixel_size=100)

    warped = geokit.raster.warp(source_raster, resampleAlg="sum", pixelWidth=400, pixelHeight=400)

    actual = geokit.raster.extractMatrix(warped)
    expected = gdal_warp_to_float64_array(source_raster, "sum", 400)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)
    assert actual.max() == 3200


@xfail_until_fixed("D15: reprojection creates unflagged zero pixels without any warning")
def test_D15_warp_reprojection_warns_about_created_pixels():
    """A reprojection without noData warns that the created pixels hold 0 instead of an explicitly set noData value."""
    int16_values = np.arange(1, 401, dtype=np.int16).reshape(20, 20)
    source_raster = create_gdal_raster(int16_values, gdal.GDT_Int16, x_min=4_000_000, y_max=3_000_000)

    with pytest.warns(UserWarning, match="not flagged"):
        warped = geokit.raster.warp(source_raster, srs=4326, resampleAlg="near")

    matrix = geokit.raster.extractMatrix(warped)
    assert (matrix == 0).any()  # GDAL behaviour is kept; the user is told about it


@xfail_until_fixed("D16: rasterize(add=True) overflows Int8")
def test_D16_rasterize_add_overflow():
    """rasterize(add=True) sums overlapping features without overflowing the band type."""
    overlapping_squares_vector = geokit.vector.createVector([square_polygon(0), square_polygon(500)])

    raster = geokit.vector.rasterize(overlapping_squares_vector, pixelWidth=100, pixelHeight=100, value=100, add=True)

    matrix = geokit.raster.extractMatrix(raster)
    assert matrix.max() == 200


@xfail_until_fixed("D17: gradient of a UInt16 DEM wraps around")
def test_D17_gradient_unsigned():
    """The gradient of a UInt16 DEM (digital elevation model) is computed in float, so it does not wrap around."""
    # The terrain rises by 1 m per pixel from west to east; the pixels are 100 m wide.
    uint16_dem_values = np.array([[100, 101, 102, 103]] * 4, np.uint16)
    dem_raster = create_gdal_raster(uint16_dem_values, gdal.GDT_UInt16)

    east_west_gradient = geokit.raster.gradient(dem_raster, mode="east-west", asMatrix=True)

    # Central difference over two pixels: (100 - 102) m / (2 * 100 m) = -0.01. The sign is negative because the
    # terrain rises towards the east, so the slope faces west. Computed in UInt16, 100 - 102 would wrap around to
    # 65534 and the gradient would be a huge positive number instead.
    # The edge columns are left out, as they have no neighbour on one side.
    interior_columns = east_west_gradient[:, 1:-1]
    np.testing.assert_allclose(interior_columns, -0.01)


@xfail_until_fixed("incidental: gradient(mode='ew') raises UnboundLocalError")
def test_gradient_ew_mode():
    """gradient(mode='ew') is the same as mode='east-west'."""
    float64_dem_values = np.array([[100.0, 101.0, 102.0, 103.0]] * 4)
    dem_raster = create_gdal_raster(float64_dem_values, gdal.GDT_Float64)

    gradient_from_short_mode = geokit.raster.gradient(dem_raster, mode="ew", asMatrix=True)
    gradient_from_long_mode = geokit.raster.gradient(dem_raster, mode="east-west", asMatrix=True)

    np.testing.assert_array_equal(gradient_from_short_mode, gradient_from_long_mode)


@xfail_until_fixed("D18: KernelProcessor pads with an integer array and truncates floats")
def test_D18_kernel_processor_float_padding():
    """KernelProcessor pads in the matrix dtype, so an integer edgeValue does not truncate floats."""
    float_matrix = np.array([[0.5, 1.5], [2.5, 3.5]])

    # Returning the centre pixel makes the processor an identity, so only the padding can change the values.
    # The windows are cut from the padded matrix, so their dtype is the padding dtype.
    window_dtypes = []

    def center_pixel_of_window(window):
        window_dtypes.append(window.dtype)
        return window[1, 1]

    # The integer edgeValue gives an int64 padded matrix, which truncates the floats copied into it.
    kernel_processor = geokit.util.KernelProcessor(1, edgeValue=0)
    process_every_window = kernel_processor(center_pixel_of_window)

    output = process_every_window(float_matrix)

    assert set(window_dtypes) == {np.dtype(np.float64)}
    assert output.dtype == np.float64
    np.testing.assert_array_equal(output, float_matrix)  # truncated, it would be [[0, 1], [2, 3]]


@xfail_until_fixed("D19: combineSimilarRasters turns Byte into Int8")
def test_D19_combineSimilarRasters_keeps_byte(tmp_path):
    """Byte and the values of both inputs survive combineSimilarRasters."""
    from geokit._algorithms.combineSimilarRasters import combineSimilarRasters

    left_tile = write_gdal_geotiff(tmp_path / "a.tif", np.full((2, 2), 100, np.uint8), gdal.GDT_Byte)
    right_tile = write_gdal_geotiff(tmp_path / "b.tif", np.full((2, 2), 200, np.uint8), gdal.GDT_Byte, x_min=200)
    combined_path = str(tmp_path / "combined.tif")

    combineSimilarRasters([left_tile, right_tile], output=combined_path, verbose=False)

    matrix = geokit.raster.extractMatrix(combined_path)
    assert band_type_name(combined_path) == "Byte"
    assert set(np.unique(matrix)) == {100, 200}


@xfail_until_fixed("M1: warp(near) of a Byte raster returns Int8")
def test_M1_warp_near_keeps_byte():
    """warp(near) keeps the Byte type of the source."""
    uint8_values = np.array([[0, 1], [2, 3]], np.uint8)
    source_raster = create_gdal_raster(uint8_values, gdal.GDT_Byte)

    warped = geokit.raster.warp(source_raster, resampleAlg="near")

    assert band_type_name(warped) == "Byte"


@pytest.mark.parametrize(
    "dtype, value, expected_type",
    [(np.uint8, 200, "Byte"), (np.uint16, 40000, "UInt16")],
    ids=["uint8", "uint16"],
)
@xfail_until_fixed("M11: mutateRaster clips unsigned processor output")
def test_M11_mutateRaster_unsigned_output(dtype, value, expected_type):
    """An unsigned processor output is kept by mutateRaster instead of being clipped to the signed range."""
    uint8_ones = np.ones((4, 4), np.uint8)
    source_raster = create_gdal_raster(uint8_ones, gdal.GDT_Byte)

    def multiply_into_unsigned_dtype(matrix):
        unsigned_matrix = matrix.astype(dtype)
        return unsigned_matrix * dtype(value)

    mutated = geokit.raster.mutateRaster(source_raster, processor=multiply_into_unsigned_dtype)

    matrix = geokit.raster.extractMatrix(mutated)
    assert band_type_name(mutated) == expected_type
    assert matrix.max() == value


@xfail_until_fixed("D28: dtype='bool' maps to Int8; Byte is the compatible choice")
def test_D28_mutateRaster_bool_gives_byte():
    """mutateRaster(dtype='bool') gives a Byte band holding 0 and 1."""
    uint8_values = np.array([[0, 5], [10, 15]], np.uint8)
    source_raster = create_gdal_raster(uint8_values, gdal.GDT_Byte)

    def is_greater_than_five(matrix):
        return matrix > 5

    mutated = geokit.raster.mutateRaster(source_raster, processor=is_greater_than_five, dtype="bool")

    matrix = geokit.raster.extractMatrix(mutated)
    assert band_type_name(mutated) == "Byte"
    assert set(np.unique(matrix)) == {0, 1}


# ----------------------------------------------------------------------------------------------
# D. Crossing between raster and vector


@xfail_until_fixed("D20: uint32 above 2**31 is written as OFTInteger and clipped")
def test_D20_createVector_uint32_large():
    """A uint32 value above 2**31 survives createVector."""
    uint32_values = np.array([3_000_000_000], np.uint32)
    attributes = pd.DataFrame({"geom": [GEOM_THREE_POINTS[0]], "v": uint32_values})

    point_vector = geokit.vector.createVector(attributes)

    first_feature = point_vector.GetLayer().GetNextFeature()
    assert first_feature.GetField("v") == 3_000_000_000


@pytest.mark.parametrize(
    "column, expected",
    [
        (np.array([5], np.uint64), 5),
        (np.array([1.5], np.float16), 1.5),
        (pd.array([True], dtype="boolean"), 1),
    ],
    ids=["uint64", "float16", "pandas-boolean"],
)
@xfail_until_fixed("D20: uint64, float16 and pandas boolean columns fall back to OFTString")
def test_D20_createVector_column_types(column, expected):
    """uint64, float16 and pandas boolean columns become numeric fields, not strings."""
    attributes = pd.DataFrame({"geom": [GEOM_THREE_POINTS[0]], "v": column})

    point_vector = geokit.vector.createVector(attributes)

    first_feature = point_vector.GetLayer().GetNextFeature()
    field_value = first_feature.GetField("v")
    assert not isinstance(field_value, str)
    assert field_value == expected


@xfail_until_fixed("D21: polygonizeRaster of UInt32 values above 2**31 returns 2147483647")
def test_D21_polygonizeRaster_uint32():
    """UInt32 values above 2**31 survive polygonizeRaster."""
    uint32_values = np.full((4, 4), 3_000_000_000, np.uint32)
    source_raster = create_gdal_raster(uint32_values, gdal.GDT_UInt32)

    polygons = geokit.raster.polygonizeRaster(source_raster)

    assert list(polygons["value"]) == [3_000_000_000]


@xfail_until_fixed("D21: polygonizeRaster silently truncates float rasters and merges polygons")
def test_D21_polygonizeRaster_float_raises():
    """A float raster is rejected by polygonizeRaster instead of being truncated and merged into one polygon."""
    float32_values = np.full((4, 4), 1.7, np.float32)
    float32_values[:, 2:] = 2.4
    float_raster = create_gdal_raster(float32_values, gdal.GDT_Float32)

    with pytest.raises(GeoKitDataTypeError):
        geokit.raster.polygonizeRaster(float_raster)


@xfail_until_fixed("D22: polygonizeMatrix always uses an Int32 raster")
def test_D22_polygonizeMatrix_uint32():
    """Values above 2**31 in a uint32 matrix survive polygonizeMatrix."""
    uint32_matrix = np.full((2, 2), 3_000_000_000, np.uint32)

    polygons = geokit.geom.polygonizeMatrix(uint32_matrix)

    assert list(polygons["value"]) == [3_000_000_000]


# ----------------------------------------------------------------------------------------------
# E. Values handled in NumPy


def create_scaled_int16_raster_with_nodata():
    """Int16 raster [-9999, 100, 200] with noData=-9999 and scale=0.1, so the data pixels read 10.0 and 20.0."""
    int16_values = np.array([[-9999, 100, 200]], np.int16)
    return create_gdal_raster(int16_values, gdal.GDT_Int16, noData=-9999, scale=0.1)


@xfail_until_fixed("D23: extractMatrix(autocorrect=True) compares noData after scaling")
def test_D23_extractMatrix_autocorrect_scaled():
    """extractMatrix(autocorrect=True) masks noData on the raw values before applying the scale."""
    scaled_raster = create_scaled_int16_raster_with_nodata()

    matrix = geokit.raster.extractMatrix(scaled_raster, autocorrect=True)

    assert np.isnan(matrix[0, 0])
    np.testing.assert_allclose(matrix[0, 1:], [10.0, 20.0])


@xfail_until_fixed("D23: extractValues compares noData after scaling")
def test_D23_extractValues_scaled_nodata():
    """The noData mask of extractValues is built on the raw values, before the scale is applied."""
    scaled_raster = create_scaled_int16_raster_with_nodata()

    extracted = geokit.raster.extractValues(scaled_raster, [(50, 50), (150, 50)], pointSRS=3035)

    assert np.isnan(extracted.data[0])
    assert np.isclose(extracted.data[1], 10.0)


@xfail_until_fixed("D24: rasterStats treats scaled noData as data")
def test_D24_rasterStats_scaled():
    """The statistics of rasterStats leave out the noData pixels of a scaled raster."""
    scaled_raster = create_scaled_int16_raster_with_nodata()

    stats = geokit.raster.rasterStats(scaled_raster)

    assert stats.nobs == 2
    assert np.isclose(stats.mean, 15.0)


@pytest.mark.parametrize("noData", [-1, np.nan], ids=["-1", "nan"])
@xfail_until_fixed("D25: indicateValues writes noData into a bool array and indicates every noData pixel")
def test_D25_indicateValues_nodata_not_indicated(noData):
    """NoData pixels are not indicated by indicateValues, for an integer and for a NaN noData."""
    left_half_nan_values = np.full((10, 10), 5.0, np.float32)
    left_half_nan_values[:, :5] = np.nan
    source_raster = geokit.raster.createRaster(
        bounds=(0, 0, 1000, 1000),
        pixelWidth=100,
        pixelHeight=100,
        srs=3035,
        data=left_half_nan_values,
        noData=np.nan,
    )
    region_mask = geokit.RegionMask.fromGeom(geokit.geom.box(0, 0, 1000, 1000, srs=3035), pixelRes=100, srs=3035)

    indicator = region_mask.indicateValues(
        source_raster, value="[0-10]", noData=noData, applyMask=False, multiProcess=False
    )

    if np.isnan(noData):
        is_nodata = np.isnan(indicator)
    else:
        is_nodata = indicator == noData
    assert is_nodata.sum() == 50
    assert (indicator[~is_nodata] == 1).sum() == 50


@xfail_until_fixed("D26: applyMask wraps a NumPy-integer noData into uint8")
def test_D26_applyMask_numpy_int_nodata():
    """A uint8 matrix is promoted by applyMask so that a NumPy-integer noData of -1 is stored, not wrapped to 255."""
    triangle = geokit.geom.polygon([(0, 0), (1000, 0), (0, 1000), (0, 0)], srs=3035)
    triangle_mask = geokit.RegionMask.fromGeom(triangle, pixelRes=100, srs=3035)
    uint8_ones = np.ones(triangle_mask.mask.shape, np.uint8)

    masked = triangle_mask.applyMask(uint8_ones, noData=np.int64(-1))

    assert masked.min() == -1
    assert masked[triangle_mask.mask].min() == 1


# ----------------------------------------------------------------------------------------------
# F. Other


@xfail_until_fixed("M6: createRaster(noData=x) without fill or data fills with 0")
def test_M6_createRaster_nodata_fill():
    """createRaster(noData=x) without data or fill is filled with noData, not with 0."""
    raster = geokit.raster.createRaster(bounds=(0, 0, 200, 100), pixelWidth=100, pixelHeight=100, srs=3035, noData=5)

    matrix = geokit.raster.extractMatrix(raster)
    assert set(np.unique(matrix)) == {5}


@xfail_until_fixed("M8: vectorInfo reports GDAL names of OGR constants")
def test_M8_vectorInfo_field_type_names():
    """The field type names reported by vectorInfo are OGR names, not GDAL names of the OGR constants."""
    attributes = pd.DataFrame({"geom": GEOM_THREE_POINTS, "i": [1, 2, 3], "f": [0.5, 1.5, 2.5], "s": ["a", "b", "c"]})
    point_vector = geokit.vector.createVector(attributes)

    info = geokit.vector.vectorInfo(point_vector)

    assert info.attribute_data_types_str == {"i": "Integer64", "f": "Real", "s": "String"}


@xfail_until_fixed("M4: RegionMask.createRaster() creates Int8 and RegionMask.rasterize() returns int8")
def test_M4_regionmask_byte():
    """RegionMask.createRaster gives a Byte band and RegionMask.rasterize returns uint8."""
    region_mask = geokit.RegionMask.fromGeom(square_polygon(0), pixelRes=100, srs=3035)
    square_vector = geokit.vector.createVector([square_polygon(0)])

    created_raster = region_mask.createRaster()
    rasterized_matrix = region_mask.rasterize(square_vector, value=1)

    assert band_type_name(created_raster) == "Byte"
    assert rasterized_matrix.dtype == np.uint8
    assert rasterized_matrix.max() == 1
