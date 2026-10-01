"""warp against plain gdal.Warp as the reference (ADR 1, ADR 3).

For every resampling algorithm, source type and operation: under "auto" the values equal the same gdal.Warp
into Float64, so nothing is rounded or clipped; under "preserve_input" they equal the same gdal.Warp into the
source type, which is the GDAL convention; under "smallest" they are the values of "auto" in the narrowest
type that stores them.
"""

import warnings

import numpy as np
import pytest
from osgeo import gdal

import geokit
from geokit import dtypes
from geokit.error import GeoKitDataTypeWarning
from geokit.raster import DETERMINISTIC_WARP_OPTIONS
from test.gdal_builders import band_type_name, create_gdal_raster

RESAMPLING_ALGORITHMS = ["near", "mode", "bilinear", "average", "cubic", "sum"]
SOURCE_TYPES = {
    "Byte": (np.uint8, gdal.GDT_Byte),
    "Int16": (np.int16, gdal.GDT_Int16),
    "Int32": (np.int32, gdal.GDT_Int32),
    "Float32": (np.float32, gdal.GDT_Float32),
}
OPERATIONS = ["downsample", "upsample", "reproject"]


def source_raster(type_name):
    """A 20 x 20 raster of 100 m pixels in EPSG:3035 with values 0 to 199 (plus 0.5 for the float type)."""
    numpy_type, gdal_type = SOURCE_TYPES[type_name]
    random_values = np.random.default_rng(seed=7).integers(0, 200, size=(20, 20))
    pixel_values = random_values.astype(numpy_type)
    if numpy_type is np.float32:
        pixel_values = pixel_values + np.float32(0.5)
    return create_gdal_raster(pixel_values, gdal_type, x_min=4_000_000, y_max=3_000_000)


def warp_arguments(operation):
    if operation == "downsample":
        return dict(pixelWidth=400, pixelHeight=400)
    if operation == "upsample":
        return dict(pixelWidth=50, pixelHeight=50)
    return dict(srs=4326)


def geokit_warp(source, resampleAlg, operation, dtype=None):
    """Call warp without the created-pixel warning, which the reprojection cases issue on purpose."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoKitDataTypeWarning)
        return geokit.raster.warp(source, resampleAlg=resampleAlg, dtype=dtype, **warp_arguments(operation))


def reference_warp(source, geokit_result, resampleAlg, output_type):
    """The same warp done by plain GDAL onto the grid of the GeoKit result, into the given type."""
    result_info = geokit.raster.rasterInfo(geokit_result)
    return gdal.Warp(
        "",
        source,
        format="MEM",
        outputBounds=result_info.bounds,
        width=result_info.xWinSize,
        height=result_info.yWinSize,
        dstSRS=result_info.srs.ExportToWkt(),
        resampleAlg=resampleAlg,
        outputType=output_type,
        multithread=False,
        warpOptions=DETERMINISTIC_WARP_OPTIONS,
    )


def pixel_values(dataset):
    return dataset.GetRasterBand(1).ReadAsArray()


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("type_name", list(SOURCE_TYPES))
@pytest.mark.parametrize("resampleAlg", RESAMPLING_ALGORITHMS)
def test_auto_equals_gdal_into_float64(resampleAlg, type_name, operation):
    """Under auto the values equal the same gdal.Warp into Float64: nothing is rounded or clipped."""
    source = source_raster(type_name)

    result = geokit_warp(source, resampleAlg, operation)
    reference = reference_warp(source, result, resampleAlg, gdal.GDT_Float64)

    np.testing.assert_allclose(pixel_values(result), pixel_values(reference), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("type_name", list(SOURCE_TYPES))
@pytest.mark.parametrize("resampleAlg", RESAMPLING_ALGORITHMS)
def test_preserve_input_equals_gdal_into_the_source_type(resampleAlg, type_name, operation):
    """Under preserve_input the source type is kept and the values are what GDAL writes into that type."""
    source = source_raster(type_name)
    _, source_gdal_type = SOURCE_TYPES[type_name]

    result = geokit_warp(source, resampleAlg, operation, dtype="preserve_input")
    reference = reference_warp(source, result, resampleAlg, source_gdal_type)

    assert band_type_name(result) == type_name
    np.testing.assert_array_equal(pixel_values(result), pixel_values(reference))


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("type_name", list(SOURCE_TYPES))
@pytest.mark.parametrize("resampleAlg", RESAMPLING_ALGORITHMS)
def test_smallest_has_the_values_of_auto_in_the_narrowest_type(resampleAlg, type_name, operation):
    """Under smallest the values are those of auto, stored in the smallest type that holds them exactly."""
    source = source_raster(type_name)

    auto_result = geokit_warp(source, resampleAlg, operation)
    smallest_result = geokit_warp(source, resampleAlg, operation, dtype="smallest")

    auto_values = pixel_values(auto_result)
    np.testing.assert_allclose(pixel_values(smallest_result), auto_values, rtol=1e-6, atol=1e-6)
    assert dtypes.from_band(smallest_result) == dtypes.smallest_dtype_for_array(auto_values)
