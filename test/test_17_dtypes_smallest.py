"""Unit tests of ``geokit.dtypes.smallest``: the lossless shrink of a finished output (the "smallest" mode of ADR 1).

The rasters used here are built with plain GDAL; no GeoKit function is called.
"""

import numpy as np
import pytest
from osgeo import gdal

from geokit import dtypes
from test.gdal_builders import create_gdal_raster


def array_id(value):
    """An id that shows the dtype of an array, because several rows differ only in it."""
    if isinstance(value, np.ndarray):
        return f"{value.dtype}{value.tolist()}"
    return None


@pytest.mark.parametrize(
    "values, no_data, expected",
    [
        (np.arange(6, dtype=np.int64), None, "uint8"),
        (np.ones((3, 3), np.uint8), -1, "int8"),
        (np.array([[0.0, 1.0], [2.0, 3.0]]), None, "uint8"),
        (np.array([0.5, 1.5]), None, "float32"),
        (np.array([0.1, 0.2]), None, "float64"),
        (np.array([1.0, 2.0, np.nan]), np.nan, "float64"),
        (np.array([1.0, 2.0, np.nan], np.float32), np.nan, "float32"),
        (np.array([1.0, np.nan]), None, "float32"),
        (np.array([-5, 5], np.int16), None, "int8"),
        (np.array([300, 400], np.int32), None, "int16"),
        (np.array([40000], np.int64), None, "uint16"),
        (np.array([3200.0]), None, "int16"),
        (np.array([123456789.0, 0.0]), None, "int32"),
        (np.array([0.5]), -9999, "float32"),
        (np.array([0.5]), 123456789, "float64"),
        (np.array([True, False]), None, "uint8"),
        (np.array([], dtype=np.float64), None, "uint8"),
        (np.array([-9999, 100, 200], np.int16), -9999, "int16"),
        (np.array([2**40], np.int64), None, "int64"),
        (np.array([2**24 + 1.0]), None, "int32"),
        (np.array([np.inf, 1.0]), None, "float32"),
    ],
    ids=array_id,
)
def test_smallest_dtype_for_array(values, no_data, expected):
    """The smallest type holds every value and the noData value exactly; NaN keeps a float type."""
    assert dtypes.smallest_dtype_for_array(values, no_data) == np.dtype(expected)


def test_shrink_dataset_translates_to_the_smallest_type_and_keeps_the_values():
    """A Float64 raster holding whole numbers becomes Int16 with the same values, noData, scale and offset."""
    pixel_values = np.array([[0.0, 1.0, 250.0], [3.0, -9999.0, 2.0]])
    source = create_gdal_raster(pixel_values, gdal.GDT_Float64, noData=-9999, scale=0.1)
    source.GetRasterBand(1).SetOffset(5.0)

    shrunk = dtypes.shrink_dataset(source)

    shrunk_band = shrunk.GetRasterBand(1)
    assert dtypes.from_band(shrunk) == np.dtype("int16")
    np.testing.assert_array_equal(shrunk_band.ReadAsArray(), pixel_values.astype(np.int16))
    assert shrunk_band.GetNoDataValue() == -9999
    assert shrunk_band.GetScale() == 0.1
    assert shrunk_band.GetOffset() == 5.0


def test_shrink_dataset_returns_the_same_dataset_when_nothing_changes():
    """A raster that already has the smallest type is returned as is, without a copy."""
    source = create_gdal_raster(np.array([[0, 1], [2, 3]], np.uint8), gdal.GDT_Byte)

    assert dtypes.shrink_dataset(source) is source


def test_smallest_dtype_for_dataset_reads_block_by_block():
    """Reading one row at a time gives the same type as reading the whole array."""
    pixel_values = np.arange(20, dtype=np.int32).reshape(5, 4) * 1000
    source = create_gdal_raster(pixel_values, gdal.GDT_Int32)

    block_wise = dtypes.smallest_dtype_for_dataset(source, rows_per_block=1)

    assert block_wise == dtypes.smallest_dtype_for_array(pixel_values)
    assert block_wise == np.dtype("int16")
