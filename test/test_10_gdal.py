import os
import pathlib

import numpy as np
from osgeo import gdal, osr

from test.helpers import CLC_RASTER_PATH, result


def test_gdal_warp_basic():

    # Perform warp using gdal.Warp
    ds = gdal.Warp(
        result("warp_basic.tif"), CLC_RASTER_PATH, xRes=200, yRes=200, format="GTiff"
    )

    assert ds is not None, "Warping failed"
    ds = None  # Close dataset

    # Read the warped file
    ds = gdal.Open(result("warp_basic.tif"))
    assert ds is not None, "Failed to open output raster"

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    assert isinstance(arr, np.ndarray), "Output is not a NumPy array"

    expected_mean = 16.26
    assert np.isclose(
        arr.mean(), expected_mean, rtol=1e-3
    ), f"Mean mismatch: got {arr.mean()}"

    expected_shape = (int(ds.RasterYSize), int(ds.RasterXSize))
    assert (
        arr.shape == expected_shape
    ), f"Shape mismatch: got {arr.shape}, expected {expected_shape}"

    ds = None
    os.remove(result("warp_basic.tif"))
