import os
import pathlib

import numpy as np
from osgeo import gdal, osr

from test.helpers import CLC_RASTER_PATH, result


def test_gdal_warp_basic():

    output_path = result("warp_basic.tif")

    # Perform warp using gdal.Warp (write to disk)
    warped_ds = gdal.Warp(
        output_path,
        CLC_RASTER_PATH,
        xRes=200,
        yRes=200,
        resampleAlg="nearest",
        format="GTiff",
    )

    assert warped_ds is not None, "Warping failed"

    # Read array directly from the warped dataset (in-memory reference)
    warped_band = warped_ds.GetRasterBand(1)
    arr_inmem = warped_band.ReadAsArray()
    warped_ds = None  # Close dataset

    assert isinstance(arr_inmem, np.ndarray), "Warped output is not a NumPy array"

    # Open the same file and read again (reload from disk)
    ds_reload = gdal.Open(output_path)
    assert ds_reload is not None, "Failed to reopen output raster"

    band_reload = ds_reload.GetRasterBand(1)
    arr_reload = band_reload.ReadAsArray()
    ds_reload = None  # Close dataset after reading

    # Assert shape equality
    assert (
        arr_inmem.shape == arr_reload.shape
    ), f"Shape mismatch: {arr_inmem.shape} vs {arr_reload.shape}"

    expected_mean = 16.264478  # Expected mean value based on the warped raster
    assert np.isclose(
        arr_inmem.mean(), expected_mean, rtol=1e-3
    ), f"Mean mismatch: got {arr_inmem.mean()}"

    # Assert arrays match exactly (bitwise)
    assert np.array_equal(arr_inmem, arr_reload), "Warped array differs after reload"

    os.remove(output_path)
