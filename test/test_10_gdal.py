import os
import pathlib

import numpy as np
from osgeo import gdal

from test.helpers import CLC_RASTER_PATH, result


def test_warp_bare_bones():
    current_dir = pathlib.Path(__file__).parent
    root_dir = current_dir.parent
    path_to_input_file = CLC_RASTER_PATH
    pixelHeight = 200
    pixelWidth = 200
    output = str(current_dir.joinpath("results", "warp1.tif"))

    # # Check some for bad input configurations
    # if not srs is None:
    #     if (pixelHeight is None or pixelWidth is None):
    #         raise GeoKitRasterError("When warping between srs's and writing to a file, pixelWidth and pixelHeight must be given")

    # Arange inputs
    co = ["COMPRESS=LZW"]
    copyMeta = True
    aligned = True

    # Fix the bounds issue by making them  just a little bit smaller, which should be fixed by gdalwarp

    bounds = (4012000.2, 3031800.2, 4094599.8, 3110999.8)

    # Let gdalwarp do everything...
    opts = gdal.WarpOptions(
        outputType=getattr(gdal, "GDT_Byte"),
        xRes=pixelWidth,
        yRes=pixelHeight,
        creationOptions=co,
        outputBounds=bounds,
        dstSRS=None,
        dstNodata=None,
        resampleAlg="bilinear",
        copyMetadata=copyMeta,
        targetAlignedPixels=aligned,
        cutlineDSName=None,
    )
    result = gdal.Warp(output, path_to_input_file, options=opts)

    new_raster = gdal.Open(output)
    path_to_comparison_file = root_dir.joinpath(
        "data", "results_for_comparison", "warp1.tif"
    )
    comparison_raster = gdal.Open(str(path_to_comparison_file))
    new_array = np.array(new_raster.ReadAsArray())
    array_for_comparison = np.array(comparison_raster.ReadAsArray())
    assert new_array.shape == (396, 413)
    assert array_for_comparison.shape == (396, 413)
    assert np.array_equal(new_array, array_for_comparison)


if __name__ == "__main__":
    test_warp_bare_bones()
