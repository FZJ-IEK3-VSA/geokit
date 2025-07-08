import os
import pathlib

import numpy as np
from osgeo import gdal, osr

from test.helpers import CLC_RASTER_PATH, result


def test_warp_bare_bones():
    current_dir = pathlib.Path(__file__).parent
    root_dir = current_dir.parent
    path_to_input_file = CLC_RASTER_PATH
    source = gdal.Open(path_to_input_file, 0)
    # ds = gdal.Open(path_to_input_file, 0)

    # assert isinstance(ds, gdal.Dataset)
    # projection_string = ds.GetProjectionRef()
    projection_string = 'PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Northing",NORTH],AXIS["Easting",EAST],AUTHORITY["EPSG","3035"]]'
    srs = osr.SpatialReference()
    srs.SetFromUserInput(projection_string)
    output = str(current_dir.joinpath("results", "warp1.tif"))

    # Fix the bounds issue by making them  just a little bit smaller, which should be fixed by gdalwarp

    bounds = (4012000.2, 3031800.2, 4094599.8, 3110999.8)

    # Let gdalwarp do everything...
    opts = gdal.WarpOptions(
        outputType=getattr(gdal, "GDT_Byte"),
        xRes=200,
        yRes=200,
        creationOptions=["COMPRESS=LZW"],
        outputBounds=bounds,
        dstSRS=srs,
        dstNodata=0,
        resampleAlg="bilinear",
        copyMetadata=True,
        targetAlignedPixels=True,
        cutlineDSName=None,
    )
    result = gdal.Warp(output, source, options=opts)
    new_raster = gdal.Open(output)

    new_array = np.array(new_raster.ReadAsArray())

    path_to_comparison_file = root_dir.joinpath(
        "data", "results_for_comparison", "warp1.tif"
    )
    comparison_raster = gdal.Open(str(path_to_comparison_file))
    array_for_comparison = np.array(comparison_raster.ReadAsArray())
    assert new_array.shape == (396, 413)
    assert array_for_comparison.shape == (396, 413)
    assert np.array_equal(new_array, array_for_comparison)


if __name__ == "__main__":
    test_warp_bare_bones()
