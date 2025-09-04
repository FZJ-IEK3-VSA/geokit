import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import osr

import geokit as gk


def create_test_raster_3x4():
    raster_matrix = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    import pathlib

    output_path = pathlib.Path(__file__).parent.joinpath(r"test_raster_3x3_v2.tif")
    ras = gk.raster.createRaster(
        # bounds=[5, 48, 8, 51],
        bounds=[3, 9, 6, 12],
        pixelWidth=1,
        pixelHeight=1,
        data=raster_matrix,
        srs=srs,
        output=str(output_path),
    )


def create_test_raster_3x4():
    raster_matrix = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    import pathlib

    output_path = pathlib.Path(__file__).parent.joinpath(r"test_raster_3x3.tif")
    ras = gk.raster.createRaster(
        # bounds=[5, 48, 8, 51],
        bounds=[3, 9, 6, 13],
        pixelWidth=1,
        pixelHeight=1,
        data=raster_matrix,
        srs=srs,
        output=str(output_path),
    )


def plot_3x3():
    output_path = pathlib.Path(__file__).parent.joinpath(r"test_raster_3x3.tif")
    output_path_str = str(output_path)
    raster_plot = gk.drawRaster(
        source=output_path_str,
    )
    raster_plot
    plt.show()
