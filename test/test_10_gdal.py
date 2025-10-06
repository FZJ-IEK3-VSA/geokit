import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from osgeo.gdal import Driver
from osgeo.ogr import Feature, FieldDefn, Layer, OFTInteger, wkbPolygon

from geokit import raster
from geokit.core.get_test_data import get_test_data
from test.helpers import (
    AACHEN_ELIGIBILITY_RASTER,
    AACHEN_URBAN_LC,
    CLC_RASTER_PATH,
    result,
)


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
    assert arr_inmem.shape == arr_reload.shape, f"Shape mismatch: {arr_inmem.shape} vs {arr_reload.shape}"

    # expected_mean = 16.264478  # Expected mean value based on the warped raster
    assert np.isclose(arr_inmem.mean(), arr_reload.mean(), rtol=1e-3), (
        f"Mean mismatch: got {arr_inmem.mean()} vs {arr_reload.mean()}"
    )

    # Assert arrays match exactly (bitwise)
    assert np.array_equal(arr_inmem, arr_reload), "Warped array differs after reload"
    os.remove(output_path)


# Single-threshold isoband (≥0.5)
def test_ContourGenerateEx_single_isoband():
    contourEdges = [0.5]
    # Open raster file
    ds = gdal.Open(AACHEN_ELIGIBILITY_RASTER, 0)
    assert isinstance(ds, gdal.Dataset)

    band = ds.GetRasterBand(1)
    myarray1 = np.array(band.ReadAsArray())

    # Set spatial reference system
    rasterSRS = osr.SpatialReference()
    _val = rasterSRS.SetFromUserInput(ds.GetProjectionRef())
    assert _val == 0
    # Create layer
    driver: Driver = gdal.GetDriverByName("Memory")
    source: gdal.Dataset = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)
    layer: Layer = source.CreateLayer("", rasterSRS, wkbPolygon)
    field = FieldDefn("DN", OFTInteger)
    layer.CreateField(field)

    # Setup contour function
    args = []
    args.append("ID_FIELD=DN")
    args.append("POLYGONIZE=YES")
    opt = "FIXED_LEVELS="
    for edge in contourEdges:
        opt += str(edge) + ","
    args.append(opt[:-1])

    # Determine contours
    result = gdal.ContourGenerateEx(band, layer, options=args)
    layer.CommitTransaction()
    IDs = []
    geoms = []
    for ftrid in range(layer.GetFeatureCount()):
        ftr: Feature = layer.GetFeature(ftrid)
        geom = ftr.GetGeometryRef()
        value = ftr.GetField(0)
        for gi in range(geom.GetGeometryCount()):
            geoms.append(geom.GetGeometryRef(gi).Clone())
            IDs.append(value)
    countour_data_frame = pd.DataFrame(dict(geom=geoms, ID=IDs))
    total_area = np.sum([countour_data_frame.geom[i].Area() for i in countour_data_frame.index])
    assert countour_data_frame.shape[0] == 114  # geom count
    assert np.isclose(total_area, 0.20382200000004147)
    assert np.isclose(countour_data_frame.ID[59], 1)


# Multi-level isobands using interger step sizes
def test_ContourGenerateEx_multilevel_isobands():
    ###
    contourEdges = [
        1,
        2,
        3,
    ]

    # Open raster file
    ds = gdal.Open(AACHEN_URBAN_LC, 0)
    assert isinstance(ds, gdal.Dataset)

    band = ds.GetRasterBand(1)

    # Set spatial reference system
    rasterSRS = osr.SpatialReference()
    _val = rasterSRS.SetFromUserInput(ds.GetProjectionRef())
    assert _val == 0

    # Create layer
    driver: Driver = gdal.GetDriverByName("Memory")
    source: gdal.Dataset = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)
    layer: Layer = source.CreateLayer("", rasterSRS, wkbPolygon)
    field = FieldDefn("DN", OFTInteger)
    layer.CreateField(field)

    # Setup contour function
    args = []
    args.append("ID_FIELD=DN")
    args.append("POLYGONIZE=YES")
    opt = "FIXED_LEVELS="
    for edge in contourEdges:
        opt += str(edge) + ","
    args.append(opt[:-1])

    # Determine contours
    result = gdal.ContourGenerateEx(band, layer, options=args)
    layer.CommitTransaction()
    IDs = []
    geoms = []
    for ftrid in range(layer.GetFeatureCount()):
        ftr: Feature = layer.GetFeature(ftrid)
        geom = ftr.GetGeometryRef()
        value = ftr.GetField(0)
        for gi in range(geom.GetGeometryCount()):
            geoms.append(geom.GetGeometryRef(gi).Clone())
            IDs.append(value)
    countour_data_frame = pd.DataFrame(dict(geom=geoms, ID=IDs))

    ##

    try:
        assert len(countour_data_frame) == 326
    except AssertionError:
        raise (
            AssertionError(
                " The length of the data frame was " + str(len(countour_data_frame)) + " even though 326 was expected"
            )
        )
    try:
        assert np.isclose(countour_data_frame.iloc[63].geom.Area(), 285000.2699996233)
    except AssertionError:
        raise (
            AssertionError(
                " The area was "
                + str(countour_data_frame.iloc[63].geom.Area())
                + " even though 285000.2699996233 was expected"
            )
        )
    try:
        assert countour_data_frame.iloc[63].ID == 1
    except AssertionError:
        raise (
            AssertionError(
                " The index was "
                + str(countour_data_frame.iloc[63].geom.Area())
                + " even though 285000.2699996233 was expected"
            )
        )


# Multi-level isobands using interger step sizes on a small raster (3x3)
def test_ContourGenerateEx_3x3_raster():
    # Configuration option for the output
    # https://gdal.org/en/stable/programs/gdal_contour.html

    ###
    output_path = get_test_data(r"test_raster_3x3.tif")
    output_path_str = str(output_path)

    contourEdges = [
        1,
        2,
        3,
    ]

    # Open raster file
    ds = gdal.Open(output_path_str, 0)
    assert isinstance(ds, gdal.Dataset)
    band = ds.GetRasterBand(1)

    # Set spatial reference system
    rasterSRS = osr.SpatialReference()
    _val = rasterSRS.SetFromUserInput(ds.GetProjectionRef())
    assert _val == 0

    # Create layer
    driver: Driver = gdal.GetDriverByName("Memory")
    source: gdal.Dataset = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)
    layer: Layer = source.CreateLayer("", rasterSRS, wkbPolygon)
    field = FieldDefn("DN", OFTInteger)
    layer.CreateField(field)

    # Setup contour function
    args = []
    args.append("ID_FIELD=DN")
    args.append("POLYGONIZE=YES")
    opt = "FIXED_LEVELS="
    for edge in contourEdges:
        opt += str(edge) + ","
    args.append(opt[:-1])

    # Determine contours
    result = gdal.ContourGenerateEx(band, layer, options=args)
    layer.CommitTransaction()

    # Convert output to data frame
    IDs = []
    geoms = []
    for ftrid in range(layer.GetFeatureCount()):
        ftr: Feature = layer.GetFeature(ftrid)
        geom = ftr.GetGeometryRef()
        value = ftr.GetField(0)
        for gi in range(geom.GetGeometryCount()):
            geoms.append(geom.GetGeometryRef(gi).Clone())
            IDs.append(value)
    countour_data_frame = pd.DataFrame(dict(geom=geoms, ID=IDs))
    # import geokit as gk

    # path_to_file = pathlib.Path(__file__).parent.joinpath(
    #     "gdal_3_11_3_contours_3x3.png"
    # )
    # gk.drawGeoms(countour_data_frame)
    # plt.savefig(path_to_file)
    try:
        assert len(countour_data_frame) == 3
    except AssertionError:
        raise (
            AssertionError(
                " The length of the data frame was " + str(len(countour_data_frame)) + " even though 3 was expected"
            )
        )
    try:
        assert np.isclose(countour_data_frame.iloc[0].geom.Area(), 4.4999970000059974)
    except AssertionError:
        raise (
            AssertionError(
                " The area was "
                + str(countour_data_frame.iloc[0].geom.Area())
                + " even though 4.4999970000059974 was expected"
            )
        )
    try:
        assert np.isclose(countour_data_frame.iloc[1].geom.Area(), 3)
    except AssertionError:
        raise (
            AssertionError(
                " The area was " + str(countour_data_frame.iloc[1].geom.Area()) + " even though 3 was expected"
            )
        )
