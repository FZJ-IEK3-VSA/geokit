import os

import numpy as np
import pandas as pd
from osgeo import gdal, osr
from osgeo.gdal import Driver
from osgeo.ogr import Feature, FieldDefn, Layer, OFTInteger, wkbPolygon

from test.helpers import (
    AACHEN_ELIGIBILITY_RASTER,
    AACHEN_URBAN_LC,
    CLC_RASTER_PATH,
    result,
)

# def test_gdal_warp_basic():
#     output_path = result("warp_basic.tif")

#     # Perform warp using gdal.Warp (write to disk)
#     warped_ds = gdal.Warp(
#         output_path,
#         CLC_RASTER_PATH,
#         xRes=200,
#         yRes=200,
#         resampleAlg="nearest",
#         format="GTiff",
#     )

#     assert warped_ds is not None, "Warping failed"

#     # Read array directly from the warped dataset (in-memory reference)
#     warped_band = warped_ds.GetRasterBand(1)
#     arr_inmem = warped_band.ReadAsArray()
#     warped_ds = None  # Close dataset

#     assert isinstance(arr_inmem, np.ndarray), "Warped output is not a NumPy array"

#     # Open the same file and read again (reload from disk)
#     ds_reload = gdal.Open(output_path)
#     assert ds_reload is not None, "Failed to reopen output raster"

#     band_reload = ds_reload.GetRasterBand(1)
#     arr_reload = band_reload.ReadAsArray()
#     ds_reload = None  # Close dataset after reading

#     # Assert shape equality
#     assert arr_inmem.shape == arr_reload.shape, (
#         f"Shape mismatch: {arr_inmem.shape} vs {arr_reload.shape}"
#     )

#     # expected_mean = 16.264478  # Expected mean value based on the warped raster
#     assert np.isclose(arr_inmem.mean(), arr_reload.mean(), rtol=1e-3), (
#         f"Mean mismatch: got {arr_inmem.mean()} vs {arr_reload.mean()}"
#     )

#     # Assert arrays match exactly (bitwise)
#     assert np.array_equal(arr_inmem, arr_reload), "Warped array differs after reload"

#     os.remove(output_path)


# from geokit import raster


# def test_contours():
#     geoms = raster.contours(AACHEN_ELIGIBILITY_RASTER, contourEdges=[0.5])

#     # ri = raster.rasterInfo(AACHEN_ELIGIBILITY_RASTER)

#     total_area = np.sum([geoms.geom[i].Area() for i in geoms.index])

#     assert geoms.shape[0] == 114  # geom count
#     # assert np.isclose(geoms.geom[59].Area(), 0.022376976699986426) # TODO Why is geom with same area returned at index 61 instead of 59 when utilizing gdal version >= 3.0.0 ?
#     assert np.isclose(total_area, 0.20382200000004147)
#     assert np.isclose(geoms.ID[59], 1)
#     # assert geoms.geom[59].GetSpatialReference().IsSame(ri.srs)


# def test_contours_pure_gdal():
#     contourEdges = [0.5]
#     # Open raster file
#     ds = gdal.Open(AACHEN_ELIGIBILITY_RASTER, 0)
#     assert isinstance(ds, gdal.Dataset)

#     pass
#     band = ds.GetRasterBand(1)
#     myarray1 = np.array(band.ReadAsArray())
#     pass
#     # gk.drawRaster(band)
#     pass
#     # Set spatial reference system
#     rasterSRS = osr.SpatialReference()
#     _val = rasterSRS.SetFromUserInput(ds.GetProjectionRef())
#     assert _val == 0

#     # Create layer
#     driver: Driver = gdal.GetDriverByName("Memory")
#     source: gdal.Dataset = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)
#     layer: Layer = source.CreateLayer("", rasterSRS, wkbPolygon)
#     field = FieldDefn("DN", OFTInteger)
#     layer.CreateField(field)

#     # Setup contour function
#     args = []
#     args.append("ID_FIELD=DN")
#     args.append("POLYGONIZE=YES")

#     opt = "FIXED_LEVELS="
#     for edge in contourEdges:
#         opt += str(edge) + ","
#     args.append(opt[:-1])

#     # Determine contours
#     result = gdal.ContourGenerateEx(band, layer, options=args)
#     layer.CommitTransaction()

#     IDs = []
#     geoms = []
#     for ftrid in range(layer.GetFeatureCount()):
#         ftr: Feature = layer.GetFeature(ftrid)
#         geom = ftr.GetGeometryRef()
#         value = ftr.GetField(0)
#         for gi in range(geom.GetGeometryCount()):
#             geoms.append(geom.GetGeometryRef(gi).Clone())
#             IDs.append(value)

#     countour_data_frame = pd.DataFrame(dict(geom=geoms, ID=IDs))
#     total_area = np.sum(
#         [countour_data_frame.geom[i].Area() for i in countour_data_frame.index]
#     )

#     assert countour_data_frame.shape[0] == 114  # geom count
#     assert np.isclose(total_area, 0.20382200000004147)
#     assert np.isclose(countour_data_frame.ID[59], 1)


def test_Extent_contoursFromRaster():
    ###
    contourEdges = [1, 2, 3]
    # Open raster file
    ds = gdal.Open(AACHEN_URBAN_LC, 0)
    assert isinstance(ds, gdal.Dataset)

    pass
    band = ds.GetRasterBand(1)
    myarray1 = np.array(band.ReadAsArray())
    unique_arary_values = np.unique(myarray1)
    pass
    # gk.drawRaster(band)
    pass
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

    # assert countour_data_frame.iloc[0].geom.GetSpatialReference().IsSame(ext.srs)
    try:
        assert len(countour_data_frame) == 324
    except AssertionError:
        raise (
            AssertionError(
                " The length of the data frame was "
                + str(len(countour_data_frame))
                + " even though 324 was expected"
            )
        )
    try:
        assert np.isclose(countour_data_frame.iloc[63].geom.Area(), 285000.2699996233)
    except AssertionError:
        raise (
            AssertionError(
                " The area was"
                + str(countour_data_frame.iloc[63].geom.Area())
                + " even though 285000.2699996233 was expected"
            )
        )
    try:
        assert countour_data_frame.iloc[63].ID == 1
    except AssertionError:
        raise (
            AssertionError(
                " The index was"
                + str(countour_data_frame.iloc[63].geom.Area())
                + " even though 285000.2699996233 was expected"
            )
        )

    # assert len(countour_data_frame) == 95
    # assert np.isclose(
    #     countour_data_frame.iloc[63].geom.Area(), 0.08834775465377398
    # )  # index of geom changed from 61 to 63 with GDAL >= 3.0.0
    # assert countour_data_frame.iloc[63].ID == 1


if __name__ == "__main__":
    test_Extent_contoursFromRaster()
#     # test_contours()
#     test_contours_pure_gdal()
