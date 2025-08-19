import os
import pathlib

import numpy as np
from osgeo import gdal, osr
from osgeo.gdal import Driver
from test.helpers import CLC_RASTER_PATH, result,AACHEN_ELIGIBILITY_RASTER
from osgeo.ogr import FieldDefn,Layer,wkbPolygon,OFTInteger,Feature
import pandas as pd

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

    # expected_mean = 16.264478  # Expected mean value based on the warped raster
    assert np.isclose(
        arr_inmem.mean(), arr_reload.mean(), rtol=1e-3
    ), f"Mean mismatch: got {arr_inmem.mean()} vs {arr_reload.mean()}"

    # Assert arrays match exactly (bitwise)
    assert np.array_equal(arr_inmem, arr_reload), "Warped array differs after reload"

    os.remove(output_path)

from geokit import  raster

def test_contours():
    geoms = raster.contours(AACHEN_ELIGIBILITY_RASTER, contourEdges=[0.5])

    # ri = raster.rasterInfo(AACHEN_ELIGIBILITY_RASTER)

    total_area = np.sum([geoms.geom[i].Area() for i in geoms.index])

    assert geoms.shape[0] == 114  # geom count
    # assert np.isclose(geoms.geom[59].Area(), 0.022376976699986426) # TODO Why is geom with same area returned at index 61 instead of 59 when utilizing gdal version >= 3.0.0 ?
    assert np.isclose(total_area, 0.20382200000004147)
    assert np.isclose(geoms.ID[59], 1)
    # assert geoms.geom[59].GetSpatialReference().IsSame(ri.srs)


def test_contours_pure_gdal():
    contourEdges = [0.5]
    ds = gdal.Open(AACHEN_ELIGIBILITY_RASTER, 0)
    assert isinstance(ds, gdal.Dataset)
    band = raster.GetRasterBand(1)
    ds.GetProjectionRef()
    rasterSRS = osr.SpatialReference()
    _val = rasterSRS.SetFromUserInput(source)
    assert _val == 0
    
    driver: Driver = gdal.GetDriverByName("Memory")
    source: gdal.Dataset = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)
    layer: Layer = source.CreateLayer(
        "", rasterSRS, wkbPolygon 
    )
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

    result = gdal.ContourGenerateEx(band, layer, options=args)
    if not result == gdal.CE_None:
        raise GeoKitRasterError("Failed to compute raster contours")
    layer.CommitTransaction()

    IDs = []
    geoms = []
    iterator_n = 0
    for ftrid in range(layer.GetFeatureCount()):
        ftr: ogr.Feature = layer.GetFeature(ftrid)
        geom = ftr.GetGeometryRef()
        value = ftr.GetField(0)
        print(iterator_n)
        iterator_n = iterator_n+1
        if unpack:
            for gi in range(geom.GetGeometryCount()):
                geoms.append(geom.GetGeometryRef(gi).Clone())
                IDs.append(value)
        else:
            geoms.append(geom.Clone())
            IDs.append(value)

    countour_data_frame = pd.DataFrame(dict(geom=geoms, ID=IDs))