import os
import pathlib
import sys

import numpy as np
import pytest
import structlog
from osgeo import gdal

import geokit.raster
from geokit import geom, raster, util
from geokit.error import GeoKitRasterError
from geokit.location import Location, LocationSet
from test.helpers import *  # NUMPY_FLOAT_ARRAY, CLC_RASTER_PATH, result
from test.test_case_creator import (
    TEST_CASE_NAMES,
    golden_raster_path,
    load_test_raster,
)

# gdalType

log: structlog.stdlib.BoundLogger = structlog.get_logger()


# Describe Raster


def test_rasterInfo():
    info = raster.rasterInfo(CLC_RASTER_PATH)

    assert (info.xMin, info.yMin, info.xMax, info.yMax) == (
        4012100.0,
        3031800.0,
        4094600.0,
        3111000.0,
    )  # min/max values
    assert info.dx == 100 and info.dy == 100  # dx/dy
    assert info.bounds == (4012100.0, 3031800.0, 4094600.0, 3111000.0)  # bounds
    assert info.dtype == gdal.GDT_Byte  # datatype
    assert info.srs.IsSame(EPSG3035)  # srs
    assert info.noData == 0  # noData
    assert info.flipY is True  # flipY


# createRaster


def test_create_raster_from_fill_values():
    ######################
    # run and check funcs

    # mem creation
    inputBounds = (10.0, 30.0, 15.0, 40.0)
    inputPixelHeight = 0.02
    inputPixelWidth = 0.01
    inputSRS = "latlon"
    inputDataType = "Float32"
    inputNoData = -9999
    inputFillValue = 12.34

    memRas = raster.createRaster(
        bounds=inputBounds,
        pixelHeight=inputPixelHeight,
        pixelWidth=inputPixelWidth,
        srs=inputSRS,
        dtype=inputDataType,
        noData=inputNoData,
        fill=inputFillValue,
    )

    assert memRas is not None  # creating raster in memory

    mri = raster.rasterInfo(memRas)  # memory raster info
    assert mri.bounds == inputBounds  # bounds
    assert mri.dx == inputPixelWidth  # pixel width
    assert mri.dy == inputPixelHeight  # pixel height
    assert mri.noData == inputNoData  # no data
    assert mri.srs.IsSame(EPSG4326)  # srs
    assert mri.data_type_name_str == inputDataType

    numpy_array_raster = raster.extractMatrix(source=memRas)
    assert np.isclose(numpy_array_raster, inputFillValue).all()
    assert numpy_array_raster.shape == (500, 500)


def test_create_raster_from_numpy_array():
    # Disk creation
    data = (np.ones((1000, 500)) * np.arange(500)).astype("float32")
    outputFileName = result("util_raster1.tif")

    raster.createRaster(
        bounds=(10, 30, 15, 40),
        output=outputFileName,
        pixelHeight=0.01,
        pixelWidth=0.01,
        compress=True,
        srs=EPSG4326,
        noData=100,
        data=data,
        overwrite=True,
        meta=dict(bob="bob", TIM="TIMMY"),
    )

    ds = gdal.Open(outputFileName)
    bd = ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())

    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    assert srs.IsSame(EPSG4326)  # disk raster, srs mismatch

    arr = bd.ReadAsArray()
    assert not (arr.sum() != data.sum())  # disk rsater, data mismatch")

    meta = ds.GetMetadata_Dict()
    assert meta["bob"] == "bob"  # dist raster, data mismatch
    assert meta["TIM"] == "TIMMY"  # dist raster, data mismatch


# Get values directly from a raster


def test_extractValues():
    points = [(6.06590, 50.51939), (6.02141, 50.61491), (6.371634, 50.846025)]
    realValue = [24, 3, 23]
    realDiffs = [
        (-0.18841865745838504, -0.1953854267578663),
        (0.03190063584128211, -0.019478775579500507),
        (0.18415527009869948, 0.022563403500242885),
    ]

    # test simple case
    v1 = raster.extractValues(source=CLC_RASTER_PATH, points=points, pointSRS=4326)
    for v, real in zip(v1.itertuples(), realValue):
        assert v.data == real

    for v, real in zip(v1.itertuples(), realDiffs):
        assert np.isclose(v.xOffset, real[0], rtol=1e-4)
        assert np.isclose(v.yOffset, real[1], rtol=1e-4)

    pass

    # test flipped
    v2 = raster.extractValues(CLC_FLIPCHECK_PATH, points, pointSRS=4326)

    for v, real in zip(v2.itertuples(), realValue):
        assert v.data == real

    for v, real in zip(v2.itertuples(), realDiffs):
        assert np.isclose(v.xOffset, real[0], rtol=1e-4)
        assert np.isclose(v.yOffset, real[1], rtol=1e-4)

    # test point input
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(4061794.7, 3094718.4)
    pt.AssignSpatialReference(EPSG3035)

    pass
    v3 = raster.extractValues(source=CLC_RASTER_PATH, points=pt)

    assert v3.data == 3
    assert np.isclose(v3.xOffset, 0.44700000000187856, rtol=1e-4)
    assert np.isclose(v3.yOffset, 0.31600000000094042, rtol=1e-4)

    # test window fetch
    real = np.array(
        [
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 3, 3, 12],
            [12, 12, 12, 3, 3],
            [12, 3, 3, 3, 3],
        ]
    )

    v4 = raster.extractValues(source=CLC_RASTER_PATH, points=pt, pointSRS=EPSG3035, winRange=2)
    assert np.isclose(np.abs(v4.data - real).sum(), 0.0)

    # now test multiple sources
    sources = [DIVIDED_RASTER_1_PATH, DIVIDED_RASTER_2_PATH, DIVIDED_RASTER_3_PATH]
    pts = [
        geom.point(4040000, 2900000, srs=3035),  # tile 1
        geom.point(4060000, 2980000, srs=3035),  # tile 1
        geom.point(4140000, 2930000, srs=3035),  # tile 2+3
        geom.point(4140000, 2980000, srs=3035),  # tile 2
        geom.point(4040000, 2980000, srs=3035),  # tile 3
        geom.point(4200000, 2980000, srs=3035),  # out of bounds for all tiles
    ]
    # The last point on the PTS list is intentionally out of
    # bounds, so a warning should be raised, but this should
    # not be displayed to the testing user.
    with pytest.warns(UserWarning):
        v4 = raster.extractValues(
            source=sources,
            points=pts,
        )

    assert np.allclose(v4.data.array, np.array([2.0, 24.0, 12.0, 12.0, 23, np.nan]), equal_nan=True)


def test_extractValues_location():
    points = Location(lon=6.06590, lat=50.51939)
    realValue = 24
    realDiffs = (-0.18841865745838504, -0.1953854267578663)

    v1 = raster.extractValues(source=CLC_RASTER_PATH, points=points)

    assert v1.data == realValue

    assert np.isclose(v1.xOffset, realDiffs[0], rtol=1e-4)
    assert np.isclose(v1.yOffset, realDiffs[1], rtol=1e-4)


def test_extractValues_locationSet():
    locations_list = [
        Location(lon=6.06590, lat=50.51939),
        Location(lon=6.02141, lat=50.61491),
        Location(lon=6.371634, lat=50.846025),
    ]
    realValue = [24, 3, 23]
    realDiffs = [
        (-0.18841865745838504, -0.1953854267578663),
        (0.03190063584128211, -0.019478775579500507),
        (0.18415527009869948, 0.022563403500242885),
    ]
    location_set = LocationSet(locations=locations_list)
    v1 = raster.extractValues(source=CLC_RASTER_PATH, points=location_set, pointSRS=4326)

    for v, real in zip(v1.itertuples(), realValue):
        assert v.data == real

    for v, real in zip(v1.itertuples(), realDiffs):
        assert np.isclose(v.xOffset, real[0], rtol=1e-4)
        assert np.isclose(v.yOffset, real[1], rtol=1e-4)


# A nicer way to get a single value


def test_interpolateValues():
    point = (4061794.7, 3094718.4)

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="near")
    assert np.isclose(v, 3)

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="linear-spline")
    assert np.isclose(v, 4.572732)  # linear-spline

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="cubic-spline")
    assert np.isclose(v, 2.4197586642)  # cubic-spline

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="average")
    assert np.isclose(v, 9.0612244898)  # average

    def max_value_interpolator(data, _xo, _yo):
        return data.max()

    v = raster.interpolateValues(
        CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="func", func=max_value_interpolator
    )
    assert np.isclose(v, 12)  # func

    # check also for multi-dimensional window (multiple cells window + multiple points)
    points = [(6.06590, 50.51939), (6.02141, 50.61491), (6.371634, 50.846025)]
    v = raster.interpolateValues(CLC_RASTER_PATH, points, mode="average")
    assert np.isclose(v, np.array([31.83673469, 14.75510204, 7.08163265])).all()


def test_interpolateValues_from_list():
    point = [(4061794.7, 3094718.4)]

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="near")
    assert np.isclose(v, 3)

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="linear-spline")
    assert np.isclose(v, 4.572732)  # linear-spline

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="cubic-spline")
    assert np.isclose(v, 2.4197586642)  # cubic-spline

    v = raster.interpolateValues(CLC_RASTER_PATH, point, pointSRS="europe_laea", mode="average")
    assert np.isclose(v, 9.0612244898)  # average

    def max_value_interpolator(data, _xo, _yo):
        return data.max()

    v = raster.interpolateValues(
        CLC_RASTER_PATH,
        point,
        pointSRS="europe_laea",
        mode="func",
        func=max_value_interpolator,
    )
    assert np.isclose(v, 12)  # func

    # check also for multi-dimensional window (multiple cells window + multiple points)
    points = [(6.06590, 50.51939), (6.02141, 50.61491), (6.371634, 50.846025)]
    v = raster.interpolateValues(CLC_RASTER_PATH, points, mode="average")
    assert np.isclose(v, np.array([31.83673469, 14.75510204, 7.08163265])).all()


def test_extractMatrix():
    # source, bounds=None, boundsSRS='latlon', maskBand=False, autocorrect=False
    ri = raster.rasterInfo(CLC_RASTER_PATH)

    # Do a full read
    mat1 = raster.extractMatrix(CLC_RASTER_PATH)
    assert np.isclose(10650913, mat1.sum())  # full read values
    assert np.isclose(7.93459728918, mat1.std())  # full read values

    # Read within a boundary
    mat2 = raster.extractMatrix(
        CLC_RASTER_PATH,
        bounds=(4015000.0, 3032000.0, 4020000.0, 3040000.0),
        boundsSRS=3035,
    )
    assert np.isclose(mat1[710:790, 29:79], mat2).all()  # extract bounds

    # Read with conversion
    mat3, bounds = raster.extractMatrix(
        CLC_RASTER_PATH, bounds=(6, 50.5, 6.5, 50.75), boundsSRS=4326, returnBounds=True
    )
    assert bounds == (4037300.0, 3049000.0, 4074100.0, 3078700.0)
    assert np.isclose(mat3.sum(), 2294983)
    assert np.isclose(mat3.std(), 7.4207103498004985)

    # Test flipped raster
    mat4, bounds = raster.extractMatrix(
        CLC_FLIPCHECK_PATH,
        bounds=(6, 50.5, 6.5, 50.75),
        boundsSRS=4326,
        returnBounds=True,
    )
    assert np.isclose(mat4, mat3).all()  # flipped raster


def test_gradient():
    # create a sloping surface dataset
    x, y = np.meshgrid(np.abs(np.arange(-100, 100)), np.abs(np.arange(-150, 150)))
    arr = np.ones((300, 200)) + 0.01 * y + x * 0.03
    with pytest.warns(UserWarning, match="No srs given when creating raster."):
        slopingDS = raster.createRaster(bounds=(0, 0, 200, 300), pixelWidth=1.0, pixelHeight=1.0, data=arr, srs=None)

    # do tests
    total = raster.gradient(slopingDS, mode="total", asMatrix=True)
    assert np.isclose(total.mean(), 0.0312809506031)  # total - mean
    assert np.isclose(total[10, 10], 0.0316227766017)  # total - nw quartile
    assert np.isclose(total[200, 10], 0.0316227766017)  # total - sw quartile
    assert np.isclose(total[10, 150], 0.0316227766017)  # total - ne quartile
    assert np.isclose(total[200, 150], 0.0316227766017)  # total - se quartile

    ns = raster.gradient(slopingDS, mode="north-south", asMatrix=True)
    assert np.isclose(ns.mean(), -3.33333333333e-05)  # north-south - mean
    assert np.isclose(ns[10, 10], -0.01)  # north-south - nw quartile
    assert np.isclose(ns[200, 10], 0.01)  # north-south - sw quartile
    assert np.isclose(ns[10, 150], -0.01)  # north-south - ne quartile
    assert np.isclose(ns[200, 150], 0.01)  # north-south - se quartile

    ew = raster.gradient(slopingDS, mode="east-west", asMatrix=True)
    assert np.isclose(ew.mean(), 0.00015)  # east-west - mean
    assert np.isclose(ew[10, 10], 0.03)  # east-west - nw quartile
    assert np.isclose(ew[200, 10], 0.03)  # east-west - sw quartile
    assert np.isclose(ew[10, 150], -0.03)  # east-west - ne quartile
    assert np.isclose(ew[200, 150], -0.03)  # east-west - se quartile

    aspect = raster.gradient(slopingDS, mode="dir", asMatrix=True)
    assert np.isclose(aspect.mean(), 0.0101786336761)  # aspect - mean
    assert np.isclose(180 * aspect[10, 10] / np.pi, -18.4349488229)  # aspect - nw quartile
    assert np.isclose(180 * aspect[200, 10] / np.pi, 18.4349488229)  # aspect - sw quartile
    assert np.isclose(180 * aspect[10, 150] / np.pi, -161.565051177)  # aspect - ne quartile
    assert np.isclose(180 * aspect[200, 150] / np.pi, 161.565051177)  # aspect - se quartile

    # calculate elevation slope
    output = result("slope_calculation.tif")
    slopeDS = raster.gradient(ELEVATION_PATH, factor="latlonToM", output=output, overwrite=True)
    slopeMat = raster.extractMatrix(output)

    assert np.isclose(slopeMat.mean(), 0.0663805622803)  # elevation slope


def test_mutateRaster():
    # Setup
    def isOdd(mat):
        return np.mod(mat, 2)

    source = gdal.Open(CLC_RASTER_PATH)
    sourceInfo = raster.rasterInfo(source)

    # Process Raster with no processor or extent
    # , overwrite=True, output=result("algorithms_mutateRaster_1.tif"))
    res1 = raster.mutateRaster(source, processor=None)

    info1 = raster.rasterInfo(res1)
    assert info1.srs.IsSame(sourceInfo.srs)  # srs
    assert info1.bounds == sourceInfo.bounds  # bounds

    # mutateRaster with a simple processor
    output2 = result("algorithms_mutateRaster_2.tif")
    raster.mutateRaster(source, processor=isOdd, overwrite=True, output=output2)
    res2 = gdal.Open(output2)

    info2 = raster.rasterInfo(res2)
    assert info2.srs.IsSame(sourceInfo.srs)  # srs
    assert np.isclose(info2.xMin, sourceInfo.xMin)  # bounds
    assert np.isclose(info2.xMax, sourceInfo.xMax)  # bounds
    assert np.isclose(info2.yMin, sourceInfo.yMin)  # bounds
    assert np.isclose(info2.yMax, sourceInfo.yMax)  # bounds

    band2 = res2.GetRasterBand(1)
    arr2 = band2.ReadAsArray()

    assert arr2.sum() == 156515  # data

    # Process Raster with a simple processor (flip check)
    output2f = output = result("algorithms_mutateRaster_2f.tif")
    raster.mutateRaster(CLC_FLIPCHECK_PATH, processor=isOdd, overwrite=True, output=output2f)
    res2f = gdal.Open(output2f)

    info2f = raster.rasterInfo(res2f)
    assert info2f.srs.IsSame(sourceInfo.srs)  # srs
    assert np.isclose(info2f.xMin, sourceInfo.xMin)  # bounds
    assert np.isclose(info2f.xMax, sourceInfo.xMax)  # bounds
    assert np.isclose(info2f.yMin, sourceInfo.yMin)  # bounds
    assert np.isclose(info2f.yMax, sourceInfo.yMax)  # bounds

    arr2f = raster.extractMatrix(res2f)

    assert arr2f.sum() == 156515  # data

    # Check flipped data
    assert (arr2f == arr2).all()  # flipping error!


def test_loadRaster():
    s3 = util.isRaster(raster.loadRaster(CLC_RASTER_PATH))
    assert s3 == True


def test_createRasterLike():
    source = gdal.Open(CLC_RASTER_PATH)
    sourceInfo = raster.rasterInfo(sourceDS=source)

    data = raster.extractMatrix(source=source)

    # From raster, no output
    newRaster = raster.createRasterLike(source=source, data=data * 2)
    newdata = raster.extractMatrix(newRaster)
    assert np.isclose(data, newdata / 2).all()

    # From raster, with output
    raster.createRasterLike(source=source, data=data * 3, output=result("createRasterLike_A.tif"))
    newdata = raster.extractMatrix(result("createRasterLike_A.tif"))
    assert np.isclose(data, newdata / 3).all()

    # From rasterInfo, no output
    newRaster = raster.createRasterLike(source=sourceInfo, data=data * 4)
    newdata = raster.extractMatrix(newRaster)

    assert np.isclose(data, newdata / 4).all(), f"data:\n{data}!=newdata\n:{newdata / 4}"


def test_saveRasterAsTif():
    source = gdal.Open(CLC_RASTER_PATH)
    data = raster.extractMatrix(source)

    # Saving from osgeo.gdal.Dataset, with output
    raster.saveRasterAsTif(source, output=result("saveRasterAsTif.tif"))

    newdata = raster.extractMatrix(result("saveRasterAsTif.tif"))
    assert np.isclose(data, newdata).all()


def test_rasterStats():
    result = raster.rasterStats(CLC_RASTER_PATH, AACHEN_SHAPE_PATH)
    assert np.isclose(result.mean, 15.711518944519621)


def test_indexToCoord():
    rasterSource = gdal.Open(CLC_RASTER_PATH)

    # Test single index
    xy = raster.indexToCoord(xi=10, yi=5, source=rasterSource)
    assert np.isclose(xy, np.array([[4013150.0, 3110450.0]])).all()

    # Test multiple indexes
    xy = raster.indexToCoord(xi=np.array([10, 11, 22, 5]), yi=np.array([5, 5, 3, 5]), source=rasterSource)
    assert np.isclose(
        xy,
        np.array(
            [
                [4013150.0, 3110450.0],
                [4013250.0, 3110450.0],
                [4014350.0, 3110650.0],
                [4012650.0, 3110450.0],
            ]
        ),
    ).all()

    # Test multiple indexes, with a flipped source
    rasterSource = gdal.Open(CLC_FLIPCHECK_PATH)
    YS = raster.rasterInfo(CLC_FLIPCHECK_PATH).yWinSize - 1

    xy_flipped = raster.indexToCoord(
        xi=np.array([10, 11, 22, 5]),
        yi=np.array([YS - 5, YS - 5, YS - 3, YS - 5]),
        source=rasterSource,
    )
    assert np.isclose(xy_flipped, xy).all()


def test_drawRaster():
    r = raster.drawRaster(AACHEN_URBAN_LC)
    plt.savefig(result("drawRaster-1.png"), dpi=100)

    # shift
    r = raster.drawRaster(AACHEN_URBAN_LC, rightMargin=0.2)
    plt.savefig(result("drawRaster-2.png"), dpi=100)

    # projection
    r = raster.drawRaster(AACHEN_URBAN_LC, srs=4326)
    plt.savefig(result("drawRaster-3.png"), dpi=100)

    # cutline
    r = raster.drawRaster(AACHEN_URBAN_LC, cutline=AACHEN_SHAPE_PATH, resolution=0.001, srs=4326)
    plt.savefig(result("drawRaster-4.png"), dpi=100)

    assert True


def test_polygonizeRaster():
    geoms = raster.polygonizeRaster(AACHEN_URBAN_LC)
    assert np.isclose(geoms.shape[0], 423)  # geom count
    is3 = geoms.value == 3
    assert np.isclose(is3.sum(), 2)  # value count
    assert np.isclose(geoms.geom[is3].apply(lambda x: x.Area()).sum(), 120529999.18190208)  # geom area

    geoms = raster.polygonizeRaster(AACHEN_URBAN_LC, flat=True)
    assert np.isclose(geoms.shape[0], 3)  # geom count
    is3 = geoms.value == 3
    assert np.isclose(is3.sum(), 1)  # value count
    assert np.isclose(geoms.geom[is3].apply(lambda x: x.Area()).sum(), 120529999.18190208)  # geom area

    geoms = raster.polygonizeRaster(RASTER_GDAL_244, flat=True)
    assert np.equal(geoms.shape[0], 2)  # geom count

    # geom areas
    assert np.isclose(geoms.loc[0, "geom"].Area(), 949049962.3788521)
    assert np.isclose(geoms.loc[1, "geom"].Area(), 5584949959.933687)
    assert np.isclose(geoms.geom.apply(lambda x: x.Area()).sum(), 6533999922.312539)

    # geom validity
    assert geoms.geom.map(lambda g: g.IsValid()).all()


@pytest.mark.filterwarnings("ignore: The current behavior of geokits's contours function is deprecated.")
def test_contours():
    geoms = raster.contours(AACHEN_ELIGIBILITY_RASTER, contourEdges=[0.5])

    ri = raster.rasterInfo(AACHEN_ELIGIBILITY_RASTER)

    total_area = np.sum([geoms.geom[i].Area() for i in geoms.index])

    assert geoms.shape[0] == 114  # geom count
    # assert np.isclose(geoms.geom[59].Area(), 0.022376976699986426) # TODO Why is geom with same area returned at index 61 instead of 59 when utilizing gdal version >= 3.0.0 ?
    assert np.isclose(total_area, 0.20382200000004147)
    assert np.isclose(geoms.ID[59], 1)
    assert geoms.geom[59].GetSpatialReference().IsSame(ri.srs)


def test_warp():
    # Test 1a: Change resolution and save to disk
    d1 = raster.warp(
        CLC_RASTER_PATH,
        resampleAlg="near",
        pixelHeight=200,
        pixelWidth=200,
        output=result("warp1.tif"),
    )

    assert isinstance(d1, str)
    v1 = raster.extractMatrix(d1)

    log.debug("Warped Matrix %s", v1)
    assert np.isclose(v1.mean(), 16.3012082, rtol=1e-4)  # mean value

    # # Test 1b: Load from disk and check
    v2 = raster.extractMatrix(result("warp1.tif"))
    assert np.isclose(v1, v2, atol=0).all()  # values match
    assert np.array_equal(v1, v2)

    # Test 2: change resolution to memory
    d3 = raster.warp(CLC_RASTER_PATH, resampleAlg="near", pixelHeight=200, pixelWidth=200)
    v3 = raster.extractMatrix(d3)
    assert np.isclose(v1, v3, atol=0).all()

    # Test 3: Do a cutline from disk
    d4 = raster.warp(
        CLC_RASTER_PATH,
        resampleAlg="near",
        cutline=AACHEN_SHAPE_PATH,
        output=result("warp3.tif"),
        noData=99,
    )
    v4 = raster.extractMatrix(d4)
    assert np.isclose(v4.mean(), 89.9568135904, rtol=1e-4)
    assert np.isclose(v4[0, 0], 99)

    # Test 4: Do a cutline from memory
    d4 = raster.warp(
        CLC_RASTER_PATH,
        resampleAlg="near",
        cutline=geom.box(*AACHEN_SHAPE_EXTENT_3035, srs=EPSG3035),
        noData=99,
    )
    v4 = raster.extractMatrix(d4)
    assert np.isclose(v4[0, 0], 99, atol=0)
    assert np.isclose(v4.mean(), 76.72702479, atol=0)

    # Test 5a: Do a flipped-source check
    d5 = raster.warp(
        CLC_FLIPCHECK_PATH,
        resampleAlg="near",
        cutline=geom.box(*AACHEN_SHAPE_EXTENT_3035, srs=EPSG3035),
        noData=99,
    )
    v5 = raster.extractMatrix(d5)
    assert np.isclose(v4, v5, atol=0).all()


@pytest.fixture(scope="module")
def simple_4x4_raster():
    """In-memory 4x4 raster with values 1..16, pixel size 100, bounds (0,0,400,400)."""
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    return raster.createRaster(
        bounds=(0, 0, 400, 400),
        data=data,
        pixelWidth=100,
        pixelHeight=100,
        srs=EPSG3035,
    )


@pytest.fixture(scope="module")
def uniform_raster():
    """In-memory 4x4 raster filled with a constant value (7.0), pixel size 100, bounds (0,0,400,400).

    This is based on the fact that any resampling algorithm applied to a uniform
    raster must return that same constant value in every output pixel.
    """
    data = np.full((4, 4), fill_value=7.0, dtype=np.float32)
    return raster.createRaster(
        bounds=(0, 0, 400, 400),
        data=data,
        pixelWidth=100,
        pixelHeight=100,
        srs=EPSG3035,
    )


@pytest.mark.parametrize(
    "resample_alg",
    [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "rms",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
    ],
)
def test_warp_resampling(simple_4x4_raster, resample_alg):
    """Test that each resampling algorithm produces output consistent with its GDAL definition.

    Source: 4x4 raster with values 1..16 (mean=8.5), pixel size 100, warped to pixel size 200.
    Each output pixel aggregates a 2x2 block of input pixels:
      top-left [1,2,5,6], top-right [3,4,7,8], bottom-left [9,10,13,14], bottom-right [11,12,15,16]

    Invariants are derived directly from the GDAL API descriptions for each algorithm.
    """
    warped = raster.warp(simple_4x4_raster, resampleAlg=resample_alg, pixelHeight=200, pixelWidth=200)
    arr = raster.extractMatrix(warped)

    assert arr.shape == (2, 2), f"Expected (2,2) output for {resample_alg}, got {arr.shape}"
    assert np.isfinite(arr).all(), f"{resample_alg}: output contains non-finite values"

    input_mean = 8.5  # mean of 1..16
    input_values = set(range(1, 17))
    output_mean = float(arr.mean())

    def _warp_mean(alg):
        return float(
            raster.extractMatrix(
                raster.warp(simple_4x4_raster, resampleAlg=alg, pixelHeight=200, pixelWidth=200)
            ).mean()
        )

    if resample_alg in ("near", "mode"):
        # Both pick an existing input value per output pixel — no interpolation.
        # near: value of the nearest input pixel centre.
        # mode: value that appears most often among contributing pixels.
        assert set(arr.flatten().astype(int)).issubset(input_values), (
            f"{resample_alg}: output contains values not present in the input set"
        )

    elif resample_alg == "average":
        # Weighted average of all contributing pixels — preserves the mean exactly.
        assert np.isclose(output_mean, input_mean, rtol=1e-3), (
            f"average: output mean {output_mean:.4f} should equal input mean {input_mean}"
        )

    elif resample_alg == "bilinear":
        # Bilinear is a distance-weighted average — also preserves the mean for a uniform grid.
        assert np.isclose(output_mean, input_mean, rtol=0.05), (
            f"bilinear: output mean {output_mean:.4f} should be close to input mean {input_mean}"
        )

    elif resample_alg == "rms":
        # RMS = sqrt(mean(x²)) ≥ arithmetic mean for positive values (power mean inequality).
        assert output_mean >= _warp_mean("average"), f"rms: output mean {output_mean:.4f} should be >= average mean"

    elif resample_alg == "max":
        # Selects the maximum of contributing pixels — output mean must exceed the average mean.
        assert output_mean > _warp_mean("average"), f"max: output mean {output_mean:.4f} should exceed average mean"

    elif resample_alg == "min":
        # Selects the minimum of contributing pixels — output mean must be below the average mean.
        assert output_mean < _warp_mean("average"), f"min: output mean {output_mean:.4f} should be below average mean"

    elif resample_alg == "med":
        # Median lies between the minimum and maximum by definition.
        assert _warp_mean("min") <= output_mean <= _warp_mean("max"), (
            f"med: output mean {output_mean:.4f} should be between min and max means"
        )

    elif resample_alg == "q1":
        # First quartile ≤ median ≤ third quartile by definition.
        assert output_mean <= _warp_mean("med"), f"q1: output mean {output_mean:.4f} should be <= med mean"

    elif resample_alg == "q3":
        # Third quartile ≥ median ≥ first quartile by definition.
        assert output_mean >= _warp_mean("med"), f"q3: output mean {output_mean:.4f} should be >= med mean"

    elif resample_alg == "sum":
        # Weighted sum of contributing pixels.
        # With a clean 2:1 downsampling (4 input pixels → 1 output pixel, full overlap),
        # each output pixel = sum of its 2x2 input block, so the mean scales by factor 4.
        expected = input_mean * 4
        assert np.isclose(output_mean, expected, rtol=1e-3), (
            f"sum: output mean {output_mean:.4f} should be ~{expected} (input mean × 4)"
        )

    # cubic, cubicspline, lanczos: convolution-based interpolation that can produce values
    # outside the input range (ringing). No simple closed-form invariant applies for a
    # 4x4 input, so shape and finiteness checks above are sufficient.


@pytest.mark.parametrize(
    "resample_alg",
    [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "rms",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
    ],
)
def test_warp_resampling_uniform(uniform_raster, resample_alg):
    """Any algorithm applied to a uniform raster must return that same constant value.

    This is the one invariant that holds for every algorithm except "sum".
    """
    fill_value = 7.0
    warped = raster.warp(uniform_raster, resampleAlg=resample_alg, pixelHeight=200, pixelWidth=200)
    arr = raster.extractMatrix(warped)

    assert arr.shape == (2, 2), f"Expected (2,2) output for {resample_alg}, got {arr.shape}"

    if resample_alg == "sum":
        # sum adds all contributing pixels, so a 2:1 downsampling (4 inputs per output)
        # of a uniform raster with value f produces f * 4, not f.
        scale = (200 / 100) ** 2
        assert np.allclose(arr, fill_value * scale, rtol=1e-5), (
            f"sum: uniform input ({fill_value}) should produce {fill_value * scale} after 2x downsampling, got {arr}"
        )
    else:
        assert np.allclose(arr, fill_value, rtol=1e-5), (
            f"{resample_alg}: uniform input ({fill_value}) should produce uniform output, got {arr}"
        )


@pytest.mark.parametrize("source_key", ["simple_4x4", "clc"])
@pytest.mark.parametrize(
    "resample_alg",
    [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "rms",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
    ],
)
def test_warp_memory_vs_disk_equal(source_key, resample_alg, tmp_path, request):
    """The in-memory and on-disk warp paths must produce byte-identical output.

    Regression guard for the path unification (issue #168): both paths share one grid definition
    (geokit.util.canonicalGrid) and the same gdal.WarpOptions, so for identical inputs the only
    difference is the output driver (MEM vs GTiff), which cannot change pixel values. Exact
    equality is therefore the correct assertion.
    """
    source = request.getfixturevalue("simple_4x4_raster") if source_key == "simple_4x4" else CLC_RASTER_PATH
    warp_kwargs = dict(resampleAlg=resample_alg, pixelHeight=200, pixelWidth=200)

    mem = raster.warp(source, **warp_kwargs)
    disk = raster.warp(source, output=str(tmp_path / "warp_mem_vs_disk.tif"), **warp_kwargs)

    assert_raster_equal(mem, disk)


def test_warp_cropToCutline_in_memory(tmp_path):
    """In-memory warp now honours cropToCutline (previously ignored with a warning, issue #168).

    The unified path lets GDAL create the in-memory dataset itself, so the cutline can drive the
    output extent. The cropped output must be smaller than the un-cropped one and must match the
    on-disk cropToCutline result exactly.
    """
    cutline = geom.box(*AACHEN_SHAPE_EXTENT_3035, srs=EPSG3035)  # a sub-box of the CLC extent
    common = dict(resampleAlg="near", pixelHeight=200, pixelWidth=200, cutline=cutline, noData=99)

    cropped = raster.warp(CLC_RASTER_PATH, cropToCutline=True, **common)
    full = raster.warp(CLC_RASTER_PATH, **common)  # cutline applied, but extent NOT cropped

    cropped_shape = raster.extractMatrix(cropped).shape
    full_shape = raster.extractMatrix(full).shape
    assert cropped_shape != full_shape, "cropToCutline should shrink the in-memory output extent"
    assert cropped_shape[0] <= full_shape[0] and cropped_shape[1] <= full_shape[1]

    disk = raster.warp(CLC_RASTER_PATH, output=str(tmp_path / "crop.tif"), cropToCutline=True, **common)
    assert_raster_equal(cropped, disk)


def test_warp_cropToCutline_requires_cutline():
    """CropToCutline without a cutline is a user error, not a silent no-op."""
    with pytest.raises(GeoKitRasterError):
        raster.warp(CLC_RASTER_PATH, resampleAlg="near", pixelHeight=200, pixelWidth=200, cropToCutline=True)


def test_warp_provenance_metadata(tmp_path):
    """Every warp output is stamped with the toolchain versions + resampling algorithm."""
    expected_keys = (
        "GEOKIT_PROVENANCE_GDAL_VERSION",
        "GEOKIT_PROVENANCE_PROJ_VERSION",
        "GEOKIT_PROVENANCE_GEOKIT_VERSION",
        "GEOKIT_PROVENANCE_RESAMPLE_ALG",
    )
    warp_kwargs = dict(resampleAlg="cubic", pixelHeight=200, pixelWidth=200)

    mem = raster.warp(CLC_RASTER_PATH, **warp_kwargs)
    disk_path = str(tmp_path / "prov.tif")
    raster.warp(CLC_RASTER_PATH, output=disk_path, **warp_kwargs)

    for ds in (mem, raster.loadRaster(disk_path)):
        for key in expected_keys:
            assert ds.GetMetadataItem(key), f"missing provenance key {key}"
        assert ds.GetMetadataItem("GEOKIT_PROVENANCE_GDAL_VERSION") == gdal.__version__
        assert ds.GetMetadataItem("GEOKIT_PROVENANCE_RESAMPLE_ALG") == "cubic"


RESAMPLE_ALGS = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "rms",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
]

# The committed goldens are generated on x86_64; Linux and Windows runners reproduce them at full
# accuracy, so the golden canary compares them exactly there. Only platforms whose GDAL build deviates
# from the x86_64 reference get a tolerance -- currently just macOS/arm64, where the interpolating
# kernels differ in the low-order bits (and the Int16 case additionally flips by +-1 when that
# difference straddles a rounding boundary). Keeping the strict comparison on Linux/Windows means the
# canary still catches a real numeric regression there; warp() emits its own macOS/arm64 warning.
_TOLERANT_RESAMPLE_PLATFORMS = {"darwin"}
# atol=1 covers the worst case (one Int16 rounding step); rtol adds headroom proportional to magnitude.
_GOLDEN_RTOL = 1e-3
_GOLDEN_ATOL = 1.0


@pytest.fixture(scope="session")
def resampling_test_rasters():
    """Load each warp test-case raster (integer + float, both with nodata) once per session.

    The rasters are read from their committed files under geokit/data/raster_data/input_data rather
    than rebuilt in memory, so the inputs the tests run against are exactly what is checked into the
    repo (see test.test_case_creator).
    """
    return {name: load_test_raster(name) for name in TEST_CASE_NAMES}


@pytest.mark.parametrize("case", list(TEST_CASE_NAMES))
@pytest.mark.parametrize("resample_alg", RESAMPLE_ALGS)
def test_warp_resampling_golden(resampling_test_rasters, case, resample_alg):
    """Golden-regression canary: warp output must match the committed reference (issue #168).

    Each test-case raster (an integer and a float raster, both carrying a nodata region, see
    test.test_case_creator) is downsampled with every algorithm and compared, via
    assert_raster_equal, against a GeoTIFF checked into
    geokit/data/raster_data/golden_regression_results. Because the warp is deterministic
    (single-threaded, explicit grid), a mismatch means the resampling numerics changed -- almost
    always a GDAL upgrade. The comparison ignores metadata, so the provenance stamp never causes a
    false failure.

    On Linux and Windows the output is compared to the golden exactly (full accuracy), so the canary
    still fires on a genuine numeric regression there. On platforms that deviate from the x86_64
    reference (_TOLERANT_RESAMPLE_PLATFORMS -- currently macOS/arm64) the comparison uses a tolerance,
    because GDAL's floating-point kernels differ in the low-order bits and the Int16 case rounds those
    differences to +-1; without it the canary is red on every Apple-Silicon runner for benign hardware
    noise. warp() emits its own macOS/arm64 warning, so the looser comparison is never silent.

    References are generated automatically on first run (the test is skipped). To regenerate after a
    deliberate toolchain change, run with GEOKIT_REGEN_GOLDEN=1 and commit the updated files.
    """
    source = resampling_test_rasters[case]
    warp_kwargs = dict(resampleAlg=resample_alg, pixelHeight=200, pixelWidth=200)
    golden_path = golden_raster_path(case, resample_alg)

    if os.environ.get("GEOKIT_REGEN_GOLDEN") or not os.path.isfile(golden_path):
        os.makedirs(os.path.dirname(golden_path), exist_ok=True)
        raster.warp(source, output=golden_path, overwrite=True, **warp_kwargs)
        pytest.skip(f"(re)generated golden reference for '{case}/{resample_alg}': {golden_path}")

    produced = raster.warp(source, **warp_kwargs)

    if sys.platform in _TOLERANT_RESAMPLE_PLATFORMS:
        assert_raster_equal(golden_path, produced, rtol=_GOLDEN_RTOL, atol=_GOLDEN_ATOL)
    else:
        assert_raster_equal(golden_path, produced)


@pytest.mark.parametrize("case", list(TEST_CASE_NAMES))
@pytest.mark.parametrize("resample_alg", RESAMPLE_ALGS)
def test_warp_resampling_nodata_respected(resampling_test_rasters, case, resample_alg):
    """Wherever the source is entirely nodata, every algorithm must output nodata and preserve it.

    'average' marks an output pixel nodata only when every contributing source pixel is nodata --
    the minimal 'fully nodata' set. No algorithm can synthesise valid data there, so that set must
    be nodata for every algorithm. (Kernel algorithms like near/bilinear additionally spread nodata
    into partially-covered pixels; that wider behaviour is captured by the golden test above.)
    """
    source = resampling_test_rasters[case]
    nodata = raster.rasterInfo(source).noData
    warp_kwargs = dict(pixelHeight=200, pixelWidth=200)

    out = raster.warp(source, resampleAlg=resample_alg, **warp_kwargs)
    assert raster.rasterInfo(out).noData == nodata, f"{case}/{resample_alg}: output lost the nodata value"

    out_mat = raster.extractMatrix(out)
    core_nodata = raster.extractMatrix(raster.warp(source, resampleAlg="average", **warp_kwargs)) == nodata
    assert core_nodata.any(), "expected some fully-nodata output pixels in the test raster"
    assert (out_mat[core_nodata] == nodata).all(), (
        f"{case}/{resample_alg}: produced valid data over a fully-nodata region"
    )


def test_warpLike():
    _rstr = raster.warpLike(
        dataSource=SINGLE_HILL_PATH,
        contextSource=ELEVATION_PATH,
    )
    assert np.isclose(
        raster.rasterInfo(_rstr).bounds,
        raster.rasterInfo(ELEVATION_PATH).bounds,
        rtol=0.001,
    ).all()
    assert raster.rasterInfo(_rstr).srs.IsSame(raster.rasterInfo(ELEVATION_PATH).srs)

    # must fail with meta and copyMetaData = True
    with pytest.raises(GeoKitRasterError):
        _rstr = raster.warpLike(
            dataSource=SINGLE_HILL_PATH,
            contextSource=ELEVATION_PATH,
            copyMetadata=True,
            meta={"must": "fail"},  # combination with copyMetadata is impossible
        )

    # check forced kwargs
    _rstr = raster.warpLike(
        dataSource=SINGLE_HILL_PATH,
        contextSource=ELEVATION_PATH,
        copyMetadata=False,
        dtype="Float32",
    )
    # The input raster are both in Float32
    assert raster.rasterInfo(_rstr).data_type_name_str == "Float32"


@pytest.fixture()
def sieve_ds() -> np.ndarray:
    data_arr = np.array(
        [
            [0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 1, 1],
        ]
    )

    data_raster = raster.createRaster(
        bounds=(0, 0, 7, 5),
        pixelHeight=1,
        pixelWidth=1,
        srs=3035,
        data=data_arr,
    )

    return data_raster


@pytest.fixture()
def sieve_mask():
    mask_arr = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )

    mask_raster = raster.createRaster(
        bounds=(0, 0, 7, 5),
        pixelHeight=1,
        pixelWidth=1,
        srs=3035,
        data=mask_arr,
        noData=0,
    )
    return mask_raster


@pytest.mark.parametrize(
    "source, threshold, connectedness, mask, expected_output",
    [
        (
            "sieve_ds",
            2,
            4,
            "none",
            np.array(
                [
                    [0, 0, 1, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1],
                ],
            ),
        ),
        (
            "sieve_ds",
            2,
            8,
            "none",
            np.array(
                [
                    [0, 0, 1, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1],
                ],
            ),
        ),
        (
            "sieve_ds",
            2,
            8,
            "sieve_mask",
            np.array(
                [
                    [0, 0, 1, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1],
                ],
            ),
        ),
    ],
)
def test_sieve(source, threshold, connectedness, mask, expected_output, request):
    raster_fixture = request.getfixturevalue(source)
    if mask == "none":
        sieved_raster = raster.sieve(
            source=raster_fixture,
            threshold=threshold,
            connectedness=connectedness,
            mask=mask,
        )
        arr_out = raster.extractMatrix(source=sieved_raster)
    else:
        sieved_raster = raster.sieve(
            source=raster_fixture,
            threshold=threshold,
            connectedness=connectedness,
            mask=request.getfixturevalue(mask),
        )
        arr_out = raster.extractMatrix(source=sieved_raster)

    assert (arr_out == expected_output).all()


def test_rasterCellNo():
    # define some base inputs
    bounds = (-180, -90, 180, 90)
    cellWidth = 0.1
    cellHeight = 0.1

    # define list of input tuples
    points_tups = [(6.2, 50.8), (6.35, 50.55)]
    # generate geoms from points
    points_geoms = [geom.point(tup[0], tup[1], srs=4326) for tup in points_tups]

    # first make sure that safety checks against "wrong" coordinate systems work
    points_geoms_3857 = [geom.transform(p, toSRS=3857) for p in points_geoms]
    with pytest.raises(ValueError):
        # must raise error due to "wrong" points SRS
        raster.rasterCellNo(
            points=points_geoms_3857,  # this is EPSG:3857
            source=AACHEN_ELIGIBILITY_RASTER,  # this is EPSG:4326
        )
    with pytest.raises(ValueError):
        # make sure only EPSG:3857 coordinate system rasters are accepted
        raster.rasterCellNo(
            points=points_geoms,  # this is EPSG:4326
            source=CLC_RASTER_PATH,  # the CLC_RASTER_PATH example is EPSG:3035
        )

    # now test single location first
    cellNo_tup = raster.rasterCellNo(
        points=points_tups[0],
        bounds=bounds,
        cellWidth=cellWidth,
        cellHeight=cellHeight,
    )
    assert cellNo_tup == (1861, 392)  # must be tuple type return and value match

    # then test multiple
    cellNos_tup = raster.rasterCellNo(
        points=points_tups,
        bounds=bounds,
        cellWidth=cellWidth,
        cellHeight=cellHeight,
    )
    assert cellNos_tup == [(1861, 392), (1863, 394)]  # list of tuples with values

    # test with geoms generated based on tuples
    cellNos_geoms = raster.rasterCellNo(
        points=points_geoms,
        bounds=bounds,
        cellWidth=cellWidth,
        cellHeight=cellHeight,
    )
    assert cellNos_geoms == cellNos_tup  # must be the same as tuple inputs

    # test again with source raster input to determine cells
    cellNos_geoms_rstr = raster.rasterCellNo(
        points=points_geoms,
        source=AACHEN_ELIGIBILITY_RASTER,  # use the Aachen eligibility raster as epsg:4326 example
    )
    assert cellNos_geoms_rstr == [(225, 151), (375, 401)]


def test_warp_meta_argument_in_memory():
    rInfo = raster.rasterInfo(SINGLE_HILL_PATH)

    output_raster = raster.warp(
        source=ELEVATION_PATH,
        meta={"AREA_OR_POINT": "Area"},
        bounds=rInfo.bounds,
        pixelWidth=rInfo.pixelWidth,
        pixelHeight=rInfo.pixelHeight,
        srs=rInfo.srs,
    )
    assert raster.rasterInfo(output_raster).meta["AREA_OR_POINT"] == "Area"


def test_warp_meta_argument_hard_drive():
    output_path = pathlib.Path(__file__).parent.joinpath("results", "warped_raster_with_meta_data.tif")
    raster_info_input = raster.rasterInfo(SINGLE_HILL_PATH)

    raster.warp(
        source=ELEVATION_PATH,
        meta={"AREA_OR_POINT": "Area"},
        output=output_path,
        bounds=raster_info_input.bounds,
        pixelWidth=raster_info_input.pixelWidth,
        pixelHeight=raster_info_input.pixelHeight,
        srs=raster_info_input.srs,
        overwrite=True,
    )
    raster_info_output = raster.rasterInfo(output_path)

    assert raster_info_output.meta["AREA_OR_POINT"] == "Area"
    pathlib.Path.unlink(output_path)


# def test_():
#     # generate the same raster twice, once with and once without srs
#     arr = np.array([[50, 100, 150], [200, 250, 255]])
#     rstr_withsrs = geokit.raster.createRaster(
#         data=arr,
#         bounds=(0, 0, 3, 2),
#         pixelWidth=1,
#         pixelHeight=1,
#         srs=4326,
#     )

#     rstr_nosrs = gk.raster.createRaster(
#         data=arr,
#         bounds=(0, 0, 3, 2),
#         pixelWidth=1,
#         pixelHeight=1,
#     )

#     # then warp to another noData value, once for the raster with and once without srs
#     rstr_wrpdwithsrs = gk.raster.warp(
#         source=rstr_withsrs,
#         bounds=(0, 0, 3, 2),
#         pixelWidth=1,
#         pixelHeight=1,
#         noData=np.nan,
#     )
#     print("NoData of new raster:", gk.raster.rasterInfo(rstr_wrpdwithsrs).noData)

#     rstr_wrpdnosrs = gk.raster.warp(  # this one fails!
#         source=rstr_nosrs,
#         bounds=(0, 0, 3, 2),
#         pixelWidth=1,
#         pixelHeight=1,
#         noData=np.nan,
#     )
#     print("NoData of new raster:", gk.raster.rasterInfo(rstr_wrpdnosrs).noData)


# def test_warp():
# import numpy as np
# import geokit as gk

# raster_matrix_2x3 = np.array(
#     [
#         [5, 255, 0],
#         [2, 3, 7],
#     ],
#     dtype=np.uint8,
# )

# raster = gk.raster.createRaster(
#     bounds=[0, 0, 3, 2],
#     pixelWidth=1,
#     pixelHeight=1,
#     data=raster_matrix_2x3,
#     srs=4326,
#     noData=255,
#     # output=intermediate_raster_tif_str,
# )

# raster_warped = geokit.raster.warp(source=raster, pixelWidth=1, pixelHeight=1, noData=255, fill=-9999)
# raster_warped_matrix = geokit.raster.extractMatrix(source=raster_warped)
# print(raster_warped_matrix)
# pass


if __name__ == "__main__":
    test_createRasterLike()
