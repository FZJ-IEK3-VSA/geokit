from .helpers import *  # NUMPY_FLOAT_ARRAY, CLC_RASTER_PATH, result
from geokit import raster, geom, util
from osgeo import gdal
import pytest

# gdalType


def test_gdalType():
    assert raster.gdalType(bool) == "GDT_Byte"
    assert raster.gdalType("InT64") == "GDT_Int32"
    assert raster.gdalType("float32") == "GDT_Float32"
    assert raster.gdalType(NUMPY_FLOAT_ARRAY) == "GDT_Float64"
    assert raster.gdalType(NUMPY_FLOAT_ARRAY.dtype) == "GDT_Float64"

# Describe Raster


def test_rasterInfo():
    info = raster.rasterInfo(CLC_RASTER_PATH)

    assert (info.xMin, info.yMin, info.xMax, info.yMax) == (
        4012100.0, 3031800.0, 4094600.0, 3111000.0)  # min/max values
    assert (info.dx == 100 and info.dy == 100)  # dx/dy
    assert (info.bounds == (4012100.0, 3031800.0, 4094600.0, 3111000.0))  # bounds
    assert (info.dtype == gdal.GDT_Byte)  # datatype
    assert (info.srs.IsSame(EPSG3035))  # srs
    assert (info.noData == 0)  # noData
    assert (info.flipY == True)  # flipY

# createRaster


def test_createRaster():
    ######################
    # run and check funcs

    # mem creation
    inputBounds = (10.0, 30.0, 15.0, 40.0)
    inputPixelHeight = 0.02
    inputPixelWidth = 0.01
    inputSRS = 'latlon'
    inputDataType = 'Float32'
    inputNoData = -9999
    inputFillValue = 12.34

    memRas = raster.createRaster(bounds=inputBounds, pixelHeight=inputPixelHeight, pixelWidth=inputPixelWidth, srs=inputSRS,
                                 dtype=inputDataType, noData=inputNoData, fillValue=inputFillValue)

    assert not memRas is None  # creating raster in memory

    mri = raster.rasterInfo(memRas)
    assert mri.bounds == inputBounds  # bounds
    assert mri.dx == inputPixelWidth  # pixel width
    assert mri.dy == inputPixelHeight  # pixel height
    assert mri.noData == inputNoData  # no data
    assert mri.srs.IsSame(EPSG4326)  # srs

    # Disk creation
    data = (np.ones((1000, 500))*np.arange(500)).astype("float32")
    outputFileName = result("util_raster1.tif")

    raster.createRaster(bounds=(10, 30, 15, 40), output=outputFileName, pixelHeight=0.01, pixelWidth=0.01, compress=True,
                        srs=EPSG4326, noDataValue=100, data=data, overwrite=True, meta=dict(bob="bob", TIM="TIMMY"))

    ds = gdal.Open(outputFileName)
    bd = ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
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
    realDiffs = [(-0.18841865745838504, -0.1953854267578663),
                 (0.03190063584128211, -0.019478775579500507),
                 (0.18415527009869948, 0.022563403500242885)]

    # test simple case
    v1 = raster.extractValues(CLC_RASTER_PATH, points)
    for v, real in zip(v1.itertuples(), realValue):
        assert (v.data == real)

    for v, real in zip(v1.itertuples(), realDiffs):
        assert np.isclose(v.xOffset, real[0])
        assert np.isclose(v.yOffset, real[1])

    # test flipped
    v2 = raster.extractValues(CLC_FLIPCHECK_PATH, points)

    for v, real in zip(v2.itertuples(), realValue):
        assert (v.data == real)

    for v, real in zip(v2.itertuples(), realDiffs):
        assert np.isclose(v.xOffset, real[0])
        assert np.isclose(v.yOffset, real[1])

    # test point input
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(4061794.7, 3094718.4)
    pt.AssignSpatialReference(EPSG3035)

    v3 = raster.extractValues(CLC_RASTER_PATH, pt)

    assert (v3.data == 3)
    assert np.isclose(v3.xOffset, 0.44700000000187856)
    assert np.isclose(v3.yOffset, 0.31600000000094042)

    # test window fetch
    real = np.array([[12, 12, 12, 12, 12],
                     [12, 12, 12, 12, 12],
                     [12, 12,  3,  3, 12],
                     [12, 12, 12,  3,  3],
                     [12,  3,  3,  3,  3]])

    v4 = raster.extractValues(CLC_RASTER_PATH, pt, winRange=2)
    assert np.isclose(np.abs(v4.data-real).sum(), 0.0)

# A nicer way to get a single value


def test_interpolateValues():
    point = (4061794.7, 3094718.4)

    v = raster.interpolateValues(
        CLC_RASTER_PATH, point, pointSRS='europe_m', mode="near")
    assert np.isclose(v, 3)

    v = raster.interpolateValues(
        CLC_RASTER_PATH, point, pointSRS='europe_m', mode="linear-spline")
    assert np.isclose(v, 4.572732)  # linear-spline

    v = raster.interpolateValues(
        CLC_RASTER_PATH, point, pointSRS='europe_m', mode="cubic-spline")
    assert np.isclose(v, 2.4197586642)  # cubic-spline

    v = raster.interpolateValues(
        CLC_RASTER_PATH, point, pointSRS='europe_m', mode="average")
    assert np.isclose(v, 9.0612244898)  # average

    v = raster.interpolateValues(
        CLC_RASTER_PATH, point, pointSRS='europe_m', mode="func", func=lambda d, xo, yo: d.max())
    assert np.isclose(v, 12)  # func


def test_gradient():
    # create a sloping surface dataset
    x, y = np.meshgrid(np.abs(np.arange(-100, 100)),
                       np.abs(np.arange(-150, 150)))
    arr = np.ones((300, 200)) + 0.01*y + x*0.03
    slopingDS = raster.createRaster(
        bounds=(0, 0, 200, 300), pixelWidth=1.0, pixelHeight=1.0, data=arr, srs=None)

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
    assert np.isclose(180*aspect[10, 10]/np.pi, -
                      18.4349488229)  # aspect - nw quartile
    assert np.isclose(180*aspect[200, 10]/np.pi,
                      18.4349488229)  # aspect - sw quartile
    assert np.isclose(180*aspect[10, 150]/np.pi, -
                      161.565051177)  # aspect - ne quartile
    assert np.isclose(180*aspect[200, 150]/np.pi,
                      161.565051177)  # aspect - se quartile

    # calculate elevation slope
    output = result("slope_calculation.tif")
    slopeDS = raster.gradient(
        ELEVATION_PATH, factor='latlonToM', output=output, overwrite=True)
    slopeMat = raster.extractMatrix(output)

    assert np.isclose(slopeMat.mean(), 0.0663805622803)  # elevation slope


def test_mutateRaster():
    # Setup
    def isOdd(mat): return np.mod(mat, 2)

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
    raster.mutateRaster(source, processor=isOdd,
                        overwrite=True, output=output2)
    res2 = gdal.Open(output2)

    info2 = raster.rasterInfo(res2)
    assert info2.srs.IsSame(sourceInfo.srs)  # srs
    assert np.isclose(info2.xMin, sourceInfo.xMin)  # bounds
    assert np.isclose(info2.xMax, sourceInfo.xMax)  # bounds
    assert np.isclose(info2.yMin, sourceInfo.yMin)  # bounds
    assert np.isclose(info2.yMax, sourceInfo.yMax)  # bounds

    band2 = res2.GetRasterBand(1)
    arr2 = band2.ReadAsArray()

    assert (arr2.sum() == 156515)  # data

    # Process Raster with a simple processor (flip check)
    output2f = output = result("algorithms_mutateRaster_2f.tif")
    raster.mutateRaster(CLC_FLIPCHECK_PATH, processor=isOdd,
                        overwrite=True, output=output2f)
    res2f = gdal.Open(output2f)

    info2f = raster.rasterInfo(res2f)
    assert info2f.srs.IsSame(sourceInfo.srs)  # srs
    assert np.isclose(info2f.xMin, sourceInfo.xMin)  # bounds
    assert np.isclose(info2f.xMax, sourceInfo.xMax)  # bounds
    assert np.isclose(info2f.yMin, sourceInfo.yMin)  # bounds
    assert np.isclose(info2f.yMax, sourceInfo.yMax)  # bounds

    arr2f = raster.extractMatrix(res2f)

    assert (arr2f.sum() == 156515)  # data

    # Check flipped data
    assert (arr2f == arr2).all()  # flipping error!


def test_loadRaster():
    s3 = util.isRaster(raster.loadRaster(CLC_RASTER_PATH))
    assert s3 == True


def test_createRasterLike():
    source = gdal.Open(CLC_RASTER_PATH)
    sourceInfo = raster.rasterInfo(source)

    data = raster.extractMatrix(source)

    # From raster, no output
    newRaster = raster.createRasterLike(source, data=data*2)
    newdata = raster.extractMatrix(newRaster)
    assert np.isclose(data, newdata/2).all()

    # From raster, with output
    raster.createRasterLike(source, data=data*3,
                            output=result("createRasterLike_A.tif"))
    newdata = raster.extractMatrix(result("createRasterLike_A.tif"))
    assert np.isclose(data, newdata/3).all()

    # From rasterInfo, no output
    newRaster = raster.createRasterLike(sourceInfo, data=data*4)
    newdata = raster.extractMatrix(newRaster)
    assert np.isclose(data, newdata/4).all()


def test_extractMatrix():
    # source, bounds=None, boundsSRS='latlon', maskBand=False, autocorrect=False
    ri = raster.rasterInfo(CLC_RASTER_PATH)

    # Do a full read
    mat1 = raster.extractMatrix(CLC_RASTER_PATH)
    assert np.isclose(10650913, mat1.sum())  # full read values
    assert np.isclose(7.93459728918, mat1.std())  # full read values

    # Read within a boundary
    mat2 = raster.extractMatrix(CLC_RASTER_PATH, bounds=(
        4015000.0, 3032000.0, 4020000.0, 3040000.0), boundsSRS=3035)
    assert np.isclose(mat1[710:790, 29:79], mat2).all()  # extract bounds

    # Read with conversion
    mat3, bounds = raster.extractMatrix(CLC_RASTER_PATH, bounds=(
        6, 50.5, 6.5, 50.75), boundsSRS=4326, returnBounds=True)
    assert bounds == (4037300.0, 3049100.0, 4074100.0, 3078600.0)
    assert np.isclose(mat3.sum(), 2280501)
    assert np.isclose(mat3.std(), 7.40666185608)

    # Test flipped raster
    mat4, bounds = raster.extractMatrix(CLC_FLIPCHECK_PATH, bounds=(
        6, 50.5, 6.5, 50.75), boundsSRS=4326, returnBounds=True)
    assert np.isclose(mat4, mat3).all()  # flipped raster


def test_rasterStats():
    result = raster.rasterStats(CLC_RASTER_PATH, AACHEN_SHAPE_PATH)
    assert np.isclose(result.mean, 15.711518944519621)


def test_indexToCoord():
    rasterSource = gdal.Open(CLC_RASTER_PATH)

    # Test single index
    xy = raster.indexToCoord(xi=10, yi=5, source=rasterSource)
    assert np.isclose(xy, np.array([[4013150.0, 3110450.0]])).all()

    # Test multiple indexes
    xy = raster.indexToCoord(xi=np.array([10, 11, 22, 5]), yi=np.array(
        [5, 5, 3, 5]), source=rasterSource)
    assert np.isclose(xy, np.array([[4013150., 3110450.],
                                    [4013250., 3110450.],
                                    [4014350., 3110650.],
                                    [4012650., 3110450.]])).all()

    # Test multiple indexes, with a flipped source
    rasterSource = gdal.Open(CLC_FLIPCHECK_PATH)
    YS = raster.rasterInfo(CLC_FLIPCHECK_PATH).yWinSize-1

    xy_flipped = raster.indexToCoord(
                    xi=np.array([10, 11, 22, 5]), 
                    yi=np.array([YS-5, YS-5, YS-3, YS-5]), 
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
    r = raster.drawRaster(AACHEN_URBAN_LC, cutline=AACHEN_SHAPE_PATH,
                   resolution=0.001, srs=4326)
    plt.savefig(result("drawRaster-4.png"), dpi=100)

    assert True


def test_polygonizeRaster():
    geoms = raster.polygonizeRaster(AACHEN_URBAN_LC)
    assert np.isclose(geoms.shape[0], 423) # geom count
    is3 = geoms.value == 3
    assert np.isclose(is3.sum(), 2) # value count
    assert np.isclose(geoms.geom[is3].apply(lambda x: x.Area()).sum(),
            120529999.18190208) # geom area


    geoms = raster.polygonizeRaster(AACHEN_URBAN_LC, flat=True)
    assert np.isclose(geoms.shape[0], 3) # geom count
    is3 = geoms.value == 3
    assert np.isclose(is3.sum(), 1) # value count
    assert np.isclose(geoms.geom[is3].apply(lambda x: x.Area()).sum(),
            120529999.18190208) # geom area

def test_warp():
    # Change resolution to disk
    d = raster.warp(CLC_RASTER_PATH, 
                    pixelHeight=200,
                    pixelWidth=200, 
                    output=result("warp1.tif"))
    v1 = raster.extractMatrix(d)
    assert np.isclose(v1.mean(), 16.3141463057 )

    # change resolution to memory
    d = raster.warp(CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200)
    v2 = raster.extractMatrix(d)
    assert np.isclose(v1,v2).all()

    # Do a cutline from disk
    d = raster.warp(CLC_RASTER_PATH, cutline=AACHEN_SHAPE_PATH,
                    output=result("warp3.tif"), noData=99)
    v3 = raster.extractMatrix(d)
    assert np.isclose(v3.mean(), 89.9568135904)
    assert np.isclose(v3[0, 0], 99)

    # Do a cutline from memory
    d = raster.warp(CLC_RASTER_PATH, 
                    cutline=geom.box(*AACHEN_SHAPE_EXTENT_3035, 
                                srs=EPSG3035), 
                    noData=99)
    v4 = raster.extractMatrix(d)
    assert np.isclose(v4[0, 0], 99)
    assert np.isclose(v4.mean(), 76.72702479)

    # Do a flipped-source check
    d = raster.warp(CLC_FLIPCHECK_PATH, 
                    cutline=geom.box(*AACHEN_SHAPE_EXTENT_3035, 
                                srs=EPSG3035), 
                    noData=99)
    v5 = raster.extractMatrix(d)
    assert np.isclose(v4,v5).all()

    d = raster.warp(CLC_FLIPCHECK_PATH, 
                    pixelHeight=200,
                    pixelWidth=200, 
                    output=result("warp6.tif"))
    v6 = raster.extractMatrix(d)
    assert np.isclose(v1,v6).all()
