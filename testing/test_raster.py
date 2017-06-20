from helpers import *
from geokit.raster import *

## gdalType
def test_gdalType():
    if( gdalType(bool) != "GDT_Byte" ): error("gdal type")
    if( gdalType("InT64") != "GDT_Int32" ): error("gdal type")
    if( gdalType("float32") != "GDT_Float32" ): error("gdal type")
    if( gdalType(NUMPY_FLOAT_ARRAY) != "GDT_Float64" ): error("gdal type")
    if( gdalType(NUMPY_FLOAT_ARRAY.dtype) != "GDT_Float64" ): error("gdal type")

## Describe Raster
def test_rasterInfo():
    info = rasterInfo(CLC_RASTER_PATH)

    if(not (info.xMin, info.yMin, info.xMax, info.yMax) == (4012100.0, 3031800.0, 4094600.0, 3111000.0)):
        error("rasterInfo - min/max values")
    if not (info.dx == 100 and info.dy == 100) : error("decribeRaster - dx/dy")
    if not (info.bounds == (4012100.0, 3031800.0, 4094600.0, 3111000.0)) : error("decribeRaster - bounds")
    if not (info.dtype == gdal.GDT_Byte) : error("decribeRaster - datatype")
    if not (info.srs.IsSame(EPSG3035)) : error("decribeRaster - srs")
    if not (info.noData == 0) : error("decribeRaster - noData")
    if not (info.flipY == True) : error("decribeRaster - flipY")

## createRaster
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

    memRas = createRaster( bounds=inputBounds, pixelHeight=inputPixelHeight, pixelWidth=inputPixelWidth, srs=inputSRS, 
                           dtype=inputDataType, noDataValue=inputNoData, fillValue=inputFillValue)

    if(memRas is None): error("creating raster in memory")
    mri = rasterInfo(memRas)
    if not mri.bounds==inputBounds: error("raster bounds")
    if not mri.dx==inputPixelWidth: error("raster pixel width")
    if not mri.dy==inputPixelHeight: error("raster pixel height")
    if not mri.noData==inputNoData: error("raster no data")
    if not mri.srs.IsSame(EPSG4326): error("raster srs")

    # Disk creation
    data = (np.ones((1000,500))*np.arange(500)).astype("float32")
    outputFileName = result("util_raster1.tif")

    createRaster( bounds=(10, 30, 15, 40), output=outputFileName, pixelHeight=0.01, pixelWidth=0.01, compress=True, 
                  srs=EPSG4326, noDataValue=100, data=data, overwrite=True, meta=dict(bob="bob",TIM="TIMMY"))

    ds = gdal.Open(outputFileName)
    bd = ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    if(not srs.IsSame(EPSG4326)): error("Creating raster on disk - srs mismatch")

    arr = bd.ReadAsArray()
    if(arr.sum() != data.sum()): error("Creating raster on disk - data mismatch")

    meta = ds.GetMetadata_Dict()
    if not meta["bob"] == "bob" or not meta["TIM"] == "TIMMY":
        error("saving meta data")

## Get values directly from a raster
def test_extractValues():
    points = [(6.06590,50.51939), (6.02141,50.61491), (6.371634,50.846025)]
    realValue = [24, 3, 23]
    realDiffs = [(-0.18841865745838504, -0.1953854267578663), 
                 (0.03190063584128211, -0.019478775579500507), 
                 (0.18415527009869948, 0.022563403500242885)]

    # test simple case
    v1 = extractValues(CLC_RASTER_PATH, points)

    for v, real in zip(v1, realValue): 
        if not (v.data==real) : 
            error("extractValues 1")
    for v, real in zip(v1, realDiffs): 
        if not ( isclose(v.xOffset,real[0]) or isclose(v.yOffset,real[1])): 
            error("extractValues 1")

    # test flipped 
    v2 = extractValues(CLC_FLIPCHECK_PATH, points)

    for v, real in zip(v2, realValue): 
        if not (v.data==real) : 
            error("extractValues 2")
    for v, real in zip(v2, realDiffs): 
        if not ( isclose(v.xOffset,real[0]) or isclose(v.yOffset,real[1])): 
            error("extractValues 2")

    # test point input
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(4061794.7,3094718.4)
    pt.AssignSpatialReference(EPSG3035)

    v3 = extractValues(CLC_RASTER_PATH, pt)

    if not (v3.data==3): error("extractValues 3")
    if not isclose(v3.xOffset, 0.44700000000187856): error("extractValues 3")
    if not isclose(v3.yOffset, 0.31600000000094042): error("extractValues 3")

    # test window fetch
    real = np.array([[ 12, 12, 12, 12, 12  ],
                     [ 12, 12, 12, 12, 12  ],
                     [ 12, 12,  3,  3, 12  ],
                     [ 12, 12, 12,  3,  3  ],
                     [ 12,  3,  3,  3,  3  ]])

    v4 = extractValues(CLC_RASTER_PATH, pt, winRange=2)
    if not isclose(np.abs(v4.data-real).sum(),0.0): error("extractValues 4")


# A nicer way to get a single value
def test_interpolateValues():
    point = (4061794.7,3094718.4)

    if not isclose(interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="near"), 3): 
        error("interpolateValues - ")
    if not isclose(interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="linear-spline"), 4.572732): 
        error("interpolateValues - linear-spline")
    if not isclose(interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="cubic-spline"), 2.4197586642): 
        error("interpolateValues - cubic-spline")
    if not isclose(interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="average"), 9.0612244898): 
        error("interpolateValues - average")
    if not isclose(interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="func", func = lambda d,xo,yo: d.max()),12): 
        error("interpolateValues - func")

def test_gradient():
    # create a sloping surface dataset
    x, y = np.meshgrid( np.abs(np.arange(-100,100)), np.abs(np.arange(-150,150)) )
    arr = np.ones( (300,200) ) + 0.01*y + x*0.03
    slopingDS = createRaster(bounds=(0,0,200,300), pixelWidth=1.0, pixelHeight=1.0, data=arr, srs=None )

    # do tests
    total = gradient( slopingDS , mode ="total", asMatrix=True)
    if not isclose(  total.mean(), 0.0312809506031): error("gradient - total - mean")
    if not isclose(total[ 10, 10], 0.0316227766017): error("gradient - total - nw quartile")
    if not isclose(total[200, 10], 0.0316227766017): error("gradient - total - sw quartile")
    if not isclose(total[ 10,150], 0.0316227766017): error("gradient - total - ne quartile")
    if not isclose(total[200,150], 0.0316227766017): error("gradient - total - se quartile")

    ns = gradient( slopingDS, mode ="north-south", asMatrix=True)
    if not isclose(  ns.mean(), -3.33333333333e-05): error("gradient - north-south - mean")
    if not isclose(ns[ 10, 10], -0.01): error("gradient - north-south - nw quartile")
    if not isclose(ns[200, 10], 0.01): error("gradient - north-south - sw quartile")
    if not isclose(ns[ 10,150], -0.01): error("gradient - north-south - ne quartile")
    if not isclose(ns[200,150], 0.01): error("gradient - north-south - se quartile")

    ew = gradient( slopingDS , mode ="east-west", asMatrix=True)
    if not isclose(ew.mean()  ,0.00015): error("gradient - east-west - mean")
    if not isclose(ew[ 10, 10],0.03): error("gradient - east-west - nw quartile")
    if not isclose(ew[200, 10],0.03): error("gradient - east-west - sw quartile")
    if not isclose(ew[ 10,150],-0.03): error("gradient - east-west - ne quartile")
    if not isclose(ew[200,150],-0.03): error("gradient - east-west - se quartile")

    aspect = gradient( slopingDS , mode ="dir", asMatrix=True)
    if not isclose(aspect.mean()  , 0.0101786336761): error("gradient - aspect - mean")
    if not isclose(180*aspect[ 10, 10]/np.pi, -18.4349488229): error("gradient - aspect - nw quartile")
    if not isclose(180*aspect[200, 10]/np.pi, 18.4349488229): error("gradient - aspect - sw quartile")
    if not isclose(180*aspect[ 10,150]/np.pi, -161.565051177): error("gradient - aspect - ne quartile")
    if not isclose(180*aspect[200,150]/np.pi, 161.565051177): error("gradient - aspect - se quartile")

    # calculate elevation slope
    output = result("slope_calculation.tif")
    slopeDS = gradient(ELEVATION_PATH, factor='latlonToM', output=output, overwrite=True)
    slopeMat = extractMatrix(output)

    if not isclose(slopeMat.mean(),0.0663805622803): error("gradient - elevation slope")

def test_mutateValues():
    # Setup
    def isOdd(mat): return np.mod(mat,2)
    
    source = gdal.Open(CLC_RASTER_PATH)
    sourceInfo = rasterInfo(source)

    ## Process Raster with no processor or extent
    res1 = mutateValues(source, processor=None)#, overwrite=True, output=result("algorithms_mutateValues_1.tif"))

    info1 = rasterInfo(res1)
    if not info1.srs.IsSame(sourceInfo.srs): error("mutateValues 1 - srs")
    if not info1.bounds == sourceInfo.bounds: error("mutateValues 1 - bounds")

    ## mutateValues with a simple processor
    output2 = result("algorithms_mutateValues_2.tif")
    mutateValues(source, processor=isOdd, overwrite=True, output=output2)
    res2 = gdal.Open(output2)

    info2 = rasterInfo(res2)
    if not info2.srs.IsSame(sourceInfo.srs): error("mutateValues 2 - srs")
    if not isclose(info2.xMin, sourceInfo.xMin): error("mutateValues 2 - bounds")
    if not isclose(info2.xMax, sourceInfo.xMax): error("mutateValues 2 - bounds")
    if not isclose(info2.yMin, sourceInfo.yMin): error("mutateValues 2 - bounds")
    if not isclose(info2.yMax, sourceInfo.yMax): error("mutateValues 2 - bounds")

    band2 = res2.GetRasterBand(1)
    arr2 = band2.ReadAsArray()

    if not (arr2.sum()==156515): error("mutateValues 2 - data")

    ## Process Raster with a simple processor (flip check)
    output2f = output=result("algorithms_mutateValues_2f.tif")
    mutateValues(CLC_FLIPCHECK_PATH, processor=isOdd, overwrite=True, output=output2f)
    res2f = gdal.Open(output2f)

    info2f = rasterInfo(res2f)
    if not info2f.srs.IsSame(sourceInfo.srs): error("mutateValues 2f - srs")
    if not isclose(info2f.xMin, sourceInfo.xMin): error("mutateValues 2f - bounds")
    if not isclose(info2f.xMax, sourceInfo.xMax): error("mutateValues 2f - bounds")
    if not isclose(info2f.yMin, sourceInfo.yMin): error("mutateValues 2f - bounds")
    if not isclose(info2f.yMax, sourceInfo.yMax): error("mutateValues 2f - bounds")

    arr2f = extractMatrix(res2f)

    if not (arr2f.sum()==156515): error("mutateValues 2f - data")

    ## Check flipped data
    if not (arr2f==arr2).all(): error("mutateValues 2f - flipping error!")

if __name__=="__main__":
    test_gdalType()
    test_rasterInfo()
    test_createRaster()
    test_extractValues()
    test_interpolateValues()
    test_gradient()
    test_mutateValues()
