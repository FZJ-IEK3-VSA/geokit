from helpers import *
from geokit.gk import *

## gdalType
def test_gdalType():
    if( gdalType(bool) != "GDT_Byte" ): error("gdal type")
    if( gdalType("InT64") != "GDT_Int32" ): error("gdal type")
    if( gdalType("float32") != "GDT_Float32" ): error("gdal type")
    if( gdalType(NUMPY_FLOAT_ARRAY) != "GDT_Float64" ): error("gdal type")
    if( gdalType(NUMPY_FLOAT_ARRAY.dtype) != "GDT_Float64" ): error("gdal type")

    print( "gdalType passed" )

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


    print( "rasterInfo passed" )

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
                           dtype=inputDataType, noData=inputNoData, fillValue=inputFillValue)

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


    print( "createRaster passed" )

## Get values directly from a raster
def test_extractValues():
    points = [(6.06590,50.51939), (6.02141,50.61491), (6.371634,50.846025)]
    realValue = [24, 3, 23]
    realDiffs = [(-0.18841865745838504, -0.1953854267578663), 
                 (0.03190063584128211, -0.019478775579500507), 
                 (0.18415527009869948, 0.022563403500242885)]

    # test simple case
    v1 = extractValues(CLC_RASTER_PATH, points)
    for v, real in zip(v1.itertuples(), realValue): 
        if not (v.data == real) : 
            error("extractValues 1")
    for v, real in zip(v1.itertuples(), realDiffs): 
        if not ( isclose(v.xOffset,real[0]) or isclose(v.yOffset,real[1])): 
            error("extractValues 1")

    # test flipped 
    v2 = extractValues(CLC_FLIPCHECK_PATH, points)

    for v, real in zip(v2.itertuples(), realValue): 
        if not (v.data==real) : 
            error("extractValues 2")
    for v, real in zip(v2.itertuples(), realDiffs): 
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

    print( "extractValues passed" )

# A nicer way to get a single value
def test_interpolateValues():
    point = (4061794.7,3094718.4)
    
    v = interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="near")
    compare( v, 3, "interpolateValues - ")
    
    v = interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="linear-spline")
    compare( v, 4.572732, "interpolateValues - linear-spline")
    
    v = interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="cubic-spline")
    compare( v, 2.4197586642, "interpolateValues - cubic-spline")
    
    v = interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="average")
    compare( v, 9.0612244898, "interpolateValues - average")
    
    v = interpolateValues(CLC_RASTER_PATH, point, pointSRS='europe_m', mode="func", func = lambda d,xo,yo: d.max())
    compare( v,12, "interpolateValues - func")

    print("interpolateValues passed")

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

    print("gradient passed")

def test_mutateRaster():
    # Setup
    def isOdd(mat): return np.mod(mat,2)
    
    source = gdal.Open(CLC_RASTER_PATH)
    sourceInfo = rasterInfo(source)

    ## Process Raster with no processor or extent
    res1 = mutateRaster(source, processor=None)#, overwrite=True, output=result("algorithms_mutateRaster_1.tif"))

    info1 = rasterInfo(res1)
    if not info1.srs.IsSame(sourceInfo.srs): error("mutateRaster 1 - srs")
    if not info1.bounds == sourceInfo.bounds: error("mutateRaster 1 - bounds")

    ## mutateRaster with a simple processor
    output2 = result("algorithms_mutateRaster_2.tif")
    mutateRaster(source, processor=isOdd, overwrite=True, output=output2)
    res2 = gdal.Open(output2)

    info2 = rasterInfo(res2)
    if not info2.srs.IsSame(sourceInfo.srs): error("mutateRaster 2 - srs")
    if not isclose(info2.xMin, sourceInfo.xMin): error("mutateRaster 2 - bounds")
    if not isclose(info2.xMax, sourceInfo.xMax): error("mutateRaster 2 - bounds")
    if not isclose(info2.yMin, sourceInfo.yMin): error("mutateRaster 2 - bounds")
    if not isclose(info2.yMax, sourceInfo.yMax): error("mutateRaster 2 - bounds")

    band2 = res2.GetRasterBand(1)
    arr2 = band2.ReadAsArray()

    if not (arr2.sum()==156515): error("mutateRaster 2 - data")

    ## Process Raster with a simple processor (flip check)
    output2f = output=result("algorithms_mutateRaster_2f.tif")
    mutateRaster(CLC_FLIPCHECK_PATH, processor=isOdd, overwrite=True, output=output2f)
    res2f = gdal.Open(output2f)

    info2f = rasterInfo(res2f)
    if not info2f.srs.IsSame(sourceInfo.srs): error("mutateRaster 2f - srs")
    if not isclose(info2f.xMin, sourceInfo.xMin): error("mutateRaster 2f - bounds")
    if not isclose(info2f.xMax, sourceInfo.xMax): error("mutateRaster 2f - bounds")
    if not isclose(info2f.yMin, sourceInfo.yMin): error("mutateRaster 2f - bounds")
    if not isclose(info2f.yMax, sourceInfo.yMax): error("mutateRaster 2f - bounds")

    arr2f = extractMatrix(res2f)

    if not (arr2f.sum()==156515): error("mutateRaster 2f - data")

    ## Check flipped data
    if not (arr2f==arr2).all(): error("mutateRaster 2f - flipping error!")

    print( "mutateRaster passed")

def test_isRaster(): 
    s1 = isRaster(CLC_RASTER_PATH)
    compare(s1,True,"isRaster")


    s2 = isRaster(AACHEN_SHAPE_PATH)
    compare(s2,False,"isRaster")

    s3 = isRaster(loadRaster(CLC_RASTER_PATH))
    compare(s3, True, "isRaster")


    print("isRaster passed")

def test_loadRaster(): print( "loadRaster is trivial")
def test_createRasterLike(): print( "createRasterLike not tested...")
def test_extractMatrix(): print( "extractMatrix not tested...")
def test_extractCutline(): print( "extractCutline not tested...")

def test_rasterStats(): 
    result = rasterStats(CLC_RASTER_PATH, AACHEN_SHAPE_PATH)
    compare(result.mean, 15.711518944519621)

    print("rasterStats passed")

def test_KernelProcessor(): print( "KernelProcessor not tested...")
def test_indexToCoord(): print( "indexToCoord not tested...")
def test_drawRaster(): 
    r = drawRaster(AACHEN_URBAN_LC)
    plt.savefig(result("drawRaster-1.png"), dpi=100)

    # shift
    r = drawRaster(AACHEN_URBAN_LC, rightMargin=0.2)
    plt.savefig(result("drawRaster-2.png"), dpi=100)

    # projection
    r = drawRaster(AACHEN_URBAN_LC, srs=4326)
    plt.savefig(result("drawRaster-3.png"), dpi=100)

    # cutline
    r = drawRaster(AACHEN_URBAN_LC, cutline=AACHEN_SHAPE_PATH, resolution=0.001, srs=4326)
    plt.savefig(result("drawRaster-4.png"), dpi=100)


    print("No errors in drawRaster, but they should be checked manually!")

def test_polygonizeRaster():
    geoms = polygonizeRaster(AACHEN_URBAN_LC)
    compare(geoms.shape[0], 423, "polygonizeRaster - geom count")
    is3 = geoms.value==3
    compare(is3.sum(), 2, "polygonizeRaster - value count")
    compare(geoms.geom[is3].apply(lambda x: x.Area()).sum(), 120529999.18190208, "polygonizeRaster - geom area")

    geoms = polygonizeRaster(AACHEN_URBAN_LC, flat=True)
    compare(geoms.shape[0], 3, "polygonizeRaster - geom count")
    is3 = geoms.value==3
    compare(is3.sum(), 1, "polygonizeRaster - value count")
    compare(geoms.geom[is3].apply(lambda x: x.Area()).sum(), 120529999.18190208, "polygonizeRaster - geom area")

    print( "polygonizeRaster passed")

def test_warp(): 
    # Change resolution to disk
    d = warp( CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200, output=result("warp1.tif") )
    v1 = extractMatrix(d)
    compare( v1.mean(), 16.3141463057, "warp - value" )

    # change resolution to memory
    d = warp( CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200 )
    v2 = extractMatrix(d)
    compare( (v1-v2).mean(), 0)

    # Do a cutline from disk
    d = warp( CLC_RASTER_PATH, cutline=AACHEN_SHAPE_PATH, output=result("warp3.tif"), noData=99 )
    v3 = extractMatrix(d)
    compare(v3.mean(), 89.9568135904) 
    compare(v3[0,0], 99, "warp -noData")
    
    # Do a cutline from memory
    d = warp( CLC_RASTER_PATH, cutline=box(*AACHEN_SHAPE_EXTENT_3035, srs=EPSG3035), noData=99 )
    v4 = extractMatrix(d)
    compare(v4[0,0], 99, "warp -noData")
    compare(v4.mean(), 76.72702479) 

    # Do a flipped-source check
    d = warp( CLC_FLIPCHECK_PATH, cutline=box(*AACHEN_SHAPE_EXTENT_3035, srs=EPSG3035), noData=99 )
    v5 = extractMatrix(d)
    compare( (v4-v5).mean(), 0)

    d = warp( CLC_FLIPCHECK_PATH, pixelHeight=200, pixelWidth=200, output=result("warp6.tif") )
    v6 = extractMatrix(d)
    compare( (v1-v6).mean(), 0)

    print( "warp passed")

if __name__=="__main__":
    test_isRaster()
    test_loadRaster()
    test_gdalType()
    test_createRaster()
    test_createRasterLike()
    test_extractMatrix()
    test_extractCutline()
    test_rasterStats()
    test_gradient()
    test_rasterInfo()
    test_extractValues()
    test_interpolateValues()
    test_mutateRaster()
    test_KernelProcessor()
    test_indexToCoord()
    test_warp()
    test_drawRaster()
    test_polygonizeRaster()
    