from helpers import *
from geokit import RegionMask, Extent
from geokit.geom import *
from geokit.vector import *
from geokit.util import *
from geokit.raster import rasterInfo


def test_RegionMask___init__():
    ext = Extent(0,0,100,100, srs=EPSG3035)
    # test succeed
    rm = RegionMask(ext, 1, mask=MASK_DATA)
    rm = RegionMask(ext, (1,1), mask=MASK_DATA)
    rm = RegionMask(ext, 2, geom=GEOM)

    # no mask or geometry given
    try:
        rm = RegionMask(ext, 1)
        error("RegionMask init fail")
    except GeoKitRegionMaskError as e:
        if not str(e)=='Either mask or geom should be defined':
            error("RegionMask init fail")
    else:
        error("RegionMask init fail")

    # bad pixel given
    try:
        rm = RegionMask(ext, (1,3), geom=GEOM)
        error("RegionMask bad pixel")
    except GeoKitRegionMaskError as e:
        if not str(e)=='The given extent does not fit the given pixelRes':
            error("RegionMask bad pixel")
    else:
        error("RegionMask bad pixel")

    # bad mask given
    try:
        rm = RegionMask(ext, (1,2), mask=MASK_DATA)
        error("RegionMask bad mask")
    except GeoKitRegionMaskError as e:
        if not str(e)=='Extent and pixels sizes do not correspond to mask shape':
            error("RegionMask bad mask")
    else:
        error("RegionMask bad mask")

    # bad geom given
    try:
        rm = RegionMask(ext, (1,2), geom="bananas")
        error("RegionMask bad geom type")
    except GeoKitRegionMaskError as e:
        if not str(e)=='geom is not an ogr.Geometry object':
            error("RegionMask bad geom type")
    else:
        error("RegionMask bad geom type")

    # bad geom given 2
    badGeom = GEOM.Clone()
    badGeom.AssignSpatialReference(None)

    try:
        rm = RegionMask(ext, (1,2), geom=badGeom)
        error("RegionMask bad geom srs")
    except GeoKitRegionMaskError as e:
        if not str(e)=='geom does not have an srs':
            error("RegionMask bad geom srs")
    else:
        error("RegionMask bad geom srs")

    print( "RegionMask___init__ passed")

def test_RegionMask_fromMask():
    ext = Extent(0,0,100,100, srs=EPSG3035)
    rm = RegionMask.fromMask(mask=MASK_DATA, extent=ext, attributes={"hats":5})

    if not rm.mask.sum()==MASK_DATA.sum(): error("fromSource - mask")
    if not rm.extent==ext: error("fromSource - extent")
    if not rm.srs.IsSame(ext.srs): error("fromSource - srs")
    print( "RegionMask_fromMask passed")

def test_RegionMask_fromGeom():
    # fromGeom with wkt
    rm1 = RegionMask.fromGeom( convertWKT(POLY, srs='latlon'), pixelRes=1000)
    if( rm1.extent.xXyY != (4329000.0, 4771000.0, 835000.0, 1682000.0)):
        error("fromGeom - extent bounds")
    if not ( rm1.extent.srs.IsSame(EPSG3035) ): error("fromGeom - extent srs")
    if( rm1.mask.sum()!=79274 ): error("fromGeom - mask")

    # fromGeom with geometry
    dxy = 0.05
    rm2 = RegionMask.fromGeom(GEOM, pixelRes=dxy, srs=EPSG4326, padExtent=0.2)
    
    if( not rm2.extent == Extent(9.90,30.30,14.80,38.30)):
        error("fromGeom - extent bounds")
    if not ( rm2.extent.srs.IsSame(EPSG4326) ): error("fromGeom - extent srs")
    
    g = GEOM.Clone()
    g.TransformTo(EPSG4326)
    if not abs(rm2.mask.sum()*dxy*dxy-g.Area())/g.Area()<0.01: # check if total areas are close to one another
        error("fromGeom - mask")

    # fromGeom with geometry and extent
    dxy = 0.05
    definedExtent = Extent(9.50,30.25,14.80,38.35)
    rm3 = RegionMask.fromGeom(GEOM, pixelRes=dxy, srs=EPSG4326, extent=definedExtent)
    
    if( not rm3.extent == definedExtent):
        error("fromGeom - extent bounds")
    if not ( rm3.extent.srs.IsSame(EPSG4326) ): error("fromGeom - extent srs")
    
    g = GEOM.Clone()
    g.TransformTo(EPSG4326)
    if not abs(rm3.mask.sum()*dxy*dxy-g.Area())/g.Area()<0.01: # check if total areas are close to one another
        error("fromGeom - mask")

    print( "RegionMask_fromGeom passed")

def test_RegionMask_fromVector():
    # fromVector with a padded extent and defined srs
    rm0 = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelRes=0.001, srs=EPSG4326, padExtent=0.1)
    if( rm0.mask.sum()!=90296 ): error("fromVector - mask")

    # fromVector - ID select
    rm1 = RegionMask.fromVector(MULTI_FTR_SHAPE_PATH, where=1)
    
    if not (rm1.extent == Extent(4069100, 2867000, 4109400, 2954000, srs=EPSG3035)):
        error("fromVector - extent bounds")
    if not (rm1.attributes["name"]=="dog"): error("fromVector - attributes")

    ds = ogr.Open(MULTI_FTR_SHAPE_PATH)
    lyr = ds.GetLayer()
    ftr = lyr.GetFeature(1)
    g = ftr.GetGeometryRef().Clone()
    g.TransformTo(EPSG3035)
    if not abs(rm1.mask.sum()*100*100-g.Area())/g.Area()<0.01: # check if total areas are close to one another
        error("fromVector - mask area")
    
    # fromVector - 'where' select
    rm2 = RegionMask.fromVector(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelRes=0.01, where="name='monkey'")
    
    if not (rm2.extent == Extent(6.83, 49.52, 7.53, 49.94)):
        error("fromVector - extent bounds")
    if not (rm2.attributes["id"]==3): error("fromVector - attributes")

    ftr = lyr.GetFeature(3)
    g = ftr.GetGeometryRef().Clone()
    g.TransformTo(EPSG4326)
    if not isclose(rm2.mask.sum(),1948.0): 
        error("fromVector - mask area")

    # fromVector - 'where' select fail no features
    try:
        rm3 = RegionMask.fromVector(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelRes=0.01, where="name='monkeyy'")
        error("fromVector - fail no features")
    except GeoKitRegionMaskError as e:
        if not str(e)=='Zero features found':
            error("fromVector - fail no features")
    else:
        error("fromVector - fail no features")

    # fromVector - 'where' finds many features
    try:
        rm4 = RegionMask.fromVector(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelRes=0.01, where=r"name like 'mo%'")
        error("fromVector - fail multi features")
    except GeoKitRegionMaskError as e:
        if not 'Multiple fetures found' in str(e):
            error("fromVector - fail multi features")
    else:
        error("fromVector - fail multi features")

    print( "RegionMask_fromVector passed")

def test_RegionMask_load():

    print( "RegionMask_load not tested...")

def test_RegionMask_pixelRes():
    # test succeed
    rm1 = RegionMask.fromMask(Extent(0,0,100,100,srs=EPSG3035), MASK_DATA)
    ps = rm1.pixelRes
    if not ps==1: error( "pixelRes")

    # test fail
    rm2 = RegionMask.fromMask(Extent(0,0,100,200,srs=EPSG3035), MASK_DATA)
    try:
        ps = rm2.pixelRes
        error("pixelRes - fail test")
    except GeoKitRegionMaskError as e:
        if not str(e)=='pixelRes only accessable when pixelWidth equals pixelHeight':
            error("pixelRes - fail test")
    else:
        error("pixelRes - fail test")
    print( "RegionMask_pixelRes passed")

def test_RegionMask_buildMask():
    # Build from another srs
    rm = RegionMask.load(AACHEN_SHAPE_PATH, srs=EPSG3035, pixelRes=100)
    compare( rm.mask.sum(), 70944  )
    compare( rm.mask.std(), 0.498273451386  )

    print( "RegionMask_buildMask passed")

def test_RegionMask_area():
    print( "RegionMask_area not tested...")

def test_RegionMask_buildGeometry():

    # setup
    rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)
    rm2.buildMask() # Be sure the mask is in place

    # Get the "real" geometry
    ds = ogr.Open(AACHEN_SHAPE_PATH)
    lyr = ds.GetLayer()
    ftr = lyr.GetFeature(0)
    realGeom = ftr.GetGeometryRef().Clone()
    realGeom.TransformTo(EPSG3035)

    # check initial geom and real geom area
    if not abs(rm2.geometry.Area()-realGeom.Area())/realGeom.Area() < 0.001:
        error("geometry - reading from file")

    # destroy and recreate
    rm2.buildGeometry()
    if not abs( (rm2.geometry.Area()-realGeom.Area())/realGeom.Area() ) < 0.001:
        error("geometry - building geometry")

    
    print( "RegionMask_buildGeometry passed")

def test_RegionMask_vectorPath():
    rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)
    vec = rm2.vectorPath

    if not isfile(vec): error("vector creation")

    del rm2

    if isfile(vec): error("vector deletion")

    print( "RegionMask_vectorPath passed")

def test_RegionMask_vector():

    rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)
    vec = rm2.vector
    vec.GetLayer()

    if not isVector(vec): error("vector creation")
    del vec

    if not isVector(rm2._vector): error("vector retention")

    print( "RegionMask_vector passed")

def test_RegionMask__repr_svg_():
    print( "RegionMask__repr_svg_ not tested...")

def test_RegionMask_drawMask():
    print( "RegionMask_drawMask not tested...")

def test_RegionMask_drawGeometry():
    print( "RegionMask_drawGeometry not tested...")

def test_RegionMask_applyMask():
    ## setup
    rm = RegionMask.fromGeom(point(6.20,50.75).Buffer(0.05), srs=EPSG4326, pixelRes=0.001)

    data1 = np.arange(rm.mask.size).reshape(rm.mask.shape)
    data2 = np.arange(rm.mask.shape[0]*3*rm.mask.shape[1]*3).reshape((rm.mask.shape[0]*3,rm.mask.shape[1]*3))

    ## test applying
    data1 = rm.applyMask(data1)
    if not (data1.sum()==39296070 and abs(data1.std()-3020.0893432)<0.001):
        error("applyMask 1 - data mismatch")

    data2 = rm.applyMask(data2.astype('int64'))
    if not (data2.sum()==3183264630 and abs(data2.std()-27182.1342973)<0.001):
        error("applyMask 2 - data mismatch")

    #rm.createRaster(output=result("regionMask_applyMask_1.tif"), data=data1, overwrite=True)
    #rm.createRaster(3, output=result("regionMask_applyMask_2.tif"), data=data2, overwrite=True)

    print( "RegionMask_applyMask passed")

def test_RegionMask__returnBlank():
    print( "RegionMask__returnBlank not tested...")

def test_RegionMask_indicateValues():
    ## Setup
    rm = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelRes=0.001, srs=EPSG4326)

    # Testing valueMin (with srs change)
    res1 = rm.indicateValues(CLC_RASTER_PATH, value=(20,None))
    
    if not (isclose(res1.sum(),30969.6796875, 1e-6) and isclose(res1.std(),0.3489773, 1e-6)): 
        error("indicateValues - valueMin")

    # Testing valueMax (with srs change)
    res2 = rm.indicateValues(CLC_RASTER_PATH, value=(None,24))
    if not (isclose(res2.sum(),82857.5078125, 1e-6) and isclose(res2.std(), 0.4867994, 1e-6)): 
        error("indicateValues - valueMax")

    # Testing valueEquals (with srs change)
    res3 = rm.indicateValues(CLC_RASTER_PATH, value=7, resampleAlg="cubic")
    if not (isclose(res3.sum(),580.9105835, 1e-4) and isclose(res3.std(),0.0500924, 1e-6)): 
        error("indicateValues - valueEquals")

    # Testing range
    res4 = rm.indicateValues(CLC_RASTER_PATH, value=(20,24))
    
    combi = np.logical_and(res1>0.5, res2>0.5)
    # Some pixels will not end up the same due to warping issues
    if not ( (res4>0.5) !=combi).sum()<res4.size*0.001:error("indicateValues - range")

    # Testing buffering
    res5 = rm.indicateValues(CLC_RASTER_PATH, value=(1,2), buffer=0.01, resolutionDiv=2, forceMaskShape=True)
    if not isclose(res5.sum(), 65030.75000000, 1e-4):error("indicateValues - grown")

    # make sure we get an empty mask when nothing is indicated
    res6 = rm.indicateValues(CLC_RASTER_PATH, value=2000, buffer=0.01, resolutionDiv=2, forceMaskShape=True, noData=-1)
    if not isclose(res6.sum(), -113526.0, 1e-4):error("indicateValues - empty")

    print( "RegionMask_indicateValues passed")

def test_RegionMask_indicateFeatures():
    # setup
    rm = RegionMask.fromVector(AACHEN_SHAPE_PATH)

    # Simple case
    res = rm.indicateFeatures(NATURA_PATH, where="SITECODE='DE5404303'")
    #print("%.7f"%res.sum(), "%.7f"%res.std()) 
    if not (isclose(res.sum(),649,1e-6) and isclose(res.std(),0.0646270,1e-6)): error("indicateFeatures - Simple indication")

    # Buffered Cases
    res2 = rm.indicateFeatures(NATURA_PATH, where="SITETYPE='B'", buffer=300, resolutionDiv=3, forceMaskShape=True)
    #print("%.7f"%res2.sum(), "%.7f"%res2.std())
    if not isclose(res2.sum(),13670.5555556, 1e-6): error("indicateFeatures - grown indication - 1")

    res3 = rm.indicateFeatures(NATURA_PATH, where="SITETYPE='B'", buffer=300, bufferMethod='area', resolutionDiv=5, forceMaskShape=True)
    #print("%.7f"%res3.sum(), "%.7f"%res3.std())
    if not isclose(res3.sum(),13807.320000, 1e-6): error("indicateFeatures - grown indication - 2")

    # No indication case
    res4 = rm.indicateFeatures(NATURA_PATH, where="SITETYPE='D'", buffer=300, bufferMethod='area', resolutionDiv=2, forceMaskShape=True, noData=-1)
    #print("%.7f"%res4.sum(), "%.7f"%res4.std())
    
    if not isclose(res4.sum(), -83792, 1e-6): error("indicateFeatures - empty case")

    print( "RegionMask_indicateFeatures passed")

def test_RegionMask_indicateGeoms():
    print( "RegionMask_indicateGeoms not tested...")

def test_RegionMask_subRegions():
    print( "RegionMask_subRegions not tested...")

def test_RegionMask_createRaster():

    rm = RegionMask.fromGeom(point(6.20,50.75).Buffer(0.05), srs=EPSG4326, pixelRes=0.001)

    ## Create a raster like the mask
    ds = rm.createRaster()

    dsInfo = rasterInfo(ds)
    if not abs(dsInfo.xMin- 6.15)<0.001: error("createRaster 1 - extent")
    if not abs(dsInfo.xMax- 6.25)<0.001: error("createRaster 1 - extent")
    if not abs(dsInfo.yMin-50.70)<0.001: error("createRaster 1 - extent")
    if not abs(dsInfo.yMax-50.80)<0.001: error("createRaster 1 - extent")
    if not dsInfo.srs.IsSame(EPSG4326): error("createRaster 1 - srs")
    if not dsInfo.dtype==gdal.GDT_Byte: error("createRaster 1 - dtype")

    # Fill a raster with mask data
    out2 = result("rasterMast_createRaster_2.tif")
    rm.createRaster(output=out2, data=rm.mask, overwrite=True)

    ds = gdal.Open(out2)
    band = ds.GetRasterBand(1)
    if (band.ReadAsArray()-rm.mask).any(): error("createRaster 2 - data mismatch")

    # test Scaling down
    scaledData = scaleMatrix(rm.mask,-4)

    ds = rm.createRaster(resolutionDiv=1/4, data=scaledData, overwrite=True)

    band = ds.GetRasterBand(1)
    if (band.ReadAsArray()-scaledData).any(): error("createRaster 3 - data mismatch")  

    # test Scaling up
    scaledData = scaleMatrix(rm.mask,2)

    ds = rm.createRaster(resolutionDiv=2, data=scaledData, overwrite=True)
    band = ds.GetRasterBand(1)
    if (band.ReadAsArray()-scaledData).any(): error("createRaster 4 - data mismatch")  

    print( "RegionMask_createRaster passed")

def test_RegionMask_warp():
    ## setup
    rm_3035 = RegionMask.fromGeom(point(6.20,50.75).Buffer(0.05))
    rm = RegionMask.fromGeom(point(6.20,50.75).Buffer(0.05), srs=EPSG4326, pixelRes=0.0005)

    ## basic warp Raster
    warped_1 = rm_3035.warp(CLC_RASTER_PATH)

    if not (warped_1.dtype==np.uint8): error("warp 1 - dtype")
    if not (warped_1.shape==rm_3035.mask.shape): error("warp 1 - shape")
    if not (isclose(warped_1.sum(),88128) and isclose(warped_1.std(),9.52214123991)):
        error("warp 1 - result")
    #rm_3035.createRaster(data=warped_1, output=result("regionMask_warp_1.tif"), overwrite=True)

    ## basic warp Raster (FLIP CHECK!)
    warped_1f = rm_3035.warp(CLC_FLIPCHECK_PATH)

    if not (warped_1f.dtype==np.uint8): error("warp 1f - dtype")
    if not (warped_1f.shape==rm_3035.mask.shape): error("warp 1f - shape")
    if not (isclose(warped_1f.sum(),88128) and isclose(warped_1f.std(),9.52214123991)):
        error("warp 1f - result")
    #rm_3035.createRaster(data=warped_1f, output=result("regionMask_warp_1f.tif"), overwrite=True)

    if not (warped_1==warped_1f).all(): error("warp - flipping error")

    ## basic warp Raster with srs change
    warped_2 = rm.warp(CLC_RASTER_PATH)
    if not (warped_2.dtype==np.uint8): error("warp 2 - dtype")
    if not (warped_2.shape==rm.mask.shape): error("warp 2 - shape")
    if not (isclose(warped_2.sum(),449627) and isclose(warped_2.std(),9.07520801659)):
        error("warp 2 - result")
    #rm.createRaster(data=warped_2, output=result("regionMask_warp_2.tif"), overwrite=True)


    ## Define resample alg and output type
    warped_3 = rm.warp(CLC_RASTER_PATH, dtype="float", resampleAlg='near')

    if not (warped_3.dtype==np.float64): error("warp 3 - dtype")
    if not (warped_3.shape==rm.mask.shape): error("warp 3 - shape")
    if not (isclose(warped_3.sum(),449317.0) and isclose(warped_3.std(),9.37570375729)):
        error("warp 3 - result")
    #rm.createRaster(data=warped_3, output=result("regionMask_warp_3.tif"), overwrite=True)

    ## define a resolution div
    warped_4 = rm.warp(CLC_RASTER_PATH, resolutionDiv=5, resampleAlg='near', noDataValue=0)

    if not (warped_4.dtype==np.uint8): error("warp 4 - dtype")
    if not (warped_4.shape==(rm.mask.shape[0]*5, rm.mask.shape[1]*5)): error("warp 4 - shape")
    if not (isclose(warped_4.sum(),11240881) and isclose(warped_4.std(),9.37633272361)):
        error("warp 4 - result")

    #rm.createRaster(5, data=warped_4, output=result("regionMask_warp_4.tif"), noDataValue=0, overwrite=True)

    print( "RegionMask_warp passed")

def test_RegionMask_rasterize():
    ## setup
    rm = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelRes=0.001, srs=EPSG4326)

    ## simple rasterize
    rasterize_1 = rm.rasterize( AACHEN_ZONES )

    if not (rasterize_1.dtype==np.uint8): error("rasterize 1 - dtype")
    if not (rasterize_1.shape==rm.mask.shape): error("rasterize 1 - shape")
    if not (isclose(rasterize_1.sum(),47191) and isclose(rasterize_1.std(),0.42181050527)):
        error("rasterize 1 - result")
    #rm.createRaster(data=rasterize_1, output=result("regionMask_rasterize_1.tif"), overwrite=True)

    # attribute rasterizing
    rasterize_2 = rm.rasterize( AACHEN_ZONES, attribute="YEAR", dtype="int16" )

    if not (rasterize_2.dtype==np.int16): error("rasterize 2 - dtype")
    if not (rasterize_2.shape==rm.mask.shape): error("rasterize 2 - shape")
    if not (isclose(rasterize_2.sum(),94219640) and isclose(rasterize_2.std(),842.177748527)):
        error("rasterize 2 - result")
    #rm.createRaster(data=rasterize_2, output=result("regionMask_rasterize_2.tif"), overwrite=True)

    # where statement and resolution div
    rasterize_3 = rm.rasterize( AACHEN_ZONES, burnValues=[10], resolutionDiv=5, where="YEAR>2000", dtype=float )

    if not (rasterize_3.dtype==np.float64): error("rasterize 3 - dtype")
    if not (rasterize_3.shape==(rm.mask.shape[0]*5, rm.mask.shape[1]*5)): 
        error("rasterize 3 - shape")
    if not (isclose(rasterize_3.sum(),4578070.0) and isclose(rasterize_3.std(),2.85958813405)):
        error("rasterize 3 - result")
    #rm.createRaster(data=scaleMatrix(rasterize_3,-5), output=result("regionMask_rasterize_3.tif"), overwrite=True)

    print( "RegionMask_rasterize passed")

def test_RegionMask_extractFeatures():
    print( "RegionMask_extractFeatures not tested...")

def test_RegionMask_mutateVector():
    print( "RegionMask_mutateVector not tested...")

def test_RegionMask_mutateRaster():
    print( "RegionMask_mutateRaster not tested...")


if __name__=="__main__":
    # test_RegionMask___init__()
    # test_RegionMask_fromMask()
    # test_RegionMask_fromGeom()
    # test_RegionMask_fromVector()
    # test_RegionMask_load()
    # test_RegionMask_pixelRes()
    # test_RegionMask_buildMask()
    # test_RegionMask_area()
    # test_RegionMask_buildGeometry()
    # test_RegionMask_vectorPath()
    # test_RegionMask_vector()
    # test_RegionMask__repr_svg_()
    # test_RegionMask_drawMask()
    # test_RegionMask_drawGeometry()
    # test_RegionMask_applyMask()
    # test_RegionMask__returnBlank()
    # test_RegionMask_indicateValues()
    test_RegionMask_indicateFeatures()
    test_RegionMask_indicateGeoms()
    test_RegionMask_subRegions()
    test_RegionMask_createRaster()
    test_RegionMask_warp()
    test_RegionMask_rasterize()
    test_RegionMask_extractFeatures()
    test_RegionMask_mutateVector()
    test_RegionMask_mutateRaster()