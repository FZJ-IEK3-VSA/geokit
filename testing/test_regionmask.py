from helpers import *
from geokit import RegionMask, Extent
from geokit.geom import makePoint, convertWKT
from geokit.util import GeoKitRegionMaskError, scaleMatrix
from geokit.raster import rasterInfo


def test_init():
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
        if not str(e)=='The given extent does not fit the given pixelSize':
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

def test_fromMask():
    ext = Extent(0,0,100,100, srs=EPSG3035)
    rm = RegionMask.fromMask(mask=MASK_DATA, extent=ext, attributes={"hats":5})

    if not rm.mask.sum()==MASK_DATA.sum(): error("fromSource - mask")
    if not rm.extent==ext: error("fromSource - extent")
    if not rm.srs.IsSame(ext.srs): error("fromSource - srs")


def test_fromGeom():
    # fromGeom with wkt
    rm1 = RegionMask.fromGeom( convertWKT(POLY, srs='latlon'), pixelSize=1000)
    if( rm1.extent.xXyY != (4329000.0, 4771000.0, 835000.0, 1682000.0)):
        error("fromGeom - extent bounds")
    if not ( rm1.extent.srs.IsSame(EPSG3035) ): error("fromGeom - extent srs")
    if( rm1.mask.sum()!=79274 ): error("fromGeom - mask")

    # fromGeom with geometry
    dxy = 0.05
    rm2 = RegionMask.fromGeom(GEOM, pixelSize=dxy, srs=EPSG4326, padExtent=0.2)
    
    if( not rm2.extent == Extent(9.85,30.25,14.80,38.35)):
        error("fromGeom - extent bounds")
    if not ( rm2.extent.srs.IsSame(EPSG4326) ): error("fromGeom - extent srs")
    
    g = GEOM.Clone()
    g.TransformTo(EPSG4326)
    if not abs(rm2.mask.sum()*dxy*dxy-g.Area())/g.Area()<0.01: # check if total areas are close to one another
        error("fromGeom - mask")

    # fromGeom with geometry and extent
    dxy = 0.05
    definedExtent = Extent(9.50,30.25,14.80,38.35)
    rm3 = RegionMask.fromGeom(GEOM, pixelSize=dxy, srs=EPSG4326, extent=definedExtent)
    
    if( not rm3.extent == definedExtent):
        error("fromGeom - extent bounds")
    if not ( rm3.extent.srs.IsSame(EPSG4326) ): error("fromGeom - extent srs")
    
    g = GEOM.Clone()
    g.TransformTo(EPSG4326)
    if not abs(rm3.mask.sum()*dxy*dxy-g.Area())/g.Area()<0.01: # check if total areas are close to one another
        error("fromGeom - mask")


def test_fromVectorFeature():
    #source, select=0, pixelSize=RegionMask.DEFAULT_RES, srs=RegionMask.DEFAULT_SRS, extent=None, padExtent=RegionMask.DEFAULT_PAD, **kwargs):
    # fromFourceFeature - ID select
    rm1 = RegionMask.fromVectorFeature(MULTI_FTR_SHAPE_PATH, select=1)

    if not (rm1.extent == Extent(4067700, 2866100, 4110200, 2954800, srs=EPSG3035)):
        error("fromVectorFeature - extent bounds")
    if not (rm1.attributes["name"]=="dog"): error("fromVectorFeature - attributes")

    ds = ogr.Open(MULTI_FTR_SHAPE_PATH)
    lyr = ds.GetLayer()
    ftr = lyr.GetFeature(1)
    g = ftr.GetGeometryRef().Clone()
    g.TransformTo(EPSG3035)
    if not abs(rm1.mask.sum()*100*100-g.Area())/g.Area()<0.01: # check if total areas are close to one another
        error("fromVectorFeature - mask area")
    
    # fromVectorFeature - 'where' select
    rm2 = RegionMask.fromVectorFeature(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelSize=0.01, select="name='monkey'")

    if not (rm2.extent == Extent(6.83, 49.52, 7.53, 49.94)):
        error("fromVectorFeature - extent bounds")
    if not (rm2.attributes["id"]==3): error("fromVectorFeature - attributes")

    ftr = lyr.GetFeature(3)
    g = ftr.GetGeometryRef().Clone()
    g.TransformTo(EPSG4326)
    if not isclose(rm2.mask.sum(),1948.0): 
        error("fromVectorFeature - mask area")

    # fromVectorFeature - 'where' select fail no features
    try:
        rm3 = RegionMask.fromVectorFeature(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelSize=0.01, select="name='monkeyy'")
        error("fromVectorFeature - fail no features")
    except GeoKitRegionMaskError as e:
        if not str(e)=='Zero features found':
            error("fromVectorFeature - fail no features")
    else:
        error("fromVectorFeature - fail no features")

    # fromVectorFeature - 'where' select fail too many features
    try:
        rm4 = RegionMask.fromVectorFeature(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelSize=0.01, select=r"name like 'mo%'")
        error("fromVectorFeature - fail multi features")
    except GeoKitRegionMaskError as e:
        if not str(e)=='Multiple fetures found':
            error("fromVectorFeature - fail multi features")
    else:
        error("fromVectorFeature - fail multi features")

def test_fromVector():
    # fromVector with a padded extent and defined srs
    rm1 = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelSize=0.001, srs=EPSG4326, padExtent=0.1)
    if( rm1.mask.sum()!=90296 ): error("fromVector - mask")

def test_pixelSize():
    # test succeed
    rm1 = RegionMask.fromMask(Extent(0,0,100,100,srs=EPSG3035), MASK_DATA)
    ps = rm1.pixelSize
    if not ps==1: error( "pixelSize")

    # test fail
    rm2 = RegionMask.fromMask(Extent(0,0,100,200,srs=EPSG3035), MASK_DATA)
    try:
        ps = rm2.pixelSize
        error("pixelSize - fail test")
    except GeoKitRegionMaskError as e:
        if not str(e)=='pixelSize only accessable when pixelWidth equals pixelHeight':
            error("pixelSize - fail test")
    else:
        error("pixelSize - fail test")

def test_mask():
    # create a RegionMask without a mask
    rm2 = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH,select=0)
    if not (isclose(rm2.mask.sum(),70944) and isclose(rm2.mask.std(), 0.48968559141)):
        error("mask creation")

def test_geometry():
  # setup
  rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)

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

def test_createRaster():
    rm = RegionMask.fromGeom(makePoint(6.20,50.75).Buffer(0.05), srs=EPSG4326, pixelSize=0.001)

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

def test_applyMask():
    ## setup
    rm = RegionMask.fromGeom(makePoint(6.20,50.75).Buffer(0.05), srs=EPSG4326, pixelSize=0.001)

    data1 = np.arange(rm.mask.size).reshape(rm.mask.shape)

    data2 = np.arange(rm.mask.shape[0]*3*rm.mask.shape[1]*3).reshape((rm.mask.shape[0]*3,rm.mask.shape[1]*3))

    ## test applying
    data1 = rm.applyMask(data1)
    if not (data1.sum()==39296070 and abs(data1.std()-3020.08934321)<0.001):
        error("applyMask 1 - data mismatch")

    data2 = rm.applyMask(data2.astype('int64'))
    if not (data2.sum()==3183264630 and abs(data2.std()-27182.1342973)<0.001):
        error("applyMask 2 - data mismatch")

    #rm.createRaster(output=result("regionMask_applyMask_1.tif"), data=data1, overwrite=True)
    #rm.createRaster(3, output=result("regionMask_applyMask_2.tif"), data=data2, overwrite=True)

def test_warp():
    ## setup
    rm_3035 = RegionMask.fromGeom(makePoint(6.20,50.75).Buffer(0.05))
    rm = RegionMask.fromGeom(makePoint(6.20,50.75).Buffer(0.05), srs=EPSG4326, pixelSize=0.0005)

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
    if not (isclose(warped_2.sum(),449935) and isclose(warped_2.std(),9.23575848858)):
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

def test_rasterize():
    ## setup
    rm = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH, pixelSize=0.001, srs=EPSG4326)

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

def test_indicateValues():
    ## Setup
    rm = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH, pixelSize=0.001, srs=EPSG4326)

    # Testing valueMin (with srs change)
    res1 = rm.indicateValues(CLC_RASTER_PATH, value=(20,None))
    if not (isclose(res1.sum(),30980.074218750) and isclose(res1.std(),0.350906521)): 
        error("indicateValues - valueMin")

    # Testing valueMax (with srs change)
    res2 = rm.indicateValues(CLC_RASTER_PATH, value=(None,24))

    if not (isclose(res2.sum(),82845.085938) and isclose(res2.std(),0.487606108)): 
        error("indicateValues - valueMax")

    # Testing valueEquals (with srs change)
    res3 = rm.indicateValues(CLC_RASTER_PATH, value=7)

    if not (isclose(res3.sum(),580.934937, 1e-4) and isclose(res3.std(),0.050088465, 1e-7)): 
        error("indicateValues - valueEquals")

    # Testing range
    res4 = rm.indicateValues(CLC_RASTER_PATH, value=(20,24))

    combi = np.logical_and(res1>0.5, res2>0.5)
    # Some pixels will not end up the same due to warping issues
    if not ( (res4>0.5) !=combi).sum()<res4.size*0.001:error("indicateValues - range")

def test_indicateFeatures():
  # setup
  rm = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH)

  # Its just a simple wrapper, test a simple case...
  res = rm.indicateFeatures(NATURA_PATH, where="SITECODE='DE5404303'")

  if not (isclose(res.sum(),649,1e-6) and isclose(res.std(),0.0603028809232,1e-6)): error("indicateSlopes - northMax")


if __name__=="__main__":
    test_init()
    test_fromMask()
    test_fromGeom()
    test_fromVectorFeature()
    test_fromVector()
    test_pixelSize()
    test_mask()
    test_geometry()
    test_createRaster()
    test_applyMask()
    test_warp()
    test_rasterize()
    test_indicateValues()
    test_indicateFeatures()
