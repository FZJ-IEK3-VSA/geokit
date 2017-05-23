from helpers import *

## Creation
def init_():
    # init structures
    rm1 = RegionMask(mask=MASK_DATA, extent=Extent(0,0,100,100,EPSG3035)) # mask + extent
    rm1 = RegionMask(mask=MASK_DATA, extent=Extent(0,0,100,100,EPSG3035)) # geom + pixelSize
    rm1 = RegionMask(mask=MASK_DATA, extent=Extent(0,0,100,100,EPSG3035)) # extent + geom + pixelSize
    rm1 = RegionMask(mask=MASK_DATA, extent=Extent(0,0,100,100,EPSG3035)) # mask + geom + pixelSize
    rm1 = RegionMask(mask=MASK_DATA, extent=Extent(0,0,100,100,EPSG3035)) # mask + extent + geom + pixelSize 

    if not (rm1.geometry.Area()-MASK_DATA.sum() < 0.001): error("init 1 - simple")



  # simple
  rm1 = RegionMask(MASK_DATA, Extent(0,0,100,100,EPSG3035))
  if not (rm1.geometry.Area()-MASK_DATA.sum() < 0.001): error("init 1 - simple")

  # from source, reading extent, and deault res/srs
  rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)

  if( rm2.extent.xXyY != (4035500.000000, 4069500.000000, 3048700.000000, 3101000.000000)):
    error("init 2 - fromVector - extent bounds")
  if not ( rm2.extent.srs.IsSame(EPSG3035) ): error("init 2 - fromVector - extent srs")
  if( rm2.mask.sum()!=70944 ): error("init 2 - fromVector - mask")

  # from source with defined res and extent
  rm3 = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelSize=0.001, extent=Extent(5.6,50.5,6.5,51.0))
  
  if( rm3.extent.xyXY != (5.6,50.5,6.5,51.0)):
    error("init 3 - fromVector - extent bounds")
  if not ( rm3.extent.srs.IsSame(EPSG4326) ): error("init 3 - fromVector - extent srs")
  if( rm3.mask.sum()!=90187 ): error("init 3 - fromVector - mask")

  # fromVector with a padded extent and defined srs
  rm4 = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelSize=0.001, srs=EPSG4326, padExtent=0.1)

  if( rm4.mask.sum()!=90296 ): error("init 4 - fromVector - mask")
  
  # fromGeom with wkt
  rm5 = RegionMask.fromGeom(POLY, wktSRS='latlon', pixelSize=1000)
  print(rm5.extent.xXyY)
  (4329800.0, 4770100.0, 835600.0, 1681100.0)
  if( rm5.extent.xXyY != (4329000.000000, 4771000.000000, 835000.000000, 1682000.000000)):
    error("init 5 - fromGeom - extent bounds")
  if not ( rm5.extent.srs.IsSame(EPSG3035) ): error("init 5 - fromGeom - extent srs")
  if( rm5.mask.sum()!=79274 ): error("init 5 - fromGeom - mask")

  # fromGeom with geometry
  dxy = 0.05
  rm6 = RegionMask.fromGeom(GEOM, pixelSize=dxy, srs=EPSG4326, padExtent=0.2)
  
  if( not rm6.extent == Extent(9.85,30.25,14.80,38.35)):
    error("init 6 - fromGeom - extent bounds")
  if not ( rm6.extent.srs.IsSame(EPSG4326) ): error("init 6 - fromGeom - extent srs")
  
  g = GEOM.Clone()
  g.TransformTo(EPSG4326)
  if not abs(rm6.mask.sum()*dxy*dxy-g.Area())/g.Area()<0.01: # check if total areas are close to one another
    error("init 6 - fromGeom - mask")
  
  # fromFourceFeature - ID select
  rm7 = RegionMask.fromVectorFeature(MULTI_FTR_SHAPE_PATH, select=1)

  if not (rm7.extent == Extent(4067700, 2866100, 4110200, 2954800, srs=EPSG3035)):
    error("init 7 - fromVectorFeature - extent bounds")
  if not (rm7.attributes["name"]=="dog"): error("init 7 - fromVectorFeature - attributes")

  ds = ogr.Open(MULTI_FTR_SHAPE_PATH)
  lyr = ds.GetLayer()
  ftr = lyr.GetFeature(1)
  g = ftr.GetGeometryRef().Clone()
  g.TransformTo(EPSG3035)
  if not abs(rm7.mask.sum()*100*100-g.Area())/g.Area()<0.01: # check if total areas are close to one another
    error("init 7 - fromVectorFeature - mask area")
  
  # fromVectorFeature - 'where' select
  rm8 = RegionMask.fromVectorFeature(MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelSize=0.01, select="name='monkey'")

  if not (rm8.extent == Extent(6.83, 49.52, 7.53, 49.94)):
    error("init 8 - fromVectorFeature - extent bounds")
  if not (rm8.attributes["id"]==3): error("init 8 - fromVectorFeature - attributes")

  ftr = lyr.GetFeature(3)
  g = ftr.GetGeometryRef().Clone()
  g.TransformTo(EPSG4326)
  if not isclose(rm8.mask.sum(),1948.0): 
    error("init 8 - fromVectorFeature - mask area")

def envelope_():
  # Setup
  rm1 = RegionMask(MASK_DATA, Extent(0,0,100,100,EPSG3035))

  ## envelope
  env =  rm1.envelope

  if not (env.Area() == 10000): error( "envelope - area")
  if not (env.GetSpatialReference().IsSame(EPSG3035)): error( "envelope - srs")

def pixelSize_():
  # Setup
  rm1 = RegionMask(MASK_DATA, Extent(0,0,100,100,EPSG3035))

  # pixelSize
  ps = rm1.pixelSize
  
  if not ps==1: error( "pixelSize")

def geometry_():
  # setup
  rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)

  ds = ogr.Open(AACHEN_SHAPE_PATH)
  lyr = ds.GetLayer()
  ftr = lyr.GetFeature(0)
  realGeom = ftr.GetGeometryRef().Clone()
  realGeom.TransformTo(EPSG3035)

  # check initial geom IS the geom
  if not abs(rm2.geometry.Area()-realGeom.Area())/realGeom.Area() < 0.001:
    error("geometry - reading from file")

  # destroy and recreate
  rm2.rebuildGeometry()
  if not abs( (rm2.geometry.Area()-realGeom.Area())/realGeom.Area() ) < 0.001:
    error("geometry - building geometry")

def tempFile_():
  # Setup
  rm1 = RegionMask(MASK_DATA, Extent(0,0,100,100,EPSG3035))

  ## Make a temp file
  path = rm1._tempFile()
  rm1.createRaster(data=rm1.mask, output=path)

  if not os.path.isfile(path): error("tempFile - creation")

  # try auto-deleting
  rm1 = None

  if(os.path.isfile(path)): error("tempFile - deletion")

def extractAttributes_():
  # setup
  rm_w = RegionMask.fromGeom(makeBox(7.080, 49.746, 7.199, 49.816, EPSG4326))

  # Extract Attributes
  rm_w.extractAttributes( MULTI_FTR_SHAPE_PATH, header="stats_")

  if not rm_w.attributes["stats_name"]=="monkey": error("extractAttributes")
  if not rm_w.attributes["stats_id"]==3: error("extractAttributes")

def createRaster_():
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
  
def applyMask_():
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

def warp_():
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

def rasterize_():
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

def indicateValues_():
  ## Setup
  rm = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH, pixelSize=0.001, srs=EPSG4326)

  # Testing valueMin (with srs change)
  res1 = rm.indicateValues(CLC_RASTER_PATH, value=(20,None))
  
  if not (isclose(res1.sum(),30980.074219, 1e-3) and isclose(res1.std(),0.350907), 1e-6): 
    error("indicateValues - valueMin")
  
  # Testing valueMax (with srs change)
  res2 = rm.indicateValues(CLC_RASTER_PATH, value=(None,24))
  
  if not (isclose(res2.sum(),82845.085938, 1e-4) and isclose(res2.std(),0.487606108, 1e-7)): 
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

def indicateAreas_():
  # setup
  rm = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH)

  # Its just a simple wrapper, test a simple case...
  res = rm.indicateAreas(NATURA_PATH, where="SITECODE='DE5404303'")

  if not (isclose(res.sum(),649,1e-6) and isclose(res.std(),0.0603028809232,1e-6)): error("indicateSlopes - northMax")

def draw_():
  print("MAKE 'draw' TESTER!!!!!!!!!!!!!")

if __name__=="__main__":
  init_()
  envelope_()
  pixelSize_()
  geometry_()
  tempFile_()
  extractAttributes_()
  createRaster_()
  applyMask_()
  warp_()
  rasterize_()
  indicateValues_()
  indicateAreas_()
  draw_()


  test_init()
  test_fromVector()
  test_fromGeom()
  test_fromVectorFeature()
  test_envelope()
  test_pixelSize()
  test_geometry()
  test_draw()
  test_rebuildGeometry()
  test__tempFile()
  test___del__()
  test_extractAttributes()
  test_createRaster()
  test_applyMask()
  test_warp()
  test_rasterize()
  test_indicateValues()
  test_indicateAreas()