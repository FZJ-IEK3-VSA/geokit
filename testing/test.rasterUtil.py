from helpers import *

## gdalType
def gdalType_():
  if( gdalType(bool) != "GDT_Byte" ): error("gdal type")
  if( gdalType("InT64") != "GDT_Int32" ): error("gdal type")
  if( gdalType("float32") != "GDT_Float32" ): error("gdal type")
  if( gdalType(NUMPY_FLOAT_ARRAY) != "GDT_Float64" ): error("gdal type")
  if( gdalType(NUMPY_FLOAT_ARRAY.dtype) != "GDT_Float64" ): error("gdal type")

## createRaster
def createRaster_():
  # setup
  data = (np.ones((1000,500))*np.arange(500)).astype("float32")
  outputFileName = result("util_raster1.tif")

  ######################
  # run and check funcs

  # mem creation
  memRas = createRaster( bounds=(10, 30, 15, 40), pixelHeight=0.01, pixelWidth=0.01, compress=True, srs=EPSG4326, dtype="Float32", noDataValue=-9999, fillValue=12.34)

  if(memRas is None): error("creating raster in memory")

  # Disk creation
  createRaster( bounds=(10, 30, 15, 40), output=outputFileName, pixelHeight=0.01, pixelWidth=0.01, compress=True, srs=EPSG4326, noDataValue=100, data=data, overwrite=True)

  ds = gdal.Open(outputFileName)
  bd = ds.GetRasterBand(1)
  srs = osr.SpatialReference()
  srs.ImportFromWkt(ds.GetProjection())
  if(not srs.IsSame(EPSG4326)): error("Creating raster on disk - srs mismatch")

  arr = bd.ReadAsArray()
  if(arr.sum() != data.sum()): error("Creating raster on disk - data mismatch")

## Describe Raster
def rasterInfo_():
  info = rasterInfo(CLC_RASTER_PATH)

  if(not (info.xMin, info.yMin, info.xMax, info.yMax) == (4012100.0, 3031800.0, 4094600.0, 3111000.0)):
    error("rasterInfo - min/max values")
  if not (info.dx == 100 and info.dy == 100) : error("decribeRaster - dx/dy")
  if not (info.bounds == (4012100.0, 3031800.0, 4094600.0, 3111000.0)) : error("decribeRaster - bounds")
  if not (info.dtype == gdal.GDT_Byte) : error("decribeRaster - datatype")
  if not (info.srs.IsSame(EPSG3035)) : error("decribeRaster - srs")
  if not (info.noData == 0) : error("decribeRaster - noData")
  if not (info.flipY == True) : error("decribeRaster - flipY")
      
def rasterValues_():
  print("MAKE rasterValues TESTER!!!!!!!!!")

def rasterValue_():
  print("MAKE rasterValue TESTER!!!!!!!!!")

def rasterMatrix_():
  print("MAKE rasterMatrix TESTER!!!!!!!!!")

def rasterGradient_():
  print("MAKE rasterGradient TESTER!!!!!!!!!")
"""
def indicateSlopes_():
  ## Setup
  rm = RegionMask.fromVectorFeature(AACHEN_SHAPE_PATH, pixelSize=0.001, srs=EPSG4326)

  # Testing totalMin (with srs change)
  res1 = rm.indicateSlopes(SINGLE_HILL_PATH, totalMin=5)
  
  if not (isclose(res1.sum(),28128.0,1e-6) and isclose(res1.std(),0.344903,1e-6)): error("indicateSlopes - totalMin")

  # Testing totalMax (with srs change)
  res2 = rm.indicateSlopes(SINGLE_HILL_PATH, totalMax=20)
  
  if not (isclose(res2.sum(),72235.0,1e-6) and isclose(res2.std(),0.478332,1e-6)): error("indicateSlopes - totalMax")

  # Testing totalRange (with srs change)
  res3 = rm.indicateSlopes(SINGLE_HILL_PATH, totalMax=20, totalMin=5)

  combi = np.logical_and(res1, res2)
  # Some pixels will not end up the same due to warping issues
  if not (res3!=combi).sum()<res3.size*0.001:error("indicateSlopes - total range")

  # Testing totalMin (with srs change)
  res4 = rm.indicateSlopes(SINGLE_HILL_PATH, northMin=-2)
  
  if not (isclose(res4.sum(),75517.0,1e-6) and isclose(res4.std(),0.48294,1e-6)): error("indicateSlopes - northMin")

  # Testing northMax (with srs change)
  res5 = rm.indicateSlopes(SINGLE_HILL_PATH, northMax=2)

  if not (isclose(res5.sum(),74509.0,1e-6) and isclose(res5.std(),0.481587,1e-6)): error("indicateSlopes - northMax")

  # Testing northRange (with srs change)
  res6 = rm.indicateSlopes(SINGLE_HILL_PATH, northMax=2, northMin=-2)

  combi = np.logical_and(res4, res5)
  # Some pixels will not end up the same due to warping issues
  if not (res6!=combi).sum()<res6.size*0.001:error("indicateSlopes - north range")
"""

def rasterMutate_():
  # Setup
  ext = Extent.fromVector(AACHEN_SHAPE_PATH)
  ext_3035 = ext.castTo('europe_m').fit(100)
  def isOdd(mat): return np.mod(mat,2)
  
  source = gdal.Open(CLC_RASTER_PATH)
  sourceInfo = rasterInfo(source)

  ## Process Raster with no processor or extent
  res1 = rasterMutate(source)#, overwrite=True, output=result("algorithms_rasterMutate_1.tif"))

  info1 = rasterInfo(res1)
  if not info1.srs.IsSame(sourceInfo.srs): error("rasterMutate 1 - srs")
  if not info1.bounds == sourceInfo.bounds: error("rasterMutate 1 - bounds")

  ## rasterMutate with a simple processor
  res2 = rasterMutate(source, extent=ext, processor=isOdd)#, overwrite=True, output=result("algorithms_rasterMutate_2.tif"))

  info2 = rasterInfo(res2)
  if not info2.srs.IsSame(sourceInfo.srs): error("rasterMutate 2 - srs")
  if not isclose(info2.xMin, ext_3035.xMin): error("rasterMutate 2 - bounds")
  if not isclose(info2.xMax, ext_3035.xMax): error("rasterMutate 2 - bounds")
  if not isclose(info2.yMin, ext_3035.yMin): error("rasterMutate 2 - bounds")
  if not isclose(info2.yMax, ext_3035.yMax): error("rasterMutate 2 - bounds")

  band2 = res2.GetRasterBand(1)
  arr2 = band2.ReadAsArray()

  if not (arr2.sum()==45262): error("rasterMutate 2 - data")

  ## Process Raster with a simple processor (flip check)
  res2f = rasterMutate(CLC_FLIPCHECK_PATH, extent=ext, processor=isOdd)#, overwrite=True, output=result("algorithms_rasterMutate_2f.tif"))

  info2f = rasterInfo(res2f)
  if not info2f.srs.IsSame(sourceInfo.srs): error("rasterMutate 2f - srs")
  if not isclose(info2f.xMin, ext_3035.xMin): error("rasterMutate 2f - bounds")
  if not isclose(info2f.xMax, ext_3035.xMax): error("rasterMutate 2f - bounds")
  if not isclose(info2f.yMin, ext_3035.yMin): error("rasterMutate 2f - bounds")
  if not isclose(info2f.yMax, ext_3035.yMax): error("rasterMutate 2f - bounds")

  band2f = res2f.GetRasterBand(1)
  arr2f = band2f.ReadAsArray()

  if not (arr2f.sum()==45262): error("rasterMutate 2f - data")

  ## Check flipped data
  if not (arr2f==arr2).all(): error("rasterMutate 2f - flipping error!")

if __name__=="__main__":
    gdalType_()
    createRaster_()
    rasterInfo_()
    rasterValues_()
    rasterValue_()
    rasterMatrix_()
    rasterGradient_()
    rasterMutate_()