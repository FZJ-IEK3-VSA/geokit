from helpers import *

# test creation
def init_():
  # basic
  ex1 = Extent(3, -4, -5, 10, EPSG4326)
  if(ex1.xyXY != (-5, -4, 3, 10)): error("Simple creation")

  # from source
  ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )
  if(ex2.xyXY != (6.212409755390094, 48.864894076418935, 7.782393932571588, 49.932593106005655)): error("From vector")

  # from source
  ex3 = Extent.fromRaster( source=AACHEN_ELIGIBILITY_RASTER )
  if(ex3.xyXY != (5.974, 50.494, 6.42, 50.951)): error("From raster")
  if( not ex3.srs.IsSame(SRSCOMMON.latlon)): error("From raster")

  # from wkt
  ex4 = Extent.fromGeom( POLY, srs=EPSG4326)
  if(ex4.xXyY != (10.1, 14.6, 30.5, 38.1)): error("From wkt")

def eq_():
  # setup
  ex1 = Extent(3, -4, -5, 10, EPSG4326)
  ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

  # Equality
  if( (ex2 == ex1) != False): error( "equality 1")
  if( (ex2 == ex2) != True):  error( "equality 2")

## Add padding
def pad_():
  #setup
  ex1 = Extent(3, -4, -5, 10, EPSG4326)

  # do test
  ex_pad = ex1.pad(1)
  
  # check
  if(ex_pad.xyXY != (-6, -5, 4, 11)): error ("padded")

def fit_():
  # setup
  ex1 = Extent(3, -4, -5, 10, EPSG4326)
  ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

  ## Fiting
  ex_fit1 = ex1.fit(2, float)
  if(ex_fit1.xyXY != (-6.0, -4.0, 4.0, 10.0)): error ("fit 1")

  ex_fit2 = ex2.fit(0.01)
  if(ex_fit2.xyXY != (6.21, 48.86, 7.79, 49.94)): error ("fit 2")

def castTo_():
  # setup
  ex3 = Extent.fromGeom( POLY, srs=EPSG4326)

  ## Projecting
  ex_cast = ex3.castTo(EPSG3035)
  if(ex_cast.fit(1).xyXY != (4329833, 835682, 4770009, 1681039)): error("Casting")

def inSourceExtent_():                   
  # setup
  ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

  ## Test if in source
  if( ex2.inSourceExtent(LUX_SHAPE_PATH) != True): error("In source extent 1")
  if( ex2.inSourceExtent(AACHEN_SHAPE_PATH) != False): error("In source extent 2")

def contains_():
  # setup
  ex1 = Extent(3, -4, -5, 10, EPSG4326)
  
  ## Test for contains
  if( ex1.contains(Extent(-5, -4, 3, 12)) != False): error("Contains 1")
  if( ex1.contains(Extent(-5, -3.3333, 2.0002, 8)) != True): error("Contains 2")
  if( ex1.contains(Extent(-2.0, -3.5, 1.0, 8.5), 0.5) != True): error("Contains 3")
  if( ex1.contains(Extent(-2.0, -3.25, 1.0, 8.25), 0.5) != False): error("Contains 4")

def filterSourceDir_():
  print( "MAKE 'filterSourceDir' TESTER!!!!!!!!!")

def findWithin_():
  print( "MAKE 'findWithin' TESTER!!!!!!!!!")

def clipRaster_():
  # setup
  ex1 = Extent(6.022,50.477,6.189,50.575)
  
  ## Clip raster around extent
  ds = ex1.clipRaster( CLC_RASTER_PATH )

  bd = ds.GetRasterBand(1)
  arr1 = bd.ReadAsArray()
  if (arr1.sum()!=392284): error("clipRaster 1")

  ## Make sure we get same result with a 'flipped' raster
  ds = ex1.clipRaster( CLC_FLIPCHECK_PATH )

  bd = ds.GetRasterBand(1)
  arr2 = bd.ReadAsArray()
  if (arr2.sum()!=392284): error("clipRaster 2")

  ## Make sure the two matricies are the same
  # NOTE! arr2 does NOT need flipping here since clipRaster is creating a NEW raster which by default is in the flipped-y orientation
  if not (arr1==arr2).all(): error("clipRaster - raster reading")

def extractFromRaster_():
  # setup
  ex = Extent(6.022,50.477,6.189,50.575).castTo(EPSG3035).fit(100)
  
  # extract
  mat1 = ex.extractFromRaster(CLC_RASTER_PATH)
  if mat1.sum()!=392284: error("extractFromRaster - matrix")
  
  # extract
  mat2 = ex.extractFromRaster(CLC_FLIPCHECK_PATH)
  if mat2.sum()!=392284: error("extractFromRaster - matrix")
  
  # Make sure matricies are the same
  if not (mat1==mat2).all(): error("extractFromRaster - fliping error")

  # test fail
  try:
    p = ex.shift(dx=50).extractFromRaster(CLC_RASTER_PATH)
    error("extractFromRaster - fail test")
  except RuntimeError as e:
    pass
  else:
    error("extractFromRaster - fail test")



if __name__ == "__main__":
  init_()
  eq_()
  pad_()
  fit_()
  castTo_()
  inSourceExtent_()
  contains_()
  filterSourceDir_()
  findWithin_()
  clipRaster_()
  extractFromRaster_()
