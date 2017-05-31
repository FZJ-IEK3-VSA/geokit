from helpers import *
from geokit.util import *

from geokit import Extent

# test creation
def test_init():
    # basic
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    if(ex1.xyXY != (-5, -4, 3, 10)): error("Simple creation")
    ex1b = Extent((3, -4, -5, 10), srs=EPSG4326)
    if(ex1b.xyXY != (-5, -4, 3, 10)): error("Simple creation b")

    # from source
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )
    if(ex2.xyXY != (6.212409755390094, 48.864894076418935, 7.782393932571588, 49.932593106005655)): error("From vector")

    # from source
    ex3 = Extent.fromRaster( source=AACHEN_ELIGIBILITY_RASTER )
    if(ex3.xyXY != (5.974, 50.494, 6.42, 50.951)): error("From raster")
    if( not ex3.srs.IsSame(EPSG4326)): error("From raster")

    # from wkt
    ex4 = Extent.fromGeom( POLY, srs=EPSG4326)
    if(ex4.xXyY != (10.1, 14.6, 30.5, 38.1)): error("From wkt")

def test_eq():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

    # Equality
    if( (ex2 == ex1) != False): error( "equality 1")
    if( (ex2 == ex2) != True):  error( "equality 2")

## Add padding
def test_pad():
    #setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)

    # do test
    ex_pad = ex1.pad(1)
    
    # check
    if(ex_pad.xyXY != (-6, -5, 4, 11)): error ("padded")

def test_fit():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

    ## Fiting
    ex_fit1 = ex1.fit(2, float)
    if(ex_fit1.xyXY != (-6.0, -4.0, 4.0, 10.0)): error ("fit 1")

    ex_fit2 = ex2.fit(0.01)
    if(ex_fit2.xyXY != (6.21, 48.86, 7.79, 49.94)): error ("fit 2")

def test_castTo():
    # setup
    ex3 = Extent.fromGeom( POLY, srs=EPSG4326)

    ## Projecting
    ex_cast = ex3.castTo(EPSG3035)
    if(ex_cast.fit(1).xyXY != (4329833, 835682, 4770009, 1681039)): error("Casting")

def test_inSourceExtent():
    # setup
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

    ## Test if in source
    if( ex2.inSourceExtent(LUX_SHAPE_PATH) != True): error("In source extent 1")
    if( ex2.inSourceExtent(AACHEN_SHAPE_PATH) != False): error("In source extent 2")

def test_contains():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    
    ## Test for contains
    if( ex1.contains(Extent(-5, -4, 3, 12)) != False): error("Contains 1")
    if( ex1.contains(Extent(-5, -3.3333, 2.0002, 8)) != True): error("Contains 2")
    if( ex1.contains(Extent(-2.0, -3.5, 1.0, 8.5), 0.5) != True): error("Contains 3")
    if( ex1.contains(Extent(-2.0, -3.25, 1.0, 8.25), 0.5) != False): error("Contains 4")

def test_filterSources():
    sources = source("*.shp")
    ex = Extent.fromVector( source=AACHEN_SHAPE_PATH )

    goodSources = list(ex.filterSources(sources))
    if not AACHEN_ZONES in goodSources: error("filterSources - inclusion")
    if not AACHEN_POINTS in goodSources: error("filterSources - inclusion")
    if BOXES in goodSources: error("filterSources - exclusion")
    if LUX_SHAPE_PATH in goodSources: error("filterSources - exclusion")
    if LUX_LINES_PATH in goodSources: error("filterSources - exclusion")

def test_findWithin():
    bigExt = Extent.fromRaster(CLC_RASTER_PATH)
    smallExt = Extent(4050200,3076800,4055400,3080300, srs=EPSG3035)

    # do regular
    w1 = bigExt.findWithin(smallExt, res=100)
    if not w1.xStart == 381: error("findWithin - xStart")
    if not w1.yStart == 307: error("findWithin - yStart")
    if not w1.xWin == 52: error("findWithin - xWin")
    if not w1.yWin == 35: error("findWithin - yWin")

    # do flipped (only yStart should change)
    w2 = bigExt.findWithin(smallExt, res=100, yAtTop=False)
    if not w2.xStart == 381: error("findWithin - flipped - xStart")
    if not w2.yStart == 450: error("findWithin - flipped - yStart")
    if not w2.xWin == 52: error("findWithin - flipped - xWin")
    if not w2.yWin == 35: error("findWithin - flipped - yWin")

def test_clipRaster():
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

def test_extractMatrix():
  # setup
  ex = Extent(6.022,50.477,6.189,50.575).castTo(EPSG3035).fit(100)
  
  # extract
  mat1 = ex.extractMatrix(CLC_RASTER_PATH)
  if mat1.sum()!=392284: error("extractMatrix - matrix")
  
  # extract
  mat2 = ex.extractMatrix(CLC_FLIPCHECK_PATH)
  if mat2.sum()!=392284: error("extractMatrix - matrix")
  
  # Make sure matricies are the same
  if not (mat1==mat2).all(): error("extractMatrix - fliping error")

  # test fail
  try:
      p = ex.shift(dx=1).extractMatrix(CLC_RASTER_PATH)
      error("extractMatrix - fail test")
  except GeoKitError as e:
      pass
  else:
      error("extractMatrix - fail test")



if __name__ == "__main__":
    test_init()
    test_eq()
    test_pad()
    test_fit()
    test_castTo()
    test_inSourceExtent()                
    test_contains()
    test_filterSources()
    test_findWithin()
    test_clipRaster()
    test_extractMatrix()
