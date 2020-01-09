from helpers import *
from geokit.util import *
from geokit.raster import *
from geokit.vector import *
from geokit import Extent, LocationSet


def test_Extent___init__():
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

    print("Extent___init__ passed")

def test_Extent_from_xXyY():

    ex1 = Extent.from_xXyY( (1,2,3,4) )
    compare(ex1.xMin, 1)
    compare(ex1.xMax, 2)
    compare(ex1.yMin, 3)
    compare(ex1.yMax, 4)

    print("Extent_from_xXyY passed")

def test_Extent_fromGeom():
    ex1 = Extent.fromGeom(GEOM)
    compare(ex1.xMin, 10.10000)
    compare(ex1.xMax, 14.60000)
    compare(ex1.yMin, 30.50000)
    compare(ex1.yMax, 38.10000)

    print("Extent_fromGeom passed")

def test_Extent_fromVector():

    ex1 = Extent.fromVector(AACHEN_POINTS)
    
    compare(ex1.xMin, 6.03745)
    compare(ex1.xMax, 6.33091)
    compare(ex1.yMin, 50.5367)
    compare(ex1.yMax, 50.9227)

    print("Extent_fromVector passed")

def test_Extent_fromRaster():
    ex1 = Extent.fromRaster(CLC_RASTER_PATH)
    compare(ex1.xMin, 4012100)
    compare(ex1.yMin, 3031800)
    compare(ex1.xMax, 4094600)
    compare(ex1.yMax, 3111000)

    print("Extent_fromRaster passed")

def test_Extent_fromLocationSet():
    ls = LocationSet(pointsInAachen3035)

    ex1 = Extent.fromLocationSet(ls)
    
    compare(ex1.xMin, 4039553.19006)
    compare(ex1.yMin, 3052769.53854)
    compare(ex1.xMax, 4065568.41553)
    compare(ex1.yMax, 3087947.74366)

    print("Extent_fromLocationSet passed")

def test_Extent_load():
    # Explicit
    ex1 = Extent.load( (1,2,3,4), srs=4326 )
    compare(ex1.xMin, 1)
    compare(ex1.xMax, 3)
    compare(ex1.yMin, 2)
    compare(ex1.yMax, 4)
    if not ex1.srs.IsSame(EPSG4326): error("srs")

    # Geometry
    ex2 = Extent.load(GEOM)
    compare(ex2.xMin, 10.10000)
    compare(ex2.xMax, 14.60000)
    compare(ex2.yMin, 30.50000)
    compare(ex2.yMax, 38.10000)

    if not ex2.srs.IsSame(EPSG4326): error("srs")

    # vector
    ex3 = Extent.load(AACHEN_POINTS)
    
    compare(ex3.xMin, 6.03745)
    compare(ex3.xMax, 6.33091)
    compare(ex3.yMin, 50.5367)
    compare(ex3.yMax, 50.9227)
    if not ex3.srs.IsSame(EPSG4326): error("srs")

    # Raster
    ex4 = Extent.load(CLC_RASTER_PATH)
    compare(ex4.xMin, 4012100)
    compare(ex4.yMin, 3031800)
    compare(ex4.xMax, 4094600)
    compare(ex4.yMax, 3111000)
    if not ex4.srs.IsSame(EPSG3035): error("srs")

    # Location set
    ls = LocationSet(pointsInAachen3035, srs=3035)

    ex5 = Extent.load(ls)
    
    compare(ex5.xMin, 6.02141)
    compare(ex5.yMin, 50.51939)
    compare(ex5.xMax, 6.371634)
    compare(ex5.yMax, 50.846025)
    if not ex5.srs.IsSame(EPSG4326): error("srs")

    print("Extent_load passed")

def test_Extent_xyXY():
    print("Extent_xyXY is trivial")

def test_Extent_xXyY():
    print("Extent_xXyY is trivial")

def test_Extent_ylim():
    print("Extent_ylim is trivial")

def test_Extent_xlim():
    print("Extent_xlim is trivial")

def test_Extent_box():
    print("Extent_box is trivial")

def test_Extent___eq__():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

    # Equality
    if( (ex2 == ex1) != False): error( "equality 1")
    if( (ex2 == ex2) != True):  error( "equality 2")
    print("Extent___eq__ passed")

def test_Extent_pad():
    #setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)

    # do test
    ex_pad = ex1.pad(1)
    
    # check
    if(ex_pad.xyXY != (-6, -5, 4, 11)): error ("padded")

    print("Extent_pad passed")

def test_Extent_shift():
    ex = Extent.load( (1,2,3,4), srs=4326 )
    ex1 = ex.shift(-1,2)
    compare(ex1.xMin, 0)
    compare(ex1.xMax, 2)
    compare(ex1.yMin, 4)
    compare(ex1.yMax, 6)

    print("Extent_shift passed")

def test_Extent_fitsResolution():
    ex = Extent(-6, -5, 4, 11)
    if not ex.fitsResolution(0.5): error("good fit")
    if not ex.fitsResolution(2): error("bad fit")
    print("Extent_fitsResolution passed")

def test_Extent_fit():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

    ## Fiting
    ex_fit1 = ex1.fit(2, float)
    if(ex_fit1.xyXY != (-6.0, -4.0, 4.0, 10.0)): error ("fit 1")

    ex_fit2 = ex2.fit(0.01)
    if(ex_fit2.xyXY != (6.21, 48.86, 7.79, 49.94)): error ("fit 2")

    print("Extent_fit passed")

def test_Extent_corners():
    print("Extent_corners is trivial")

def test_Extent_castTo():
    # setup
    ex3 = Extent.fromGeom( GEOM )

    ## Projecting
    ex_cast = ex3.castTo(EPSG3035)
    if(ex_cast.fit(1).xyXY != (4329833, 835682, 4770009, 1681039)): error("Casting")

    print("Extent_castTo passed")

def test_Extent_inSourceExtent():
    # setup
    ex2 = Extent.fromVector( source=MULTI_FTR_SHAPE_PATH )

    ## Test if in source
    if( ex2.inSourceExtent(LUX_SHAPE_PATH) != True): error("In source extent 1")
    if( ex2.inSourceExtent(AACHEN_SHAPE_PATH) != False): error("In source extent 2")

    print("Extent_inSourceExtent passed")

def test_Extent_filterSources():
    sources = source("*.shp")
    ex = Extent.fromVector( source=AACHEN_SHAPE_PATH )

    goodSources = list(ex.filterSources(sources))
    if not AACHEN_ZONES in goodSources: error("filterSources - inclusion")
    if not AACHEN_POINTS in goodSources: error("filterSources - inclusion")
    if BOXES in goodSources: error("filterSources - exclusion")
    if LUX_SHAPE_PATH in goodSources: error("filterSources - exclusion")
    if LUX_LINES_PATH in goodSources: error("filterSources - exclusion")

    print("Extent_filterSources passed")

def test_Extent_containsLoc():
    ex1 = Extent(3, 4, 5, 10, srs=EPSG4326)

    if not ex1.containsLoc( (4,8) ): error("containsLoc - good single")
    if ex1.containsLoc( (7,0) ): error("containsLoc - bad single")
    
    s = ex1.containsLoc( [(4,8),(7,0),(7,1),(3,4) ])
    compare(s.sum(), 2)

    print("Extent_containsLoc passed")

def test_Extent_contains():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    
    ## Test for contains
    if( ex1.contains(Extent(-5, -4, 3, 12)) != False): error("Contains 1")
    if( ex1.contains(Extent(-5, -3.3333, 2.0002, 8)) != True): error("Contains 2")
    if( ex1.contains(Extent(-2.0, -3.5, 1.0, 8.5), 0.5) != True): error("Contains 3")
    if( ex1.contains(Extent(-2.0, -3.25, 1.0, 8.25), 0.5) != False): error("Contains 4")

    print("Extent_contains passed")

def test_Extent_findWithin():
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
    

    print("Extent_findWithin passed")

def test_Extent_createRaster():
    ex = Extent.fromRaster(CLC_RASTER_PATH)

    # Test failure on bad resolution
    try:
        r = ex.createRaster( pixelHeight=200, pixelWidth=200, fill=2 )
        error("Did not fail on bad resolution")
    except GeoKitExtentError as e:
        if not str(e)=="The given resolution does not fit to the Extent boundaries":
            error("RegionMask bad message")

    # Test successful
    r = ex.createRaster( pixelHeight=100, pixelWidth=100, fill=2 )
    compare(extractMatrix(r).mean(), 2)

    print("Extent_createRaster passed")

def test_Extent_extractMatrix():

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

    # test fail since the given extent does not fit in the grid of the raster
    try:
        p = ex.shift(dx=1).extractMatrix(CLC_RASTER_PATH)
        print(p)
        error("extractMatrix - fail test")
    except GeoKitError as e:
        pass
    else:
        error("extractMatrix - fail test")

    print("Extent_extractMatrix passed")

def test_Extent_warp():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(3035).fit(200)

    # Change resolution to disk
    d = ex.warp( CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200, output=result("extent_warp1.tif") )
    v1 = extractMatrix(d)
    compare( v1.mean(), 17.18144279, "warp - value" )

    # change resolution to memory
    d = ex.warp( CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200 )
    v2 = extractMatrix(d)
    compare( (v1-v2).mean(), 0)

    # Do a cutline from disk
    d = ex.warp( CLC_RASTER_PATH, pixelHeight=100, pixelWidth=100, cutline=AACHEN_SHAPE_PATH, output=result("extent_warp3.tif"), noData=99 )
    v3 = extractMatrix(d)
    compare(v3.mean(), 66.02815723) 
    compare(v3[0,0], 99, "warp -noData")

    print("Extent_warp passed")

def test_Extent_rasterize():

    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(3035).fit(250)

    # Simple vectorization to file
    r = ex.rasterize(source=AACHEN_ZONES, pixelWidth=250, pixelHeight=250, output=result("extent_rasterized1.tif"))
    mat1 = extractMatrix(r)
    compare(mat1.mean(), 0.22881653, "rasterization - simple")

    # Simple vectorization to mem
    r = ex.rasterize(source=AACHEN_ZONES, pixelWidth=250, pixelHeight=250, )
    mat2 = extractMatrix(r)
    compare( (mat2-mat1).mean(), 0, "rasterization - memory")

    # Write attribute values to disc
    r = ex.rasterize(source=AACHEN_ZONES, value="YEAR", pixelWidth=250, pixelHeight=250, output=result("extent_rasterized3.tif"), noData=-1)
    mat = extractMatrix(r, autocorrect=True)
    compare(np.isnan(mat).sum(), 22025, "rasterization - nan values")
    compare(np.nanmean(mat), 1996.36771232, "rasterization - simple")

    print("Extent_rasterize passed")

def test_Extent_extractFeatures():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH)

    # Test loading as a dataframe
    vi = ex.extractFeatures(AACHEN_ZONES, asPandas=True)
    if vi.shape[0]!=101: error("extractFeatures - shape mismatch") 

    print("Extent_extractFeatures passed")

def test_Extent_mutateVector():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(4326)

    # Test simple clipping
    vi = ex.mutateVector(AACHEN_ZONES, matchContext=False, output=result("extent_mutateVector1.shp")) 
    info = vectorInfo(vi)
    compare(info.count, 101)
    if not info.srs.IsSame(EPSG3035): error("projection retention")

    # clip and convert
    vi = ex.mutateVector(AACHEN_ZONES, matchContext=True,) 
    info = vectorInfo(vi)
    compare(info.count, 101)
    if not info.srs.IsSame(EPSG4326): error("projection retention")

    # Simple operation
    docenter = lambda x: {'geom': x.geom.Centroid()}
    vi = ex.mutateVector(AACHEN_ZONES, matchContext=True, keepAttributes=False, processor=docenter, output=result("extent_mutateVector3.shp")) 
    info = vectorInfo(vi)
    compare(len(info.attributes), 1)
    compare(info.count, 101)
    if not info.srs.IsSame(EPSG4326): error("projection retention")

    print("Extent_mutateVector passed")

def test_Extent_mutateRaster():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(4326).fit(0.001)

    # test a simple clip
    r = ex.mutateRaster(CLC_RASTER_PATH, output=result("extent_mutateRaster1.tif"))
    mat = extractMatrix(r)
    compare(mat.mean(),17.14654805)

    # test a clip and warp
    r = ex.mutateRaster(CLC_RASTER_PATH, pixelHeight=0.001, pixelWidth=0.001, matchContext=True, output=result("extent_mutateRaster2.tif"), resampleAlg='near')
    mat2 = extractMatrix(r)
    compare(mat2.mean(),17.14768769)

    # simple processor
    @KernelProcessor(1, edgeValue=-9999)
    def max_3x3( mat ):
        goodValues = mat[mat!=-9999].flatten()
        return goodValues.max()

    r = ex.mutateRaster(CLC_RASTER_PATH, processor=max_3x3, output=result("extent_mutateRaster3.tif"))
    mat3 = extractMatrix(r)
    compare(mat3.mean(),19.27040301)

    print("Extent_mutateRaster passed")


if __name__ == "__main__":
    test_Extent___init__()
    test_Extent_from_xXyY()
    test_Extent_fromGeom()
    test_Extent_fromVector()
    test_Extent_fromRaster()
    test_Extent_fromLocationSet()
    test_Extent_load()
    test_Extent_xyXY()
    test_Extent_xXyY()
    test_Extent_ylim()
    test_Extent_xlim()
    test_Extent_box()
    test_Extent___eq__()
    test_Extent_pad()
    test_Extent_shift()
    test_Extent_fitsResolution()
    test_Extent_fit()
    test_Extent_corners()
    test_Extent_castTo()
    test_Extent_inSourceExtent()
    test_Extent_filterSources()
    test_Extent_containsLoc()
    test_Extent_contains()
    test_Extent_findWithin()
    test_Extent_createRaster()
    test_Extent_extractMatrix()
    test_Extent_warp()
    test_Extent_rasterize()
    test_Extent_extractFeatures()
    test_Extent_mutateVector()
    test_Extent_mutateRaster()