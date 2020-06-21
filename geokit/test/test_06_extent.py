from .helpers import *
from geokit import srs, Extent, LocationSet, util, raster, vector, error, _test_data_


def test_Extent___init__():
    # basic
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    assert (ex1.xyXY == (-5, -4, 3, 10))
    ex1b = Extent((3, -4, -5, 10), srs=EPSG4326)
    assert (ex1b.xyXY == (-5, -4, 3, 10))

    # from source
    ex2 = Extent.fromVector(source=MULTI_FTR_SHAPE_PATH)
    assert (ex2.xyXY == (6.212409755390094, 48.864894076418935,
                         7.782393932571588, 49.932593106005655))

    # from source
    ex3 = Extent.fromRaster(source=AACHEN_ELIGIBILITY_RASTER)
    assert (ex3.xyXY == (5.974, 50.494, 6.42, 50.951))
    assert ex3.srs.IsSame(EPSG4326)


def test_Extent_from_xXyY():
    ex1 = Extent.from_xXyY((1, 2, 3, 4))
    assert np.isclose(ex1.xMin, 1)
    assert np.isclose(ex1.xMax, 2)
    assert np.isclose(ex1.yMin, 3)
    assert np.isclose(ex1.yMax, 4)


def test_Extent_fromGeom():
    ex1 = Extent.fromGeom(GEOM)
    assert np.isclose(ex1.xMin, 10.10000)
    assert np.isclose(ex1.xMax, 14.60000)
    assert np.isclose(ex1.yMin, 30.50000)
    assert np.isclose(ex1.yMax, 38.10000)


def test_Extent_fromTile():
    # fun func
    ext = Extent.fromTile(xi=1062, yi=693, zoom=11)

    assert np.isclose(ext.xMin, 743579.411158)
    assert np.isclose(ext.xMax, 763147.290399)
    assert np.isclose(ext.yMin, 6457400.149532)
    assert np.isclose(ext.yMax, 6476968.028773)


def test_Extent_fromTileAt():
    # fun func
    tile = Extent.fromTileAt(6.083, 50.775, zoom=8, srs=EPSG4326)

    assert np.isclose(tile.xMin, 626172.135712)
    assert np.isclose(tile.xMax, 782715.169640)
    assert np.isclose(tile.yMin, 6574807.424978)
    assert np.isclose(tile.yMax, 6731350.458906)

    tile = Extent.fromTileAt(
        x=4101103, y=2978620, zoom=8, srs=EPSG3035)

    assert np.isclose(tile.xMin, 626172.135712164)
    assert np.isclose(tile.xMax, 782715.169640205)
    assert np.isclose(tile.yMin, 6418264.3910496775)
    assert np.isclose(tile.yMax, 6574807.4249777235)

def test_Extent_fromVector():

    ex1 = Extent.fromVector(AACHEN_POINTS)

    assert np.isclose(ex1.xMin, 6.03745)
    assert np.isclose(ex1.xMax, 6.33091)
    assert np.isclose(ex1.yMin, 50.5367)
    assert np.isclose(ex1.yMax, 50.9227)


def test_Extent_fromRaster():
    ex1 = Extent.fromRaster(CLC_RASTER_PATH)
    assert np.isclose(ex1.xMin, 4012100)
    assert np.isclose(ex1.yMin, 3031800)
    assert np.isclose(ex1.xMax, 4094600)
    assert np.isclose(ex1.yMax, 3111000)


def test_Extent_fromLocationSet():
    ls = LocationSet(pointsInAachen3035)

    ex1 = Extent.fromLocationSet(ls)

    assert np.isclose(ex1.xMin, 4039553.19006)
    assert np.isclose(ex1.yMin, 3052769.53854)
    assert np.isclose(ex1.xMax, 4065568.41553)
    assert np.isclose(ex1.yMax, 3087947.74366)


def test_Extent_fromWKT():
    wkt = 'POLYGON ((4031300 2684000 0,4672600 2684000 0,4672600 3551300 0,4031300 3551300 0,4031300 2684000 0))|PROJCS["ETRS89 / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","3035"]]'
    ex1 = Extent.fromWKT(wkt)

    assert np.isclose(ex1.xMin, 4031300.0)
    assert np.isclose(ex1.xMax, 4672600.0)
    assert np.isclose(ex1.yMin, 2684000.0)
    assert np.isclose(ex1.yMax, 3551300.0)
    assert ex1.srs.IsSame(srs.EPSG3035)


def test_Extent_load():
    # Explicit
    ex1 = Extent.load((1, 2, 3, 4), srs=4326)
    assert np.isclose(ex1.xMin, 1)
    assert np.isclose(ex1.xMax, 3)
    assert np.isclose(ex1.yMin, 2)
    assert np.isclose(ex1.yMax, 4)
    assert ex1.srs.IsSame(EPSG4326)

    # Geometry
    ex2 = Extent.load(GEOM)
    assert np.isclose(ex2.xMin, 10.10000)
    assert np.isclose(ex2.xMax, 14.60000)
    assert np.isclose(ex2.yMin, 30.50000)
    assert np.isclose(ex2.yMax, 38.10000)
    assert ex2.srs.IsSame(EPSG4326)

    # vector
    ex3 = Extent.load(AACHEN_POINTS)

    assert np.isclose(ex3.xMin, 6.03745)
    assert np.isclose(ex3.xMax, 6.33091)
    assert np.isclose(ex3.yMin, 50.5367)
    assert np.isclose(ex3.yMax, 50.9227)
    assert ex3.srs.IsSame(EPSG4326)

    # Raster
    ex4 = Extent.load(CLC_RASTER_PATH)
    assert np.isclose(ex4.xMin, 4012100)
    assert np.isclose(ex4.yMin, 3031800)
    assert np.isclose(ex4.xMax, 4094600)
    assert np.isclose(ex4.yMax, 3111000)
    assert ex4.srs.IsSame(EPSG3035)

    # Location set
    ls = LocationSet(pointsInAachen3035, srs=3035)
    ex5 = Extent.load(ls)

    assert np.isclose(ex5.xMin, 6.02141)
    assert np.isclose(ex5.yMin, 50.51939)
    assert np.isclose(ex5.xMax, 6.371634)
    assert np.isclose(ex5.yMax, 50.846025)
    assert ex5.srs.IsSame(EPSG4326)


def test_Extent_xyXY():
    ex1 = Extent.load((1, 2, 3, 4), srs=4326)
    assert np.isclose(ex1.xyXY, (1, 2, 3, 4)).all()


def test_Extent_xXyY():
    ex1 = Extent.load((1, 2, 3, 4), srs=4326)
    assert np.isclose(ex1.xXyY, (1, 3, 2, 4)).all()


def test_Extent_box():
    ex1 = Extent.load((1, 2, 4, 4), srs=4326)
    assert np.isclose(ex1.box.Area(), 6.0)


def test_Extent___eq__():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    ex2 = Extent.fromVector(source=MULTI_FTR_SHAPE_PATH)
    ex3 = Extent.fromVector(source=MULTI_FTR_SHAPE_PATH)

    # Equality
    assert (ex2 != ex1)
    assert (ex2 == ex2)
    assert (ex2 == ex3)


def test_Extent___add__():
    # setup
    ex1 = Extent.load((1, 2, 3, 4), srs=EPSG4326)
    ex2 = Extent.load((1, 1, 3, 3), srs=EPSG4326)

    # Simple add
    ex = ex1 + ex2
    assert ex.xMin == 1
    assert ex.xMax == 3
    assert ex.yMin == 1
    assert ex.yMax == 4


def test_Extent_exportWKT():
    # setup
    ex1 = Extent(1, 2, 3, 4, srs=srs.EPSG4326)
    s = ex1.exportWKT("|||")
    s2 = 'POLYGON ((1 2 0,3 2 0,3 4 0,1 4 0,1 2 0))|||GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    assert s == s2

    ex1 = Extent(1, 2, 3, 4, srs=srs.EPSG3857)
    s = ex1.exportWKT("|||")
    s2 = 'POLYGON ((1 2 0,3 2 0,3 4 0,1 4 0,1 2 0))|||PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'
    assert s == s2


def test_Extent_pad():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)

    # do test
    ex_pad = ex1.pad(1)

    # check
    assert np.isclose(ex_pad.xyXY, (-6, -5, 4, 11)).all()


def test_Extent_shift():
    ex = Extent.load((1, 2, 3, 4), srs=4326)
    ex1 = ex.shift(-1, 2)
    np.isclose(ex1.xMin, 0)
    np.isclose(ex1.xMax, 2)
    np.isclose(ex1.yMin, 4)
    np.isclose(ex1.yMax, 6)


def test_Extent_fitsResolution():
    ex = Extent(-6, -5, 4, 11)
    assert ex.fitsResolution(0.5)
    assert ex.fitsResolution(2)
    assert not ex.fitsResolution(3)


def test_Extent_fit():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)
    ex2 = Extent.fromVector(source=MULTI_FTR_SHAPE_PATH)

    # Fiting
    ex_fit1 = ex1.fit(2, float)
    assert np.isclose(ex_fit1.xyXY, (-6.0, -4.0, 4.0, 10.0)).all()

    ex_fit2 = ex2.fit(0.01)
    assert np.isclose(ex_fit2.xyXY, (6.21, 48.86, 7.79, 49.94)).all()


def test_Extent_corners():
    ex = Extent.load((1, 2, 3, 4), srs=4326)

    # Points as Tuples
    pts = ex.corners()
    assert np.isclose(pts[0], (1, 2)).all()  # bottom left
    assert np.isclose(pts[1], (3, 2)).all()  # bottom right
    assert np.isclose(pts[2], (1, 4)).all()  # top left
    assert np.isclose(pts[3], (3, 4)).all()  # top right

    # Points as Geometries
    pts = ex.corners(asPoints=True)
    assert pts[0].GetSpatialReference().IsSame(EPSG4326)
    assert pts[0].GetX() == 1  # bottom left, X
    assert pts[0].GetY() == 2  # bottom left, Y
    assert pts[1].GetX() == 3  # bottom right, X
    assert pts[1].GetY() == 2  # bottom right, Y
    assert pts[2].GetX() == 1  # top left, X
    assert pts[2].GetY() == 4  # top left, Y
    assert pts[3].GetX() == 3  # top right, X
    assert pts[3].GetY() == 4  # top right, Y


def test_Extent_castTo():
    # setup
    ex3 = Extent.fromGeom(GEOM)

    # Projecting
    ex_cast = ex3.castTo(EPSG3035)
    assert np.isclose(ex_cast.fit(1).xyXY,
                      (4329833, 835682, 4770009, 1681039)).all()


def test_Extent_inSourceExtent():
    # setup
    ex2 = Extent.fromVector(source=MULTI_FTR_SHAPE_PATH)

    # Test if in source
    assert (ex2.inSourceExtent(LUX_SHAPE_PATH) == True)
    assert (ex2.inSourceExtent(AACHEN_SHAPE_PATH) == False)

    # Overlapping, but not within eachother
    ext1 = Extent(0, 0, 3, 3, srs=4326)
    ext2 = Extent(-1, 1, 4, 2, srs=4326)
    vec = vector.createVector(ext2.box)

    assert (ext1.inSourceExtent(vec) == True)


def test_Extent_filterSources():
    sources = source("*.shp")
    ex = Extent.fromVector(source=AACHEN_SHAPE_PATH)

    goodSources = list(ex.filterSources(sources))
    # Inclusion
    assert AACHEN_ZONES in goodSources
    assert AACHEN_POINTS in goodSources

    # Exclusion
    assert not BOXES in goodSources
    assert not LUX_SHAPE_PATH in goodSources
    assert not LUX_LINES_PATH in goodSources


def test_Extent_containsLoc():
    ex1 = Extent(3, 4, 5, 10, srs=EPSG4326)

    assert ex1.containsLoc((4, 8))
    assert not ex1.containsLoc((7, 0))

    s = ex1.containsLoc([(4, 8), (7, 0), (7, 1), (3, 4)])
    assert s[0] == True
    assert s[1] == False
    assert s[2] == False
    assert s[3] == True


def test_Extent_contains():
    # setup
    ex1 = Extent(3, -4, -5, 10, srs=EPSG4326)

    # Test for contains
    assert (ex1.contains(Extent(-5, -4, 3, 12)) == False)
    assert (ex1.contains(Extent(-5, -3.3333, 2.0002, 8)) == True)
    assert (ex1.contains(Extent(-2.0, -3.5, 1.0, 8.5), 0.5) == True)
    assert (ex1.contains(Extent(-2.0, -3.25, 1.0, 8.25), 0.5) == False)


def test_Extent_findWithin():
    bigExt = Extent.fromRaster(CLC_RASTER_PATH)
    smallExt = Extent(4050200, 3076800, 4055400, 3080300, srs=EPSG3035)

    # do regular
    w1 = bigExt.findWithin(smallExt, res=100)
    assert w1.xStart == 381  # xStart
    assert w1.yStart == 307  # yStart
    assert w1.xWin == 52  # xWin
    assert w1.yWin == 35  # yWin

    # do flipped (only yStart should change)
    w2 = bigExt.findWithin(smallExt, res=100, yAtTop=False)
    assert w2.xStart == 381  # flipped - xStart
    assert w2.yStart == 450  # flipped - yStart
    assert w2.xWin == 52  # flipped - xWin
    assert w2.yWin == 35  # flipped - yWin


def test_Extent_computePixelSize():
    ex = Extent(0, 1, 2, 3, srs=EPSG4326)

    dx, dy = ex.computePixelSize(5)
    assert np.isclose(dx, 0.4)
    assert np.isclose(dy, 0.4)

    dx, dy = ex.computePixelSize(5, 10)
    assert np.isclose(dx, 0.4)
    assert np.isclose(dy, 0.2)


def test_Extent_createRaster():
    ex = Extent.fromRaster(CLC_RASTER_PATH)

    # Test failure on bad resolution
    # 200m does not fit to the CLC raster by design
    try:
        r = ex.createRaster(pixelHeight=200, pixelWidth=200, fill=2)
        assert False
    except error.GeoKitExtentError:
        assert True
    else:
        assert False

    # Test successful
    r = ex.createRaster(pixelHeight=100, pixelWidth=100, fill=2)
    assert np.isclose(raster.extractMatrix(r).mean(), 2)


def test_Extent_extractMatrix():

    # setup
    ex = Extent(6.022, 50.477, 6.189, 50.575).castTo(EPSG3035).fit(100)

    # extract
    mat1 = ex.extractMatrix(CLC_RASTER_PATH)
    assert np.isclose(mat1.sum(), 392284)

    # extract
    mat2 = ex.extractMatrix(CLC_FLIPCHECK_PATH)
    assert np.isclose(mat2.sum(), 392284)

    # Make sure matricies are the same
    assert np.isclose(mat1, mat2).all()

    # test fail since the given extent does not fit in the grid of the raster
    try:
        p = ex.shift(dx=1).extractMatrix(CLC_RASTER_PATH)
        assert False
    except util.GeoKitError as e:
        assert True
    else:
        assert False


def test_Extent_warp():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(3035).fit(200)

    # Change resolution to disk
    d = ex.warp(CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200,
                output=result("extent_warp1.tif"))
    v1 = raster.extractMatrix(d)
    assert np.isclose(v1.mean(), 17.18144279)

    # change resolution to memory
    d = ex.warp(CLC_RASTER_PATH, pixelHeight=200, pixelWidth=200)
    v2 = raster.extractMatrix(d)
    assert np.isclose(v1, v2).all()

    # Do a cutline from disk
    d = ex.warp(CLC_RASTER_PATH, pixelHeight=100, pixelWidth=100,
                cutline=AACHEN_SHAPE_PATH, output=result("extent_warp3.tif"), noData=99)
    v3 = raster.extractMatrix(d)
    assert np.isclose(v3.mean(), 66.02815723)
    assert np.isclose(v3[0, 0], 99)


def test_Extent_rasterize():

    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(3035).fit(250)

    # Simple vectorization to file
    r = ex.rasterize(source=AACHEN_ZONES, pixelWidth=250,
                     pixelHeight=250, output=result("extent_rasterized1.tif"))
    mat1 = raster.extractMatrix(r)
    assert np.isclose(mat1.mean(), 0.22881653)

    # Simple vectorization to mem
    r = ex.rasterize(source=AACHEN_ZONES, pixelWidth=250, pixelHeight=250, )
    mat2 = raster.extractMatrix(r)
    assert np.isclose(mat2, mat1).all()

    # Write attribute values to disc
    r = ex.rasterize(source=AACHEN_ZONES, value="YEAR", pixelWidth=250,
                     pixelHeight=250, output=result("extent_rasterized3.tif"), noData=-1)
    mat = raster.extractMatrix(r, autocorrect=True)
    assert np.isclose(np.isnan(mat).sum(), 22025)
    assert np.isclose(np.nanmean(mat), 1996.36771232)


def test_Extent_extractFeatures():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH)

    # Test loading as a dataframe
    vi = ex.extractFeatures(AACHEN_ZONES, asPandas=True)
    assert vi.shape[0] == 101


def test_Extent_mutateVector():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(4326)

    # Test simple clipping
    vi = ex.mutateVector(AACHEN_ZONES, matchContext=False,
                         output=result("extent_mutateVector1.shp"))
    info = vector.vectorInfo(vi)
    assert np.isclose(info.count, 101)
    assert info.srs.IsSame(EPSG3035)

    # clip and convert
    vi = ex.mutateVector(AACHEN_ZONES, matchContext=True,)
    info = vector.vectorInfo(vi)
    assert np.isclose(info.count, 101)
    assert info.srs.IsSame(EPSG4326)

    # Simple operation
    def docenter(x): return {'geom': x.geom.Centroid()}
    vi = ex.mutateVector(AACHEN_ZONES, matchContext=True, keepAttributes=False,
                         processor=docenter, output=result("extent_mutateVector3.shp"))
    info = vector.vectorInfo(vi)
    assert np.isclose(len(info.attributes), 1)
    assert np.isclose(info.count, 101)
    assert info.srs.IsSame(EPSG4326)


def test_Extent_mutateRaster():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH).castTo(4326).fit(0.001)

    # test a simple clip
    r = ex.mutateRaster(CLC_RASTER_PATH, output=result(
        "extent_mutateRaster1.tif"))
    mat = raster.extractMatrix(r)
    assert np.isclose(mat.mean(), 17.14654805)

    # test a clip and warp
    r = ex.mutateRaster(CLC_RASTER_PATH, pixelHeight=0.001, pixelWidth=0.001,
                        matchContext=True, output=result("extent_mutateRaster2.tif"), resampleAlg='near')
    mat2 = raster.extractMatrix(r)
    assert np.isclose(mat2.mean(), 17.14768769)

    # simple processor
    @util.KernelProcessor(1, edgeValue=-9999)
    def max_3x3(mat):
        goodValues = mat[mat != -9999].flatten()
        return goodValues.max()

    r = ex.mutateRaster(CLC_RASTER_PATH, processor=max_3x3,
                        output=result("extent_mutateRaster3.tif"))
    mat3 = raster.extractMatrix(r)
    assert np.isclose(mat3.mean(), 19.27040301)


def test_Extent_clipRaster():
    ex = Extent.fromVector(AACHEN_SHAPE_PATH)

    # test a simple clip
    r = ex.clipRaster(AACHEN_URBAN_LC)
    mat = raster.extractMatrix(r)

    assert np.isclose(mat.mean(), 1.583447145588637)
    assert np.isclose(mat.std(), 0.5784475661496283)

    ri = raster.rasterInfo(r)
    assert ri.dx == 100
    assert ri.dy == 100
    assert ri.xMin == 4038300.0
    assert ri.yMin == 3048700.0
    assert ri.xMax == 4067000.0
    assert ri.yMax == 3101000.0


def test_Extent_contoursFromRaster():
    ext = Extent.fromVector(AACHEN_SHAPE_PATH)
    geoms = ext.contoursFromRaster(AACHEN_URBAN_LC,
                                   contourEdges=[1, 2, 3], transformGeoms=True)

    assert geoms.iloc[0].geom.GetSpatialReference().IsSame(ext.srs)
    assert len(geoms) == 95
    assert np.isclose(geoms.iloc[61].geom.Area(), 0.08834775465377398)
    assert geoms.iloc[61].ID == 1


def test_Extent_subTiles():
    ext = Extent.fromVector(_test_data_['aachenShapefile.shp'])

    tiles = [Extent.fromTile(t.xi, t.yi, t.zoom) for t in ext.subTiles(9)]

    assert len(tiles) == 4

    assert np.isclose(tiles[0].xMin, 626172.135712)
    assert np.isclose(tiles[0].xMax, 704443.652676)
    assert np.isclose(tiles[0].yMin, 6574807.424978)
    assert np.isclose(tiles[0].yMax, 6653078.941942)
    assert tiles[0].srs.IsSame(srs.EPSG3857)

    assert np.isclose(tiles[1].xMin, 626172.135712)
    assert np.isclose(tiles[1].xMax, 704443.652676)
    assert np.isclose(tiles[1].yMin, 6496535.908014)
    assert np.isclose(tiles[1].yMax, 6574807.424978)
    assert tiles[1].srs.IsSame(srs.EPSG3857)

    assert np.isclose(tiles[2].xMin, 704443.652676)
    assert np.isclose(tiles[2].xMax, 782715.169640)
    assert np.isclose(tiles[2].yMin, 6574807.424978)
    assert np.isclose(tiles[2].yMax, 6653078.941942)
    assert tiles[2].srs.IsSame(srs.EPSG3857)

    assert np.isclose(tiles[3].xMin, 704443.652676)
    assert np.isclose(tiles[3].xMax, 782715.169640)
    assert np.isclose(tiles[3].yMin, 6496535.908014)
    assert np.isclose(tiles[3].yMax, 6574807.424978)
    assert tiles[3].srs.IsSame(srs.EPSG3857)

    tiles = list(ext.subTiles(10))
    assert len(tiles) == 9

    tiles = list(ext.subTiles(11))
    assert len(tiles) == 20

    tiles = list(ext.subTiles(12))
    assert len(tiles) == 63


def test_Extent_tileBox():
    ext = Extent.fromVector(_test_data_['aachenShapefile.shp'])
    ext_box = ext.tileBox(12)

    assert np.isclose(ext_box.xMin, 655523.954574)
    assert np.isclose(ext_box.xMax, 724011.531917)
    assert np.isclose(ext_box.yMin, 6525887.726875)
    assert np.isclose(ext_box.yMax, 6613943.183460)
    assert ext_box.srs.IsSame(srs.EPSG3857)


def test_Extent_mosiacTiles():
    ext = Extent.fromVector(_test_data_['aachenShapefile.shp'])
    ras = ext.tileMosaic(join(_test_data_['prior_tiles'],
                              "osm_roads_minor.{z}.{x}.{y}.tif"), 9,)
    rasmat = raster.extractMatrix(ras)
    assert np.isclose(np.nanmean(rasmat), 568.8451589061345)
    assert np.isclose(np.nanstd(rasmat), 672.636988117134)
