import numpy as np
from osgeo import ogr, osr, gdal
import warnings
import pandas as pd
import smopy
from collections import namedtuple

from . import util as UTIL
from . import srs as SRS


class GeoKitGeomError(UTIL.GeoKitError):
    pass


POINT = ogr.wkbPoint
MULTIPOINT = ogr.wkbMultiPoint
LINE = ogr.wkbLineString
MULTILINE = ogr.wkbMultiLineString
POLYGON = ogr.wkbPolygon
MULTIPOLYGON = ogr.wkbMultiPolygon

####################################################################
# Geometry convenience functions


def point(*args, srs='latlon'):
    """Make a simple point geometry

    Parameters:
    -----------
    *args : numeric, numeric or (numeric, numeric)
        The X and Y coordinate of the point to create

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the point to create
          * If not given, longitude/latitude is assumed
          * srs MUST be given as a keyword argument

    Returns:
    --------
    ogr.Geometry

    Example:
    ------
    point(x, y [,srs])
    point( (x, y) [,srs] )

    """
    if len(args) == 1:
        x, y = args[0]
    elif len(args) == 2:
        x = args[0]
        y = args[1]
    else:
        raise GeoKitGeomError(
            "Too many positional inputs. Did you mean to specify \"srs=\"?")

    """make a point geometry from given coordinates (x,y) and srs"""
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(float(x), float(y))
    if not srs is None:
        pt.AssignSpatialReference(SRS.loadSRS(srs))
    return pt


def makePoint(*args, **kwargs):
    """alias for geokit.geom.point(...)"""
    msg = "makePoint will be removed soon. Switch to 'point'"
    warnings.warn(msg, Warning)
    return point(*args, **kwargs)


def box(*args, srs=4326):
    """Make an ogr polygon object from extents

    Parameters:
    -----------
    *args : 4 numeric argument, or one tuple argument with 4 numerics
        The X_Min, Y_Min, X_Max and Y_Max bounds of the box to create

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the point to create
          * If not given, longitude/latitude is assumed
          * srs MUST be given as a keyword argument

    Returns:
    --------
    ogr.Geometry

    Example:
    ------
    box(xMin, yMin, xMax, yMax [, srs])
    box( (xMin, yMin, xMax, yMax) [, srs])
    """
    if (len(args) == 1):
        xMin, yMin, xMax, yMax = args[0]
    elif (len(args) == 4):
        xMin, yMin, xMax, yMax = args
    else:
        raise GeoKitGeomError(
            "Incorrect number positional inputs (only accepts 1 or 4). Did you mean to specify \"srs=\"?")

    # make sure inputs are good
    xMin = float(xMin)
    xMax = float(xMax)
    yMin = float(yMin)
    yMax = float(yMax)

    # make box
    outBox = ogr.Geometry(ogr.wkbPolygon)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in [(xMin, yMin), (xMax, yMin), (xMax, yMax), (xMin, yMax), (xMin, yMin)]:
        ring.AddPoint(x, y)

    outBox.AddGeometry(ring)
    if(not srs is None):
        srs = SRS.loadSRS(srs)
        outBox.AssignSpatialReference(srs)
    return outBox


def tile(xi, yi, zoom):
    """Generates a box corresponding to a tile used for "slippy maps"

    Parameters:
    -----------
    xi : int
        The tile's X-index
        - Range depends on zoom value

    yi : int
        The tile's Y-index
        - Range depends on zoom value

    zoom : int
        The tile's zoom index
        - Range is between 0 and 18

    Returns:
    --------
    ogr.Geometry

    """
    tl = smopy.num2deg(xi - 0.0, yi + 1.0, zoom)[::-1]
    br = smopy.num2deg(xi + 1.0, yi - 0.0, zoom)[::-1]

    o = SRS.xyTransform([tl, br], fromSRS=SRS.EPSG4326,
                        toSRS=SRS.EPSG3857, outputFormat='xy')

    return box(o.x.min(), o.y.min(), o.x.max(), o.y.max(), srs=SRS.EPSG3857)

def tileAt(x, y, zoom, srs):
    """Generates a box corresponding to a tile at the coordinates 'x' and 'y'
     in the given srs,

    Parameters:
    -----------
    x : float
        The X coordinate to search for a tile around

    y : float
        The Y coordinate to search for a tile around

    zoom : int
        The tile's zoom index
        - Range is between 0 and 18

    srs : anything acceptable to SRS.loadSRS
        The SRS of the given 'x' & 'y' coordinates 

    Returns:
    --------
    ogr.Geometry

    """
    t = SRS.tileIndexAt(x=x, y=y, zoom=zoom, srs=srs)

    return tile(t.xi,t.yi,t.zoom)


Tile = namedtuple("Tile", "xi yi zoom")


def subTiles(geom, zoom, checkIntersect=True, asGeom=False):
    """
    Generate a collection of tiles which encompass the passed geometry.

    Parameters:
    -----------
    geom : ogr.Geometry
        The geometry to be analyzed

    zoom : int
        The zoom level to generate tiles on

    checkIntersect : bool
        If True, only tiles which overlap the given geomtry are returned

    asGeom : bool
        If True, geometry object corresponding to each tile is yielded,
        instead of (xi,yi,zoom) tuples

    Returns:
    --------

    If asGeom is False: Generates (xi, yi, zoom) tuples
    If asGeom is True:  Generates Geometry objects
    """
    geom4326 = transform(geom, toSRS=SRS.EPSG4326)
    if checkIntersect:
        geom3857 = transform(geom, toSRS=SRS.EPSG3857)

    xmin, xmax, ymin, ymax = geom4326.GetEnvelope()

    tl_xi, tl_yi = smopy.deg2num(ymax, xmin, zoom)
    br_xi, br_yi = smopy.deg2num(ymin, xmax, zoom)

    for xi in range(tl_xi, br_xi + 1):
        for yi in range(tl_yi, br_yi + 1):

            if checkIntersect or asGeom:
                gtile = tile(xi, yi, zoom)

            if checkIntersect:
                if not geom3857.Intersects(gtile):
                    continue

            if asGeom:
                yield gtile
            else:
                yield Tile(xi, yi, zoom)


def tileize(geom, zoom):
    """Deconstruct a given geometry into a set of tiled geometries

    Returns: Generator of ogr.Geometry objects
    """
    geom = transform(geom, toSRS=SRS.EPSG3857)
    for tile_ in subTiles(geom, zoom, asGeom=True, checkIntersect=True):
        yield geom.Intersection(tile_)


def makeBox(*args, **kwargs):
    """alias for geokit.geom.box(...)"""
    msg = "makeBox will be removed soon. Switch to 'box'"
    warnings.warn(msg, Warning)
    return box(*args, **kwargs)


def polygon(outerRing, *args, srs=4326):
    """Creates an OGR Polygon obect from a given set of points

    Parameters:
    -----------
    outerRing : [(x,y), ] or Nx2 numpy.ndarray
        The polygon's outer edge

    *args : [(x,y), ] or Nx2 numpy.ndarray
        The inner edges of the polygon
          * Each input forms a single edge
          * Inner rings cannot interset the outer ring or one another 
          * NOTE! For proper drawing in matplotlib, inner rings must be given in 
            the opposite orientation as the outer ring (clockwise vs 
            counterclockwise)

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the polygon to create
          * If not given, longitude/latitude is assumed

    Returns:
    --------
    ogr.Geometry

    Example:
    ------
    Make a diamond cut out of a box...

      box = [(-2,-2), (-2,2), (2,2), (2,-2), (-2,-2)]
      diamond = [(0,1), (-0.5,0), (0,-1), (0.5,0), (0,1)]

      geom = polygon( box, diamond )
    """
    if not srs is None:
        srs = SRS.loadSRS(srs)

    # Make the complete geometry
    g = ogr.Geometry(ogr.wkbPolygon)
    if not srs is None:
        g.AssignSpatialReference(srs)

    # Make the outer ring
    otr = ogr.Geometry(ogr.wkbLinearRing)
    if not srs is None:
        otr.AssignSpatialReference(srs)
    [otr.AddPoint(float(x), float(y)) for x, y in outerRing]
    g.AddGeometry(otr)

    # Make the inner rings (maybe)
    for innerRing in args:
        tmp = ogr.Geometry(ogr.wkbLinearRing)
        if not srs is None:
            tmp.AssignSpatialReference(srs)
        [tmp.AddPoint(float(x), float(y)) for x, y in innerRing]

        g.AddGeometry(tmp)

    # Make sure geom is valid
    g.CloseRings()
    if not g.IsValid():
        raise GeoKitGeomError("Polygon is invalid")

    # Done!
    return g


def makePolygon(*args, **kwargs):
    """alias for geokit.geom.polygon(...)"""
    msg = "makePolygon will be removed soon. Switch to 'polygon'"
    warnings.warn(msg, Warning)
    return polygon(*args, **kwargs)


def line(points, srs=4326):
    """Creates an OGR Line obect from a given set of points

    Parameters:
    -----------
    Points : [(x,y), ] or Nx2 numpy.ndarray
        The point defining the line

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the line to create

    Returns:
    --------
    ogr.Geometry
    """

    # Make the complete geometry
    g = ogr.Geometry(ogr.wkbLineString)
    if not srs is None:
        g.AssignSpatialReference(srs)

    # Make the line
    [g.AddPoint(x, y) for x, y in points]
    # g.AddGeometry(otr)

    # Ensure valid
    if not g.IsValid():
        raise GeoKitGeomError("Polygon is invalid")

    # Done!
    return g


def makeLine(*args, **kwargs):
    """alias for geokit.geom.line(...)"""
    msg = "makeLine will be removed soon. Switch to 'line'"
    warnings.warn(msg, Warning)
    return line(*args, **kwargs)


def empty(gtype, srs=None):
    """
    Make a generic OGR geometry of a desired type

    *Not for the feint of heart*

    Parameters:
    -----------
    gtpe : str
        The geometry type to make
          * Point, MultiPoint, Line, MultiLine, Polygon, MultiPolygon, ect...

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometry to create

    Returns:
    --------
    ogr.Geometry
    """
    if not hasattr(ogr, "wkb" + gtype):
        raise GeoKitGeomError("Could not find geometry type: " + gtype)
    geom = ogr.Geometry(getattr(ogr, "wkb" + gtype))

    if not srs is None:
        geom.AssignSpatialReference(SRS.loadSRS(srs))

    return geom


def makeEmpty(*args, **kwargs):
    """alias for geokit.geom.empty(...)"""
    msg = "makeEmpty will be removed soon. Switch to 'empty'"
    warnings.warn(msg, Warning)
    return empty(*args, **kwargs)


def extractVerticies(geom):
    """Get all verticies found on the geometry as a Nx2 numpy.ndarray"""
    isMulti = "MULTI" in geom.GetGeometryName()
    # Check geometry type
    if "LINE" in geom.GetGeometryName():
        if isMulti:
            pts = []
            for gi in range(geom.GetGeometryCount()):
                pts.append(geom.GetGeometryRef(gi).GetPoints())
        else:
            pts = geom.GetPoints()
    elif "POLYGON" in geom.GetGeometryName():
        if isMulti:
            pts = []
            for gi in range(geom.GetGeometryCount()):
                newGeom = geom.GetGeometryRef(gi).GetBoundary()
                pts.append(extractVerticies(newGeom))
        else:
            newGeom = geom.GetBoundary()
            pts = extractVerticies(newGeom)

    elif "POINT" in geom.GetGeometryName():
        if isMulti:
            pts = []
            for gi in range(geom.GetGeometryCount()):
                pts.append(geom.GetGeometryRef(gi).GetPoints())
        else:
            pts = geom.GetPoints()

    else:
        raise GeoKitGeomError("Cannot extract points from geometry ")

    if isMulti:
        out = np.concatenate(pts)
    else:
        out = np.array(pts)

    if out.shape[1] == 3:  # This can happen when POINTs are extracted
        out = out[:, :2]
    return out

# 3
# Make a geometry from a WKT string


def convertWKT(wkt, srs=None):
    """Make a geometry from a well known text (WKT) string

    Parameters:
    -----------
    wkt : str
        The WKT string to convert

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometry to create
    """
    geom = ogr.CreateGeometryFromWkt(wkt)  # Create new geometry from string
    if geom is None:  # test for success
        raise GeoKitGeomError("Failed to create geometry")
    if srs:
        geom.AssignSpatialReference(SRS.loadSRS(srs))  # Assign the given srs
    return geom


def convertGeoJson(geojson, srs=3857):
    """Make a geometry from a well known text (WKT) string
    TODO: UPDATE!!!
    Parameters:
    -----------
    wkt : str
        The WKT string to convert

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometry to create
    """
    geom = ogr.CreateGeometryFromJson(
        geojson)  # Create new geometry from string
    if geom is None:  # test for success
        raise GeoKitGeomError("Failed to create geometry")
    if srs:
        geom.AssignSpatialReference(SRS.loadSRS(srs))  # Assign the given srs
    return geom

# 3
# Make a geometry from a matrix mask


def polygonizeMatrix(matrix, bounds=None, srs=None, flat=False, shrink=True, _raw=False):
    """Create a geometry set from a matrix of integer values

    Each unique-valued group of pixels will be converted to a geometry

    Parameters:
    -----------
    matrix : matrix_like
        The matrix which will be turned into a geometry set
          * Must be 2 dimensional
          * Must be integer or boolean type

    bounds : (xMin, yMin, xMax, yMax) or geokit.Extent
        Determines the boundary context for the given matrix and will scale
        the resulting geometry's coordinates accordingly
          * If a boundary is not given, the geometry coordinates will 
            correspond to the mask's indicies
          * If the boundary is given as an Extent object, an srs input is not 
            required


    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs context for the given matrix and of the geometries to create

    flat : bool
        If True, flattens the resulting geometries which share a contiguous matrix
        value into a single geometry object

    shrink : bool
        If True, shrink all geoms by a tiny amount in order to avoid geometry 
        overlapping issues
          * The total amount shrunk should be very very small
          * Generally this should be left as True unless it is ABSOLUTELY 
            necessary to maintain the same area

    Returns:
    --------
    pandas.DataFrame -> With columns:
                            'geom' -> The contiguous-valued geometries
                            'value' -> The value for each geometry

    """

    # Make sure we have a boolean numpy matrix
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    if matrix.dtype == np.bool or matrix.dtype == np.uint8:
        dtype = "GDT_Byte"
    elif np.issubdtype(matrix.dtype, np.integer):
        dtype = "GDT_Int32"
    else:
        raise GeoKitGeomError(
            "matrix must be a 2D boolean or integer numpy ndarray")

    # Make boundaries if not given
    if bounds is None:
        # bounds in xMin, yMin, xMax, yMax
        bounds = (0, 0, matrix.shape[1], matrix.shape[0])
        pixelHeight = 1
        pixelWidth = 1

    try:  # first try for a tuple
        xMin, yMin, xMax, yMax = bounds
    except:  # next assume the user gave an extent object
        try:
            xMin, yMin, xMax, yMax = bounds.xyXY
            srs = bounds.srs
        except:
            raise GeoKitGeomError("Could not understand 'bounds' input")

    pixelHeight = (yMax - yMin) / matrix.shape[0]
    pixelWidth = (xMax - xMin) / matrix.shape[1]

    if not srs is None:
        srs = SRS.loadSRS(srs)

    # Make a raster dataset and pull the band/maskBand objects

    # used 'round' instead of 'int' because this matched GDAL behavior better
    cols = int(round((xMax - xMin) / pixelWidth))
    rows = int(round((yMax - yMin) / abs(pixelHeight)))
    originX = xMin
    originY = yMax  # Always use the "Y-at-Top" orientation

    # Open the driver
    driver = gdal.GetDriverByName('Mem')  # create a raster in memory
    raster = driver.Create('', cols, rows, 1, getattr(gdal, dtype))

    if(raster is None):
        raise GeoKitGeomError("Failed to create temporary raster")

    raster.SetGeoTransform(
        (originX, abs(pixelWidth), 0, originY, 0, -1 * abs(pixelHeight)))

    # Set the SRS
    if not srs is None:
        rasterSRS = SRS.loadSRS(srs)
        raster.SetProjection(rasterSRS.ExportToWkt())

    # Set data into band
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(matrix)

    band.FlushCache()
    raster.FlushCache()

    #rasDS = createRaster(bounds=bounds, data=matrix, noDataValue=0, pixelWidth=pixelWidth, pixelHeight=pixelHeight, srs=srs)

    # Do a polygonize
    rasBand = raster.GetRasterBand(1)
    maskBand = rasBand.GetMaskBand()

    vecDS = gdal.GetDriverByName("Memory").Create(
        '', 0, 0, 0, gdal.GDT_Unknown)
    vecLyr = vecDS.CreateLayer("mem", srs=srs)

    field = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(field)

    # Polygonize geometry
    result = gdal.Polygonize(rasBand, maskBand, vecLyr, 0)
    if(result != 0):
        raise GeoKitGeomError("Failed to polygonize geometry")

    # Check how many features were created
    ftrN = vecLyr.GetFeatureCount()

    if(ftrN == 0):
        #raise GlaesError("No features in created in temporary layer")
        msg = "No features in created in temporary layer"
        warnings.warn(msg, UserWarning)
        return

    # Extract geometries and values
    geoms = []
    rid = []
    for i in range(ftrN):
        ftr = vecLyr.GetFeature(i)
        geoms.append(ftr.GetGeometryRef().Clone())
        rid.append(ftr.items()["DN"])

    # Do shrink, maybe
    if shrink:
        # Compute shrink factor
        shrinkFactor = -0.00001 * (xMax - xMin) / matrix.shape[1]
        geoms = [g.Buffer(shrinkFactor) for g in geoms]

    # Do flatten, maybe
    if flat:
        geoms = np.array(geoms)
        rid = np.array(rid)

        finalGeoms = []
        finalRID = []
        for _rid in set(rid):
            smallGeomSet = geoms[rid == _rid]
            finalGeoms.append(flatten(smallGeomSet) if len(
                smallGeomSet) > 1 else smallGeomSet[0])
            finalRID.append(_rid)
    else:
        finalGeoms = geoms
        finalRID = rid

    # Cleanup
    vecLyr = None
    vecDS = None
    maskBand = None
    rasBand = None
    raster = None

    # Done!
    if _raw:
        return finalGeoms, finalRID
    else:
        return pd.DataFrame(dict(geom=finalGeoms, value=finalRID))


def polygonizeMask(mask, bounds=None, srs=None, flat=True, shrink=True):
    """Create a geometry set from a matrix mask

    Each True-valued group of pixels will be converted to a geometry

    Parameters:
    -----------
    mask : matrix_like
        The mask which will be turned into a geometry set
          * Must be 2 dimensional
          * Must be boolean type
          * True values are interpreted as 'in the geometry'

    bounds : (xMin, yMin, xMax, yMax) or geokit.Extent
        Determines the boundary context for the given mask and will scale
        the resulting geometry's coordinates accordingly
          * If a boundary is not given, the geometry coordinates will 
            correspond to the mask's indicies
          * If the boundary is given as an Extent object, an srs input is not 
            required

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometries to create

    flat : bool
        If True, flattens the resulting geometries into a single geometry

    shrink : bool
        If True, shrink all geoms by a tiny amount in order to avoid geometry 
        overlapping issues
          * The total amount shrunk should be very very small
          * Generally this should be left as True unless it is ABSOLUTELY 
            neccessary to maintain the same area

    Returns:
    --------
    If 'flat' is True: ogr.Geometry
    else: [ogr.Geometry,  ]

    """
    # Make sure we have a boolean numpy matrix
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    if not (mask.dtype == np.bool or mask.dtype == np.uint8):
        raise GeoKitGeomError("Mask must be a 2D boolean numpy ndarray")

    # Do vectorization
    result = polygonizeMatrix(
        matrix=mask, bounds=bounds, srs=srs, flat=flat, shrink=shrink, _raw=True)[0]
    if flat:
        result = result[0]

    # Done!
    return result

# geometry transformer


def transform(geoms, toSRS='europe_m', fromSRS=None, segment=None):
    """Transform a geometry, or a list of geometries, from one SRS to another

    Parameters:
    -----------
    geoms : ogr.Geometry or [ogr.Geometry, ]
        The geometry or geometries to transform
          * All geometries must have the same spatial reference

    toSRS : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the output geometries
          * If no given, a Europe-centered relational system (EPSG3035) is chosen

    fromSRS : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the input geometries
          * Only needed if an SRS cannot be inferred from the geometry inputs or
            is, for whatever reason, the geometry's SRS is wrong

    segment : float; optional
        An optional segmentation length to apply to the input geometries BEFORE
        transformation occurs. The input geometries will be segmented such that 
        no line segment is longer than the given segment size
          * Units are in the input geometry's native unit
          * Use this for a more detailed transformation!

    Returns:
    --------
    ogr.Geometry or [ogr.Geometry, ]


    Note:
    -----
    When inferring the SRS from the given geometries, only the FIRST geometry
    is checked for an existing SRS

    """
    # make sure geoms is a list
    if isinstance(geoms, ogr.Geometry):
        returnSingle = True
        geoms = [geoms, ]
    else:  # assume geoms is iterable
        returnSingle = False
        try:
            geoms = list(geoms)
        except Exception as e:
            msg = "Could not determine geometry SRS"
            warnings.warn(msg, UserWarning)
            raise e

    # make sure geoms is a list
    if fromSRS is None:
        fromSRS = geoms[0].GetSpatialReference()
        if fromSRS is None:
            raise GeoKitGeomError("Could not determine fromSRS from geometry")

    # load srs's
    fromSRS = SRS.loadSRS(fromSRS)
    toSRS = SRS.loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    geoms = [g.Clone() for g in geoms]
    if not segment is None:
        [g.Segmentize(segment) for g in geoms]

    r = [g.Transform(trx) for g in geoms]
    if sum(r) > 0:  # check fro errors
        raise GeoKitGeomError("Errors in geometry transformations")

    # Done!
    if returnSingle:
        return geoms[0]
    else:
        return geoms


def boundsToBounds(bounds, boundsSRS, outputSRS):
    pts = flatten([point(bounds[0], bounds[1], srs=boundsSRS),
                   point(bounds[0], bounds[3], srs=boundsSRS),
                   point(bounds[2], bounds[1], srs=boundsSRS),
                   point(bounds[2], bounds[3], srs=boundsSRS)])
    pts.TransformTo(outputSRS)
    pts = extractVerticies(pts)

    bounds = (pts[:, 0].min(), pts[:, 1].min(),
              pts[:, 0].max(), pts[:, 1].max(), )
    return bounds

# 3
# Flatten a list of geometries


def flatten(geoms):
    """Flatten a list of geometries into a single geometry object

    Combine geometries by iteratively union-ing neighbors (according to index)
     * example, given a list of geometries (A,B,C,D,E,F,G,H,I,J):
          [ A  B  C  D  E  F  G  H  I  J ]
          [  AB    CD    EF    GH    IJ  ]
          [    ABCD        EFGH      IJ  ]
          [        ABCDEFGH          IJ  ]
          [               ABCDEFGHIJ     ]  <- This becomes the resulting geometry  

    Example:
    --------
        * A list of Polygons/Multipolygons will become a single Multipolygon
        * A list of Linestrings/MultiLinestrings will become a single MultiLinestring

    """
    if not isinstance(geoms, list):
        geoms = list(geoms)
        try:  # geoms is not a list, but it might be iterable
            geoms = list(geoms)
        except:
            raise ValueError("argument must be a list of geometries")

    if len(geoms) == 0:
        return None

    # Begin flattening
    while(len(geoms) > 1):
        newGeoms = []
        for gi in range(0, len(geoms), 2):

            try:
                if not geoms[gi].IsValid():
                    warnings.warn(
                        "WARNING: Invalid Geometry encountered", UserWarning)
                if not geoms[gi + 1].IsValid():
                    warnings.warn(
                        "WARNING: Invalid Geometry encountered", UserWarning)

                newGeoms.append(geoms[gi].Union(geoms[gi + 1]))
            except IndexError:  # should only occur when length of geoms is odd
                newGeoms.append(geoms[gi])

        geoms = newGeoms
    return geoms[0]


##########################################################################
# Drawing functions
def drawPoint(g, plotargs, ax, colorVal=None):
    kwargs = dict(marker='o', color='#C32148', linestyle='None')
    if not colorVal is None:
        kwargs["color"] = colorVal

    kwargs.update(plotargs)
    return ax.plot(g.GetX(), g.GetY(), **kwargs)


def drawMultiPoint(g, plotargs, ax, colorVal=None, skip=False):
    kwargs = dict(marker='.', color='#C32148', linestyle='None')
    if not colorVal is None:
        kwargs["color"] = colorVal
    kwargs.update(plotargs)

    points = extractVerticies(g)
    return ax.plot(points[:, 0], points[:, 1], **kwargs)


def drawLine(g, plotargs, ax, colorVal=None, skip=False):
    if skip:
        kwargs = plotargs.copy()
    else:
        kwargs = dict(marker='None', color='k', linestyle='-')
        if not colorVal is None:
            kwargs["color"] = colorVal
        kwargs.update(plotargs)

    points = extractVerticies(g)
    return ax.plot(points[:, 0], points[:, 1], **kwargs)


def drawMultiLine(g, plotargs, ax, colorVal=None):
    kwargs = dict(marker='None', color="#007959", linestyle='-')
    if not colorVal is None:
        kwargs["color"] = colorVal
    kwargs.update(plotargs)

    h = []
    for gi in range(g.GetGeometryCount()):
        h.append(drawLine(g.GetGeometryRef(gi), kwargs, ax, colorVal, True))
    return h


def drawLinearRing(g, plotargs, ax, colorVal=None):
    g.CloseRings()
    return drawLine(g, plotargs, ax)


def drawPolygon(g, plotargs, ax, colorVal=None, skip=False):
    from descartes import PolygonPatch
    from json import loads

    if g.GetGeometryCount() == 0:  # Geometry doesn't actually exist. skip it
        return None

    # Get main and hole edges
    boundaries = g.GetBoundary()
    if boundaries.GetGeometryName() == "LINESTRING":
        try:
            main = boundaries.GetPoints()
        except AttributeError:
            return  # Geometry doesn't actually exist. skip it
        holes = []
    else:
        mainG = boundaries.GetGeometryRef(0)
        try:
            main = mainG.GetPoints()
        except AttributeError:
            return  # Geometry doesn't actually exist. skip it
        holes = [boundaries.GetGeometryRef(i).GetPoints(
        ) for i in range(1, boundaries.GetGeometryCount())]

    patchData = dict(type='Polygon', coordinates=[])
    patchData["coordinates"].append(main)
    for hole in holes:
        patchData["coordinates"].append(hole)

    # Setup args
    if skip:
        kwargs = plotargs.copy()
    else:
        kwargs = dict(fc="#D9E9FF", ec="k", linestyle='-')
        if not colorVal is None:
            kwargs["fc"] = colorVal
        kwargs.update(plotargs)

    # Make patches
    mainPatch = PolygonPatch(patchData, **kwargs)
    return ax.add_patch(mainPatch)


def drawMultiPolygon(g, plotargs, ax, colorVal=None):
    kwargs = dict(fc="#D9E9FF", ec="k", linestyle='-')
    if not colorVal is None:
        kwargs["fc"] = colorVal
    kwargs.update(plotargs)

    h = []
    for gi in range(g.GetGeometryCount()):
        h.append(drawPolygon(g.GetGeometryRef(gi), kwargs, ax, colorVal, True))
    return h


def drawGeoms(geoms, srs=4326, ax=None, simplificationFactor=5000, colorBy=None, figsize=(12, 12), xlim=None, ylim=None, fontsize=16, hideAxis=False, cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap="viridis", cbar=True, cbax=None, cbargs=None, leftMargin=0.01, rightMargin=0.01, topMargin=0.01, bottomMargin=0.01, **mplArgs):
    """Draw geometries onto a matplotlib figure

    * Each geometry type is displayed as an appropriate plotting type
        -> Points/ Multipoints are displayed as points using plt.plot(...)
        -> Lines/ MultiLines are displayed as lines using plt.plot(...)
        -> Polygons/ MultiPolygons are displayed as patches using the descartes 
           library
    * Each geometry can be given its own set of matplotlib plotting parameters

    Notes:
    ------
    This function does not call plt.show() for the final display of the figure.
    This must be done manually after calling this function. Otherwise 
    plt.savefig(...) can be called to save the output somewhere.

    Sometimes geometries will disappear because of the simplification procedure.
    If this happens, the procedure can be avoided by setting simplificationFactor
    to None. This will take much more memory and will take longer to plot, however

    Parameters:
    -----------
    geoms : ogr.Geometry or [ogr.Geometry, ] or pd.DataFrame
        The geometries to be drawn
          * If a DataFrame is given, the function looks for geometries under a
            columns named 'geom'
          * plotting arguments can be given by adding a column named 'MPL:****'
            where '****' stands in for the argument to be added
              - For geometries that should ignore this argument, set it as None

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs in which to draw each geometry
          * If not given, longitude/latitude is assumed
          * Although geometries can be given in any SRS, it is very helpful if
            they are already provided in the correct SRS

    ax : matplotlib axis; optional
        The axis to draw the geometries on
          * If not given, a new axis is generated and returned

    simplificationFactor : float; optional
        The level to which geometries should be simplified. It can be thought of
        as the number of verticies allowed in either the X or Y dimension across
        the figure
          * A higher value means a more detailed plot, but may take longer to draw

    colorBy : str; optional
        The column in the geoms DataFrame to color by
          * Only useful when geoms is given as a DataFrame

    figsize : (int, int); optional
        The figure size to create when generating a new axis
          * If resultign figure looks wierd, altering the figure size is your best
            bet to make it look nicer

    xlim : (float, float); optional
        The x-axis limits

    ylim : (float, float); optional
        The y-axis limits

    fontsize : int; optional
        A base font size to apply to tick marks which appear
          * Titles and labels are given a size of 'fontsize' + 2

    hideAxis : bool; optional
        Instructs the created axis to hide its boundary
          * Only useful when generating a new axis

    cbarPadding : float; optional
        The spacing padding to add between the generated axis and the generated
        colorbar axis
          * Only useful when generating a new axis
          * Only useful when 'colorBy' is given

    cbarTitle : str; optional
        The title to give to the generated colorbar
          * If not given, but 'colorBy' is given, the same string for 'colorBy'
            is used
            * Only useful when 'colorBy' is given

    vmin : float; optional
        The minimum value to color
          * Only useful when 'colorBy' is given

    vmax : float; optional
        The maximum value to color
          * Only useful when 'colorBy' is given

    cmap : str or matplotlib ColorMap; optional
        The colormap to use when coloring
          * Only useful when 'colorBy' is given

    cbax : matplotlib axis; optional
        An explicitly given axis to use for drawing the colorbar
          * If not given, but 'colorBy' is given, an axis for the colorbar is 
            automatically generated

    cbargs : dict; optional
        keyword arguments to pass on when creating the colorbar 

    leftMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    rightMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    topMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    bottomMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    **mplArgs
        All other keyword arguments are passed on to the plotting functions called
        for each geometry
          * Will be applied to ALL geometries. Be careful since this can cause 
            errors when plotting geometries of different types

    Returns:
    --------
    A namedtuple containing:
       'ax' -> The map axis
       'handles' -> All geometry handles which were created in the order they were 
                    drawn
       'cbar' -> The colorbar handle if it was drawn

    """
    if isinstance(ax, UTIL.AxHands):
        ax = ax.ax

    if ax is None:
        newAxis = True

        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        if colorBy is None:  # We don't need a colorbar
            if not hideAxis:
                leftMargin += 0.07

            ax = plt.axes([leftMargin,
                           bottomMargin,
                           1 - (rightMargin + leftMargin),
                           1 - (topMargin + bottomMargin)])
            cbax = None

        else:  # We need a colorbar
            rightMargin += 0.08  # Add area on the right for colorbar text
            if not hideAxis:
                leftMargin += 0.07

            cbarExtraPad = 0.05
            cbarWidth = 0.04

            ax = plt.axes([leftMargin,
                           bottomMargin,
                           1 - (rightMargin + leftMargin +
                                cbarWidth + cbarPadding),
                           1 - (topMargin + bottomMargin)])

            cbax = plt.axes([1 - (rightMargin + cbarWidth),
                             bottomMargin + cbarExtraPad,
                             cbarWidth,
                             1 - (topMargin + bottomMargin + 2 * cbarExtraPad)])

        if hideAxis:
            ax.axis("off")
        else:
            ax.tick_params(labelsize=fontsize)
    else:
        newAxis = False

    # Be sure we have a list
    pargs = None
    isFrame = False
    if isinstance(geoms, ogr.Geometry):
        geoms = [geoms, ]

    elif isinstance(geoms, pd.DataFrame):  # We have a DataFrame with plotting arguments
        isFrame = True
        data = geoms.drop("geom", axis=1)
        geoms = geoms["geom"].values

        pargs = pd.DataFrame(index=data.index)
        for c in data.columns:
            if not c[:4] == "MPL:":
                continue
            pargs[c[4:]] = data[c]

        if pargs.size == 0:
            pargs = None

    else:  # Assume its an iterable
        geoms = list(geoms)

    # Check Geometry SRS
    if not srs is None:
        srs = SRS.loadSRS(srs)
        for gi, g in enumerate(geoms):
            gsrs = g.GetSpatialReference()
            if gsrs is None:
                continue  # Skip it if we don't know it...
            if not gsrs.IsSame(srs):
                geoms[gi] = transform(geoms[gi], srs)

    # Apply simplifications if required
    if not simplificationFactor is None:
        if xlim is None or ylim is None:
            xMin, yMin, xMax, yMax = 1e100, 1e100, -1e100, -1e100
            for g in geoms:
                _xMin, _xMax, _yMin, _yMax = g.GetEnvelope()

                xMin = min(_xMin, xMin)
                xMax = max(_xMax, xMax)
                yMin = min(_yMin, yMin)
                yMax = max(_yMax, yMax)

        if not xlim is None:
            xMin, xMax = xlim
        if not ylim is None:
            yMin, yMax = ylim

        simplificationValue = max(
            xMax - xMin, yMax - yMin) / simplificationFactor

        oGeoms = geoms
        geoms = []

        def doSimplify(g):
            ng = g.Simplify(simplificationValue)
            return ng

        for g in oGeoms:
            # carefulSimplification=False
            # if carefulSimplification and "MULTI" in g.GetGeometryName():
            if False and "MULTI" in g.GetGeometryName():  # This doesn't seem to help...
                subgeoms = []
                for gi in range(g.GetGeometryCount()):
                    ng = doSimplify(g.GetGeometryRef(gi))
                    subgeoms.append(ng)

                geoms.append(flatten(subgeoms))
            else:
                geoms.append(doSimplify(g))

    # Handle color value
    if not colorBy is None:
        colorVals = data[colorBy].values

        if isinstance(cmap, str):
            from matplotlib import cm
            cmap = getattr(cm, cmap)

        cValMax = colorVals.max() if vmax is None else vmax
        cValMin = colorVals.min() if vmin is None else vmin

        _colorVals = [cmap(v)
                      for v in (colorVals - cValMin) / (cValMax - cValMin)]

    # Do Plotting
    # make patches
    h = []

    for gi, g in enumerate(geoms):
        if not pargs is None:
            s = [not v is None for v in pargs.iloc[gi]]
            plotargs = pargs.iloc[gi, s].to_dict()
        else:
            plotargs = dict()
        plotargs.update(mplArgs)

        if not colorBy is None:
            colorVal = _colorVals[gi]
        else:
            colorVal = None

        # Determine type
        if g.GetGeometryName() == "POINT":
            h.append(drawPoint(g, plotargs, ax, colorVal))
        elif g.GetGeometryName() == "MULTIPOINT":
            h.append(drawMultiPoint(g, plotargs, ax, colorVal))
        elif g.GetGeometryName() == "LINESTRING":
            h.append(drawLine(g, plotargs, ax, colorVal))
        elif g.GetGeometryName() == "MULTILINESTRING":
            h.append(drawMultiLine(g, plotargs, ax, colorVal))
        elif g.GetGeometryName() == "LINEARRING":
            h.append(drawLinearRing(g, plotargs, ax, colorVal))
        elif g.GetGeometryName() == "POLYGON":
            h.append(drawPolygon(g, plotargs, ax, colorVal))
        elif g.GetGeometryName() == "MULTIPOLYGON":
            h.append(drawMultiPolygon(g, plotargs, ax, colorVal))
        else:
            msg = "Could not draw geometry of type:", pargs.index[gi], "->", g.GetGeometryName(
            )
            warnings.warn(msg, UserWarning)

    # Add the colorbar, maybe
    if not colorBy is None and cbar:
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=cValMin, vmax=cValMax)
        tmp = dict(cmap=cmap, norm=norm, orientation='vertical')
        if not cbargs is None:
            tmp.update(cbargs)
        cbar = ColorbarBase(cbax, **tmp)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(
            colorBy if cbarTitle is None else cbarTitle, fontsize=fontsize + 2)
    else:
        cbar = None

    # Do some formatting
    if newAxis:
        ax.set_aspect('equal')
        ax.autoscale(enable=True)

    if not xlim is None:
        ax.set_xlim(*xlim)
    if not ylim is None:
        ax.set_ylim(*ylim)

    # Organize return
    if isFrame:
        return UTIL.AxHands(ax, pd.Series(h, index=data.index), cbar)
    else:
        return UTIL.AxHands(ax, h, cbar)


def partition(geom, targetArea, growStep=None, _startPoint=0):
    """Partition a Polygon into some number of pieces whose areas should be close 
    to the targetArea

    WARNING: Not tested for several version. Will probably be removed later

    Inputs:
        geom : The geometry to partition
            - a single ogr Geometry object of POLYGON type

        targetArea - float : The ideal area of each partition
            * Most of the geometries will be around this area, but they can also be anywhere in the range 0 and 2x 

        growStep - float : The incremental buffer to add while searching for a suitable partition
            * Choose carefully!
                - A large growStep will make the algorithm run faster
                - A small growStep will produce a more accurate result
            * If no growStep is given, a decent one will be calculated
    """
    if growStep is None:
        growStep = np.sqrt(targetArea / np.pi) / 2

    # Be sure we are working with a polygon
    if geom.GetGeometryName() == "POLYGON":
        pass
    elif geom.GetGeometryName() == "MULTIPOLYGON":
        results = []
        for gi in range(geom.GetGeometryCount()):
            tmpResults = partition(
                geom.GetGeometryRef(gi), targetArea, growStep)
            results.extend(tmpResults)

        return results
    else:
        raise GeoKitGeomError(
            "Geometry is not a polygon or multipolygon object")

    # Check the geometry's size
    gArea = geom.Area()
    if gArea < 1.5 * targetArea:
        return [geom.Clone(), ]

    # Find the most starting boundary coordinate
    boundary = geom.Boundary()
    if boundary.GetGeometryCount() == 0:
        coords = boundary.GetPoints()
    else:
        coords = boundary.GetGeometryRef(0).GetPoints()

    y = np.array([c[1] for c in coords])
    x = np.array([c[0] for c in coords])

    if _startPoint == 0:  # start from the TOP-LEFT
        yStart = y.max()
        xStart = x[y == yStart].min()
    elif _startPoint == 1:  # Start from the LEFT-TOP
        xStart = x.min()
        yStart = y[x == xStart].max()
    elif _startPoint == 2:  # Start from the RIGHT-TOP
        xStart = x.max()
        yStart = y[x == xStart].max()
    elif _startPoint == 3:  # Start from the BOT-RIGHT
        yStart = y.min()
        xStart = x[y == yStart].max()
    elif _startPoint == 4:  # Start from the BOT-LEFT
        yStart = y.min()
        xStart = x[y == yStart].min()
    elif _startPoint == 5:  # Start from the LEFT-BOT
        xStart = x.min()
        yStart = y[x == xStart].min()
    else:
        raise GeoKitGeomError(
            "Start point failure. There may be an infinite loop in one of the geometries")

    start = point(xStart, yStart, srs=geom.GetSpatialReference())

    # start searching
    tmp = start.Buffer(growStep)
    tmp.Simplify(growStep)
    searchGeom = tmp.Intersection(geom)
    sgArea = searchGeom.Area()

    if gArea < 2 * targetArea:  # use a slightly smalled target area when the whole geometry
        #  is close to twice the target area in order to increase the
        #  liklihood of a usable leftover
        workingTarget = 0.9 * targetArea
    else:
        workingTarget = targetArea

    while sgArea < workingTarget:
        tmp = searchGeom.Buffer(growStep)
        tmp.Simplify(growStep)
        newGeom = tmp.Intersection(geom)
        newArea = newGeom.Area()

        if newArea > workingTarget * 1.1:
            dA = (newArea - sgArea) / growStep
            weightedGrowStep = (workingTarget - sgArea) / dA

            tmp = start.Buffer(weightedGrowStep)
            tmp.Simplify(growStep)
            searchGeom = tmp.Intersection(geom)
            break

        searchGeom = newGeom
        sgArea = newArea

    # fix the search geometry
    #  - For some reason the searchGeometry will sometime create a geometry and a linestring,
    #    in these cases the real geometry was always the second item...
    if searchGeom.GetGeometryName() == "GEOMETRYCOLLECTION":
        for gi in range(searchGeom.GetGeometryCount()):
            g = searchGeom.GetGeometryRef(gi)
            if g.GetGeometryName() == "POLYGON":
                searchGeom = g.Clone()
                break

    # Check the left over geometry, maybe some poops have been created and we need to glue them together
    outputGeom = searchGeom.Simplify(growStep / 20)
    geomToDo = []

    leftOvers = geom.Difference(searchGeom)
    if leftOvers.GetGeometryName() == "MULTIPOLYGON":
        for gi in range(leftOvers.GetGeometryCount()):
            leftOver = leftOvers.GetGeometryRef(gi)
            if leftOver.Area() < targetArea * 0.5:
                outputGeom = outputGeom.Union(leftOver)
            else:
                geomToDo.append(leftOver)
    elif leftOvers.GetGeometryName() == "POLYGON":
        geomToDo.append(leftOvers)
    else:
        raise GeoKitGeomError(
            "FATAL ERROR: Difference did not result in a polygon")

    # make an output array
    if outputGeom.Area() < targetArea * 1.5:
        output = [outputGeom]
    else:
        # the search geom plus some (or maybe all) of the left over poops total an area which is too large,
        #  so it will need recomputing. But in order to decrease the liklihhod of an infinite loop,
        #  use a difference starting point than the one used before
        #  - This will loop a maximum of 6 times before an exception is raised
        output = partition(outputGeom, targetArea, growStep, _startPoint + 1)

    for g in geomToDo:
        tmpOutput = partition(g, targetArea, growStep)
        output.extend(tmpOutput)

    # Done!
    return output
