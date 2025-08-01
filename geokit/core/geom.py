import warnings
from collections import namedtuple
from copy import copy

import numpy as np
import pandas as pd
import smopy
from osgeo import gdal, ogr, osr

from geokit.core import srs as SRS
from geokit.core import util as UTIL


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


def point(*args, srs="latlon"):
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
            'Too many positional inputs. Did you mean to specify "srs="?'
        )

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
    if len(args) == 1:
        xMin, yMin, xMax, yMax = args[0]
    elif len(args) == 4:
        xMin, yMin, xMax, yMax = args
    else:
        raise GeoKitGeomError(
            'Incorrect number positional inputs (only accepts 1 or 4). Did you mean to specify "srs="?'
        )

    # make sure inputs are good
    xMin = float(xMin)
    xMax = float(xMax)
    yMin = float(yMin)
    yMax = float(yMax)

    assert xMin < xMax, f"xMin must be less than xMax"
    assert yMin < yMax, f"yMin must be less than yMax"

    # make box
    outBox = ogr.Geometry(ogr.wkbPolygon)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in [(xMin, yMin), (xMax, yMin), (xMax, yMax), (xMin, yMax), (xMin, yMin)]:
        ring.AddPoint(x, y)

    outBox.AddGeometry(ring)
    if not srs is None:
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

    o = SRS.xyTransform(
        [tl, br], fromSRS=SRS.EPSG4326, toSRS=SRS.EPSG3857, outputFormat="xy"
    )

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

    return tile(t.xi, t.yi, t.zoom)


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


def polygon(outerRing, *args, srs="default"):
    """Creates an OGR Polygon obect from a given set of points

    Parameters:
    -----------
    outerRing : [(x,y), ] or [ogr.Geometry, ] or Nx2 numpy.ndarray
        The polygon's outer edge

    *args : [(x,y), ] or [ogr.Geometry, ] or Nx2 numpy.ndarray
        The inner edges of the polygon
          * Each input forms a single edge
          * Inner rings cannot interset the outer ring or one another
          * NOTE! For proper drawing in matplotlib, inner rings must be given in
            the opposite orientation as the outer ring (clockwise vs
            counterclockwise)

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the polygon to create. By default "default", i.e. if
        point geometries are passed, srs will be extracted from first point of outer ring,
        if points are passed as (x, y) tuples, EPSG:4326 will be assigned
        by default unless given otherwise. If given as None, no srs will be assigned

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
    # check if we have all point geometries
    pointGeometries = all([isinstance(_p, ogr.Geometry) for _p in outerRing])
    if srs == "default":
        if pointGeometries:
            # we have geometries, set srs to the srs of the first outer ring point
            srs = outerRing[0].GetSpatialReference()
        else:
            # set srs to EPSG:4326 as standard
            srs = SRS.loadSRS(4326)
    elif srs is not None:
        srs = SRS.loadSRS(srs)

    # Make the complete geometry
    g = ogr.Geometry(ogr.wkbPolygon)
    if not srs is None:
        g.AssignSpatialReference(srs)

    # Make the outer ring
    otr = ogr.Geometry(ogr.wkbLinearRing)
    if not srs is None:
        otr.AssignSpatialReference(srs)
    # convert to tuples if we have point geometries at hand
    if pointGeometries:
        outerRing = [(_p.GetX(), _p.GetY()) for _p in outerRing]
    else:
        assert all(
            [isinstance(x, tuple) for x in outerRing]
        ), f"All outerRing entries must be (x,y) tuples in given (or default) srs."
    [otr.AddPoint(float(x), float(y)) for x, y in outerRing]
    g.AddGeometry(otr)

    # Make the inner rings (maybe)
    for innerRing in args:
        # extract (x, y) tuples first if needed
        if all([isinstance(_p, ogr.Geometry) for _p in innerRing]):
            innerRing = [(_p.GetX(), _p.GetY()) for _p in innerRing]
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
    Points : [(x,y), ], Nx2 numpy.ndarray or list of osgeo.ogr.Geometry points.
        The points defining the line

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the line to create

    Returns:
    --------
    ogr.Geometry
    """

    # Make the complete geometry
    g = ogr.Geometry(ogr.wkbLineString)
    if not srs is None:
        g.AssignSpatialReference(SRS.loadSRS(srs))

    # Make the line
    if all([isinstance(p, ogr.Geometry) for p in points]):
        # convert points into a list of coordinate tuples in correct srs
        points = [
            (transform(p, toSRS=srs).GetX(), transform(p, toSRS=srs).GetY())
            for p in points
        ]
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
    geom = ogr.CreateGeometryFromJson(geojson)  # Create new geometry from string
    if geom is None:  # test for success
        raise GeoKitGeomError("Failed to create geometry")
    if srs:
        geom.AssignSpatialReference(SRS.loadSRS(srs))  # Assign the given srs
    return geom


# 3
# Make a geometry from a matrix mask


def polygonizeMatrix(
    matrix, bounds=None, srs=None, flat=False, shrink=True, _raw=False
):
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
    if matrix.dtype == bool or matrix.dtype == np.uint8:
        dtype = "GDT_Byte"
    elif np.issubdtype(matrix.dtype, np.integer):
        dtype = "GDT_Int32"
    else:
        raise GeoKitGeomError("matrix must be a 2D boolean or integer numpy ndarray")

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
    driver = gdal.GetDriverByName("Mem")  # create a raster in memory
    raster = driver.Create("", cols, rows, 1, getattr(gdal, dtype))

    if raster is None:
        raise GeoKitGeomError("Failed to create temporary raster")

    raster.SetGeoTransform(
        (originX, abs(pixelWidth), 0, originY, 0, -1 * abs(pixelHeight))
    )

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

    # rasDS = createRaster(bounds=bounds, data=matrix, noDataValue=0, pixelWidth=pixelWidth, pixelHeight=pixelHeight, srs=srs)

    # Do a polygonize
    rasBand = raster.GetRasterBand(1)
    maskBand = rasBand.GetMaskBand()

    vecDS = gdal.GetDriverByName("Memory").Create("", 0, 0, 0, gdal.GDT_Unknown)
    vecLyr = vecDS.CreateLayer("mem", srs=srs)

    field = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(field)

    # Polygonize geometry
    result = gdal.Polygonize(rasBand, maskBand, vecLyr, 0)
    if result != 0:
        raise GeoKitGeomError("Failed to polygonize geometry")

    # Check how many features were created
    ftrN = vecLyr.GetFeatureCount()

    if ftrN == 0:
        # raise GlaesError("No features in created in temporary layer")
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
        geoms = [g.Buffer(float(shrinkFactor)) for g in geoms]

    # Do flatten, maybe
    if flat:
        geoms = np.array(geoms)
        rid = np.array(rid)

        finalGeoms = []
        finalRID = []
        for _rid in set(rid):
            smallGeomSet = geoms[rid == _rid]
            finalGeoms.append(
                flatten(smallGeomSet) if len(smallGeomSet) > 1 else smallGeomSet[0]
            )
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

    if not (mask.dtype == bool or mask.dtype == np.uint8):
        raise GeoKitGeomError("Mask must be a 2D boolean numpy ndarray")

    # Do vectorization
    result = polygonizeMatrix(
        matrix=mask, bounds=bounds, srs=srs, flat=flat, shrink=shrink, _raw=True
    )[0]
    if flat:
        result = result[0]

    # Done!
    return result


# geometry transformer


def transform(geoms, toSRS="europe_m", fromSRS=None, segment=None):
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
        geoms = [
            geoms,
        ]
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
    pts = flatten(
        [
            point(bounds[0], bounds[1], srs=boundsSRS),
            point(bounds[0], bounds[3], srs=boundsSRS),
            point(bounds[2], bounds[1], srs=boundsSRS),
            point(bounds[2], bounds[3], srs=boundsSRS),
        ]
    )
    pts.TransformTo(outputSRS)
    pts = extractVerticies(pts)

    bounds = (
        pts[:, 0].min(),
        pts[:, 1].min(),
        pts[:, 0].max(),
        pts[:, 1].max(),
    )
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
    while len(geoms) > 1:
        newGeoms = []
        for gi in range(0, len(geoms), 2):
            try:
                if not geoms[gi].IsValid():
                    warnings.warn("WARNING: Invalid Geometry encountered", UserWarning)
                if not geoms[gi + 1].IsValid():
                    warnings.warn("WARNING: Invalid Geometry encountered", UserWarning)

                newGeoms.append(geoms[gi].Union(geoms[gi + 1]))
            except IndexError:  # should only occur when length of geoms is odd
                newGeoms.append(geoms[gi])

        geoms = newGeoms
    return geoms[0]


##########################################################################
# Drawing functions
def drawPoint(g, plotargs, ax, colorVal=None):
    kwargs = dict(marker="o", color="#C32148", linestyle="None")
    if not colorVal is None:
        kwargs["color"] = colorVal

    kwargs.update(plotargs)
    return ax.plot(g.GetX(), g.GetY(), **kwargs)


def drawMultiPoint(g, plotargs, ax, colorVal=None, skip=False):
    kwargs = dict(marker=".", color="#C32148", linestyle="None")
    if not colorVal is None:
        kwargs["color"] = colorVal
    kwargs.update(plotargs)

    points = extractVerticies(g)
    return ax.plot(points[:, 0], points[:, 1], **kwargs)


def drawLine(g, plotargs, ax, colorVal=None, skip=False):
    if skip:
        kwargs = plotargs.copy()
    else:
        kwargs = dict(marker="None", color="k", linestyle="-")
        if not colorVal is None:
            kwargs["color"] = colorVal
        kwargs.update(plotargs)

    points = extractVerticies(g)
    return ax.plot(points[:, 0], points[:, 1], **kwargs)


def drawMultiLine(g, plotargs, ax, colorVal=None):
    kwargs = dict(marker="None", color="#007959", linestyle="-")
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
    from json import loads

    from descartes import PolygonPatch

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
        holes = [
            boundaries.GetGeometryRef(i).GetPoints()
            for i in range(1, boundaries.GetGeometryCount())
        ]

    patchData = dict(type="Polygon", coordinates=[])
    patchData["coordinates"].append(main)
    for hole in holes:
        patchData["coordinates"].append(hole)

    # Setup args
    if skip:
        kwargs = plotargs.copy()
    else:
        kwargs = dict(fc="#D9E9FF", ec="k", linestyle="-")
        if not colorVal is None:
            kwargs["fc"] = colorVal
        kwargs.update(plotargs)

    # Make patches
    mainPatch = PolygonPatch(patchData, **kwargs)
    return ax.add_patch(mainPatch)


def drawMultiPolygon(g, plotargs, ax, colorVal=None):
    kwargs = dict(fc="#D9E9FF", ec="k", linestyle="-")
    if not colorVal is None:
        kwargs["fc"] = colorVal
    kwargs.update(plotargs)

    h = []
    for gi in range(g.GetGeometryCount()):
        h.append(drawPolygon(g.GetGeometryRef(gi), kwargs, ax, colorVal, True))
    return h


def drawGeoms(
    geoms,
    srs=4326,
    ax=None,
    simplificationFactor=5000,
    colorBy=None,
    figsize=(12, 12),
    xlim=None,
    ylim=None,
    fontsize=16,
    hideAxis=False,
    cbarPadding=0.01,
    cbarTitle=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    cbar=True,
    cbax=None,
    cbargs=None,
    leftMargin=0.01,
    rightMargin=0.01,
    topMargin=0.01,
    bottomMargin=0.01,
    **mplArgs,
):
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

            ax = plt.axes(
                [
                    leftMargin,
                    bottomMargin,
                    1 - (rightMargin + leftMargin),
                    1 - (topMargin + bottomMargin),
                ]
            )
            cbax = None

        else:  # We need a colorbar
            rightMargin += 0.08  # Add area on the right for colorbar text
            if not hideAxis:
                leftMargin += 0.07

            cbarExtraPad = 0.05
            cbarWidth = 0.04

            ax = plt.axes(
                [
                    leftMargin,
                    bottomMargin,
                    1 - (rightMargin + leftMargin + cbarWidth + cbarPadding),
                    1 - (topMargin + bottomMargin),
                ]
            )

            cbax = plt.axes(
                [
                    1 - (rightMargin + cbarWidth),
                    bottomMargin + cbarExtraPad,
                    cbarWidth,
                    1 - (topMargin + bottomMargin + 2 * cbarExtraPad),
                ]
            )

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
        geoms = [
            geoms,
        ]

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
        transformed_geoms = []
        for gi, g in enumerate(geoms):
            gsrs = g.GetSpatialReference()
            if gsrs is None:
                continue  # Skip it if we don't know it...
            if not gsrs.IsSame(srs):
                transformed_geoms.append(transform(geoms[gi], srs))
            else:
                transformed_geoms.append(geoms[gi])
        geoms = np.asarray(transformed_geoms)

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

        simplificationValue = max(xMax - xMin, yMax - yMin) / simplificationFactor

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

        _colorVals = [cmap(v) for v in (colorVals - cValMin) / (cValMax - cValMin)]

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
            msg = (
                "Could not draw geometry of type:",
                pargs.index[gi],
                "->",
                g.GetGeometryName(),
            )
            warnings.warn(msg, UserWarning)

    # Add the colorbar, maybe
    if not colorBy is None and cbar:
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=cValMin, vmax=cValMax)
        tmp = dict(cmap=cmap, norm=norm, orientation="vertical")
        if not cbargs is None:
            tmp.update(cbargs)
        cbar = ColorbarBase(cbax, **tmp)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(
            colorBy if cbarTitle is None else cbarTitle, fontsize=fontsize + 2
        )
    else:
        cbar = None

    # Do some formatting
    if newAxis:
        ax.set_aspect("equal")
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
            tmpResults = partition(geom.GetGeometryRef(gi), targetArea, growStep)
            results.extend(tmpResults)

        return results
    else:
        raise GeoKitGeomError("Geometry is not a polygon or multipolygon object")

    # Check the geometry's size
    gArea = geom.Area()
    if gArea < 1.5 * targetArea:
        return [
            geom.Clone(),
        ]

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
            "Start point failure. There may be an infinite loop in one of the geometries"
        )

    start = point(xStart, yStart, srs=geom.GetSpatialReference())

    # start searching
    tmp = start.Buffer(float(growStep))
    tmp.Simplify(growStep)
    searchGeom = tmp.Intersection(geom)
    sgArea = searchGeom.Area()

    if (
        gArea < 2 * targetArea
    ):  # use a slightly smalled target area when the whole geometry
        #  is close to twice the target area in order to increase the
        #  liklihood of a usable leftover
        workingTarget = 0.9 * targetArea
    else:
        workingTarget = targetArea

    while sgArea < workingTarget:
        tmp = searchGeom.Buffer(float(growStep))
        tmp.Simplify(growStep)
        newGeom = tmp.Intersection(geom)
        newArea = newGeom.Area()

        if newArea > workingTarget * 1.1:
            dA = (newArea - sgArea) / growStep
            weightedGrowStep = (workingTarget - sgArea) / dA

            tmp = start.Buffer(float(weightedGrowStep))
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
        raise GeoKitGeomError("FATAL ERROR: Difference did not result in a polygon")

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


def shift(geom, lonShift=0, latShift=0):
    """Shift a polygon in longitudinal and/or latitudinal direction.

    Inputs:
        geom : The geometry to be shifted
            - a single ogr Geometry object of POINT, LINESTRING, POLYGON or MULTIPOLYGON type
            - NOTE: Accepts only 2D geometries, z value must be zero.

        lonShift - (int, float) : The shift in longitudinal direction in units of the geom srs, may be positive or negative

        latShift - (int, float) : The shift in latitudinal direction in units of the geom srs, may be positive or negative

    Returns :
    --------
    osgeo.ogr.Geometry object of the input type with shifted coordinates
    """
    if not isinstance(geom, ogr.Geometry):
        raise TypeError(f"geom must be of type osgeo.ogr.Geometry")
    # first get srs of input geom
    _srs = geom.GetSpatialReference()
    # get the dimension of the geometry
    dims = geom.GetCoordinateDimension()
    assert dims in [2, 3], f"Only 2D and 3D points are supported, but got {dims}D"

    # define sub method to shift collection of single points
    def _movePoints(pointCollection, lonShift, latShift):
        """Auxiliary function shifting individual points"""
        points = list()
        for i in range(len(str(pointCollection).split(","))):
            points.append(pointCollection.GetPoint(i))
        # shift the points individually
        points_shifted = list()
        for p in points:
            assert p[2] == 0, f"All z-values must be zero!"
            points_shifted.append((p[0] + lonShift, p[1] + latShift))
        return points_shifted

    # first check if geom is a point and shift
    if "POINT" in geom.GetGeometryName():
        p = geom.GetPoint()
        point_shifted = point((p[0] + lonShift, p[1] + latShift), srs=_srs)
        if dims == 2:
            point_shifted.FlattenTo2D()
        return point_shifted
    # else check if line and adapt
    elif (
        "LINESTRING" in geom.GetGeometryName()
        and not "MULTILINE" in geom.GetGeometryName()
    ):
        assert not geom.IsEmpty(), f"Line is empty"
        line_shifted = line(
            _movePoints(pointCollection=geom, lonShift=lonShift, latShift=latShift),
            srs=_srs,
        )
        if dims == 2:
            line_shifted.FlattenTo2D()
        return line_shifted
    # else check if we have a (multi)polygon
    elif "POLYGON" in geom.GetGeometryName():
        assert not geom.IsEmpty(), f"Polygon is empty"
        if not "MULTIPOLYGON" in geom.GetGeometryName():
            # only a simple polygon, generate single entry list to allow iteration
            geom = [geom]
        # iterate over individual polygons
        for ip, poly in enumerate(geom):
            assert (
                "POLYGON" in poly.GetGeometryName()
            ), f"MULTIPOLYGON is not composed of only POLYGONS"
            # iterate over sub linear rings
            for ir, ring in enumerate(poly):
                assert (
                    "LINEARRING" in ring.GetGeometryName()
                ), f"POLYGON (or sub polygon of MULTIPOLYGON) is not composed of only LINEARRINGS"
                poly_shifted = polygon(
                    _movePoints(
                        pointCollection=ring, lonShift=lonShift, latShift=latShift
                    ),
                    srs=_srs,
                )
                if ip == 0 and ir == 0:
                    multi_shifted = poly_shifted
                else:
                    multi_shifted = multi_shifted.Union(poly_shifted)
        # the shifted polygon should have the same dimensions as the input
        if dims == 2:
            multi_shifted.FlattenTo2D()
        return multi_shifted
    else:
        raise TypeError(
            f"geom must be a 'POINT', 'LINESTRING', 'POLYGON' or 'MULTIPOLYGON' osgeo.ogr.Geometry"
        )


def divideMultipolygonIntoEasternAndWesternPart(geom, side="both"):
    """
    Multipolygons spanning the antimeridian (this includes polygons that are
    split at the antimeridian, with the Western half shifted Eastwards by 360°
    longitude) are separated into a part East and West of the antimeridian by
    identifying the largest longitudinal gap between any of the sub polygons and
    dividing the sub polys into one Eastern and one Western polygon list which
    is returned as multipolygons.
    NOTE: This function only works for already shifted subpolygons with an
    overall envelope betweeen -180° and +180° longitude.

    Parameters
    ----------
    geom: ogr.Geometry
        The geometry to split. Must be a MultiPolygon.
    side: str, optional
        'left' or 'right' to return the left or right side of the antimeridian
        'main' to return the side with the largest area
        'both' to return both sides as a tuple (left, right)
    """
    # check inputs
    assert side in (
        "both",
        "left",
        "right",
        "main",
    ), "side must be 'left', 'right', 'main' or 'both'"
    assert isinstance(geom, ogr.Geometry), "geom must be of type osgeo.ogr.Geometry"
    assert geom.GetGeometryName() == "MULTIPOLYGON", "Only MultiPolygon supported"
    assert geom.GetSpatialReference().IsSame(
        SRS.loadSRS(4326)
    ), "geometry must be in EPSG:4326"
    assert (
        geom.GetEnvelope()[0] >= -180 and geom.GetEnvelope()[1] <= 180
    ), "Envelope must be between -180° and +180° longitude"
    assert geom.GetSpatialReference().IsSame(
        SRS.loadSRS(4326)
    ), "Only EPSG:4326 lat/lon supported"

    # first extract sub polygons
    sub_polys = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]

    # get all the bounding boxes
    bounds = []
    for poly in sub_polys:
        env = poly.GetEnvelope()  # (minX, maxX, minY, maxY)
        bounds.append((env[0], env[1], poly))  # store (minX, maxX, polygon)

    # sort them from left to right based on minX
    bounds.sort(key=lambda x: x[0])

    # find the largest gap iteratively
    max_gap = 0
    split_index = 0
    curr_maxs = list()
    for i in range(len(bounds) - 1):
        curr_maxs.append(bounds[i][1])
        curr_max = max(curr_maxs)
        next_min = bounds[i + 1][0]
        gap = next_min - curr_max
        if gap > max_gap:
            # overwrite the max gap so far and save the index when it occurs
            max_gap = gap
            split_index = i

    # split into two sets of geoms - left and right of the gap (do only if necessary for time)
    if side in ["both", "main", "right"]:
        # filter only sub polys below (or equal) split index
        right_polys = [b[2] for i, b in enumerate(bounds) if i <= split_index]
        # merge all right polygons into a single multipolygon and assign the same spatial reference
        right_multi = ogr.Geometry(ogr.wkbMultiPolygon)
        for poly in right_polys:
            right_multi.AddGeometry(poly.Clone())
        right_multi.AssignSpatialReference(geom.GetSpatialReference())
    # same for the left side
    if side in ["both", "main", "left"]:
        # only polys above split index
        left_polys = [b[2] for i, b in enumerate(bounds) if i > split_index]
        left_multi = ogr.Geometry(ogr.wkbMultiPolygon)
        for poly in left_polys:
            left_multi.AddGeometry(poly.Clone())
        left_multi.AssignSpatialReference(geom.GetSpatialReference())

    if side == "left":
        return left_multi
    elif side == "right":
        return right_multi
    elif side == "both":
        # return both left and right
        return left_multi, right_multi
    elif side == "main":
        # return the side with the largest area
        left_area = left_multi.GetArea()
        right_area = right_multi.GetArea()
        if left_area > right_area:
            return left_multi
        else:
            return right_multi
    else:
        raise ValueError("side must be 'left', 'right', 'main' or None")


def applyBuffer(
    geom, buffer, applyBufferInSRS=False, split="shift", tol=1e-6, verbose=False
):
    """
    This function applies a buffer to any geom, avoiding edge issues with geoms
    near the SRS bounds. By shifting the geom to a zero longitude, geometry
    distortions are avoided when the buffered geom exceeds the bounds (i.e.
    antimeridian or latitudes of +/-90° e.g. in the case of EPSG:4326). Buffered
    geom areas extending over the bounds can either be clipped off or shifted to
    the respective "other end of the map". If the buffer is applied in a
    different (e.g. metric) EPSG, latitudinal overlaps will always be clipped.

    geom : osgeo.ogr.Geometry
        Geometry to be buffered.
    buffer : int, float
        The buffer value to be applied to the geom, in unit of the SRS unless
        'bufferInEPSG6933' is True, then always in meters.
    applyBufferInSRS : int, osgeo.osr.SpatialReference, optional
        Allows to specify an EPSG integer code or an osgeo.osr.SpatialReference
        instance to define the SRS in which the buffer will be applied, then in
        the unit of the specified EPSG. If e.g. 6933 is given, the buffer will
        be applied in meters in a metric system. By default False, i.e. the
        original SRS of the geom will be used.
        NOTE: 'Lambert_Azimuthal_Equal_Area' or 'Lambert_Conformal_Conic_2SP'
        projections are not allowed here, use e.g. EPSG:6933 as global metric SRS.
    split : str, optional
        'shift' : shift areas that exceed the antimeridian line to the other end (default)
        'clip' : remove/clip polygon parts that exceed the antimeridian
        'none' : do not split geoms at all that cross the antimeridian
    tol : int, float, optional
        Geoms protruding over the +/-90° latitude line will be clipped to 90°
        plus/minus this tolerance in degrees to avoid geometry issues due to
        distortions during SRS transformation. By default 1E-6.
    verbose : boolean, optional
        If True, additional notifications will be printed when geometry has to
        be clipped to enable retransformation to initial SRS. By default False.
    """
    if not applyBufferInSRS is False:
        try:
            applyBufferInSRS = SRS.loadSRS(applyBufferInSRS)
        except:
            raise ValueError(
                f"applyBufferInSRS {applyBufferInSRS} is not a known SRS to geokit.srs.loadSRS()"
            )
        assert not applyBufferInSRS.GetAttrValue("PROJECTION") in [
            "Lambert_Azimuthal_Equal_Area",
            "Lambert_Conformal_Conic_2SP",
        ], f"SRS projection must not be in: 'Lambert_Azimuthal_Equal_Area', 'Lambert_Conformal_Conic_2SP'"

    # first shift the geom to the "center of the world"
    geom_shftd = shift(
        geom,
        lonShift=-geom.Centroid().GetX(),
        latShift=(
            -geom.Centroid().GetY() if applyBufferInSRS is False else 0
        ),  # latitudinal shift would distort latlon-to-metric conversion
    )
    if applyBufferInSRS:
        # transform to applyBufferInSRS srs
        geom_shftd_epsg = transform(geom_shftd, toSRS=applyBufferInSRS)
        # apply buffer
        geom_shftd_buf_epsg = geom_shftd_epsg.Buffer(buffer)
        assert (
            geom_shftd_buf_epsg.IsValid()
        ), f"geom in EPSG:{applyBufferInSRS} invalid after buffering."
        # clip to +/-90° lat "world window" (shrink window by tolerance and transform to EPSG)
        _worldbox_epsg = transform(
            polygon(
                [
                    (-180 + tol, -90 + tol),
                    (-180 + tol, 90 - tol),
                    (180 - tol, 90 - tol),
                    (180 - tol, -90 + tol),
                ],
                srs=4326,
            ),
            toSRS=applyBufferInSRS,
        )
        _env = geom_shftd_buf_epsg.GetEnvelope()
        if (
            _worldbox_epsg.GetEnvelope()[2] < _env[2]
            or _worldbox_epsg.GetEnvelope()[3] > _env[3]
        ):
            # the vertical dimension exceeds the "world window", clip to worldbox to avoid geoms extending over +/-90° lat
            geom_shftd_buf_epsg = geom_shftd_buf_epsg.Intersection(_worldbox_epsg)
            _env_new = geom_shftd_buf_epsg.GetEnvelope()
            if verbose:
                print(f"NOTE: geometry was clipped vertically.", flush=True)
            if _env_new[0] < _env[0] or _env_new[1] < _env[1] and not split == "clip":
                # longitudinal clip, this is not supposed to happen since the geom has been shifted longitudinally!
                warnings.warn("geometry was clipped horizontally!", Warning)
        # reconvert to original SRS, still centered on (0,0)
        geom_shftd_buf = transform(
            geom_shftd_buf_epsg, toSRS=geom.GetSpatialReference()
        )
        # geom is sometimes invalid after transformation, if so try to fix with zero-buffer trick
        if not geom_shftd_buf.IsValid():
            geom_shftd_buf = geom_shftd_buf.Buffer(0)
        assert (
            geom_shftd_buf.IsValid()
        ), f"buffered geom invalid after re-transformation to initial SRS."
    else:
        # apply buffer in unit of geom SRS
        geom_shftd_buf = geom_shftd.Buffer(buffer)
    # shift back to the original centroid location
    geom_buf = shift(
        geom_shftd_buf,
        lonShift=geom.Centroid().GetX(),
        latShift=(
            geom.Centroid().GetY() if applyBufferInSRS is False else 0
        ),  # same as above
    )
    assert (
        geom_buf.IsValid()
    ), f"buffered geom in initial SRS invalid after shifting it back to initial longitude."
    if split in ["none", None]:
        # no splitting of protruding geoms required, return as is
        return geom_buf
    elif split == "clip":
        # clip protruding elements
        return fixOutOfBoundsGeoms(geom=geom_buf, how="clip")
    elif split == "shift":
        # split and shift and re-merge protruding polygon parts
        return fixOutOfBoundsGeoms(geom=geom_buf, how="shift")
    else:
        raise ValueError(f"split argument must be either 'none', 'clip' or 'shift'.")


def fixOutOfBoundsGeoms(geom, how="shift"):
    """
    This function allows to deal with polygons that protrude over the SRS bounds
    at +/-180° longitude respectively +/-90° latitude. Polygon areas that exceed
    those bounds are either clipped or shifted to the "opposite end of the map"
    with shapes at the poles being inverted and shifted by 180° to create a
    "fold-over" effect.

    geom : osgeo-ogr.Geometry
        Geometry to fix.
    how : str, optional
        The way how to deal with sub shapes extending over the bounds:
        'shift' :   split off and shift to the "opposite end of the map"
        'clip' :    clip and remove extending shapes completely
    """
    assert isinstance(geom, ogr.Geometry), f"geom must be an osgeo.ogr.Geometry"
    assert how in ["clip", "shift"], f"how must be in 'clip', 'shift'"
    # get the envelope and srs of original geom
    env = geom.GetEnvelope()
    _srs = geom.GetSpatialReference()
    assert _srs.IsSame(SRS.loadSRS(4326)), f"SRS must be EPSG:4326"

    if env[0] >= -180 and env[1] <= 180 and env[2] >= -90 and env[3] <= 90:
        # the polygon is completely within bounds already, return as is
        return geom

    # else we need to clip and shift/rotate
    geom_fixed = copy(geom)
    # HORIZONTZAL BOUNDS
    if env[0] < -180 or env[1] > 180:
        # we need to clip & shift in HORIZONTAL direction
        basebox_tripleheight = polygon(
            [(-180, -90 * 3), (-180, 90 * 3), (180, 90 * 3), (180, -90 * 3)], srs=4326
        )
        geom_fixed = geom_fixed.Intersection(
            basebox_tripleheight
        )  # clip off outer parts
        if env[0] < -180 and how == "shift":
            # extends over left edge, get the left part, shift and merge
            left_part = shift(basebox_tripleheight, lonShift=-360).Intersection(geom)
            if geom_fixed.IsEmpty():
                geom_fixed = shift(
                    left_part, lonShift=360
                )  # overwrite when no center part
            else:
                geom_fixed = geom_fixed.Union(shift(left_part, lonShift=360))
        if env[1] > 180 and how == "shift":
            # extends over right edge, get the right part, shift and merge
            right_part = shift(basebox_tripleheight, lonShift=+360).Intersection(geom)
            if geom_fixed.IsEmpty():
                geom_fixed = shift(
                    right_part, lonShift=-360
                )  # overwrite when no center part
            else:
                geom_fixed = geom_fixed.Union(shift(right_part, lonShift=-360))
    # VERTICAL BOUNDS
    if env[2] < -90 or env[3] > 90:
        # we need to clip in VERTICAL direction and rotate (horizontal issue are fixed already)
        def fold_over_pole(geom):
            """Aux function to fold a geometry over the +/-90° latitude line and rotate it."""
            env = geom.GetEnvelope()
            center_lon = (env[0] + env[1]) / 2  # x value of center axis of whole geom

            def fold_polygon(polygon):
                """Function that folds a polygon geometrie ofer 90° lat line"""

                def _fold_ring(ring):
                    """core function for every linear ring"""
                    new_ring = ogr.Geometry(ogr.wkbLinearRing)
                    for i in range(ring.GetPointCount()):
                        x, y, *z = ring.GetPoint(i)
                        # if needed, mirror y at +/-90° line and flip x value around center axis + shift by 180° (fold over)
                        if -90 <= y <= 90:  # all good
                            y_new = y
                            x_new = x
                        else:  # rotate and flip x values, fold y values
                            _x_new = x + 2 * (center_lon - x)
                            x_new = (_x_new + 180) % 360
                            if y > 90:
                                y_new = 180 - y
                                y_new = min(
                                    y_new, 90.0 - 1e-6
                                )  # avoid that geoms touch eachother at the pole
                            elif y < -90:
                                y_new = -180 - y
                                y_new = max(
                                    y_new, -90.0 + 1e-6
                                )  # avoid that geoms touch eachother at the pole
                        new_ring.AddPoint(x_new, y_new)  # create new ring point
                    return new_ring

                new_geom = ogr.Geometry(ogr.wkbPolygon)
                outer_ring = _fold_ring(polygon.GetGeometryRef(0))
                new_geom.AddGeometry(outer_ring)
                # now do this for every potential other (inner) ring
                for i in range(1, polygon.GetGeometryCount()):
                    inner_ring = _fold_ring(polygon.GetGeometryRef(i))
                    new_geom.AddGeometry(inner_ring)
                return new_geom

            if geom.GetGeometryName() == "POLYGON":
                # apply function directly
                new_geom = fold_polygon(geom)
            elif geom.GetGeometryName() == "MULTIPOLYGON":
                new_geom = ogr.Geometry(ogr.wkbMultiPolygon)
                # fold iteratively every single sub poly
                for i in range(geom.GetGeometryCount()):
                    sub_geom = geom.GetGeometryRef(i)
                    rotated = fold_polygon(sub_geom)
                    new_geom.AddGeometry(rotated)
            else:
                raise NotImplementedError(
                    f"Geometry type '{geom.GetGeometryName()}' not supported"
                )
            # assign SRS and return
            new_geom.AssignSpatialReference(geom.GetSpatialReference())
            return new_geom

        basebox = polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)], srs=4326)
        geom_fixed_horizontally = copy(geom_fixed)
        geom_fixed = geom_fixed.Intersection(basebox)
        if env[2] < -90 and how == "shift":
            # clip off the bottom part and rotate to the other side (by 180°)
            bottom_part = shift(basebox, latShift=-180).Intersection(
                geom_fixed_horizontally
            )
            bottom_part_shifted = fold_over_pole(bottom_part)
            geom_fixed = geom_fixed.Union(bottom_part_shifted)
        if env[3] > 90 and how == "shift":
            # clip off the top part and rotate to the other side (by 180°)
            top_part = shift(basebox, latShift=+180).Intersection(
                geom_fixed_horizontally
            )
            top_part_shifted = fold_over_pole(top_part)
            geom_fixed = geom_fixed.Union(top_part_shifted)
    # assign srs and return
    geom_fixed.AssignSpatialReference(_srs)
    return geom_fixed
