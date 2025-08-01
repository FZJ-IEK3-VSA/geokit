geokit.core.geom
================

.. py:module:: geokit.core.geom


Attributes
----------

.. autoapisummary::

   geokit.core.geom.POINT
   geokit.core.geom.MULTIPOINT
   geokit.core.geom.LINE
   geokit.core.geom.MULTILINE
   geokit.core.geom.POLYGON
   geokit.core.geom.MULTIPOLYGON


Exceptions
----------

.. autoapisummary::

   geokit.core.geom.GeoKitGeomError


Classes
-------

.. autoapisummary::

   geokit.core.geom.Tile


Functions
---------

.. autoapisummary::

   geokit.core.geom.point
   geokit.core.geom.makePoint
   geokit.core.geom.box
   geokit.core.geom.tile
   geokit.core.geom.tileAt
   geokit.core.geom.subTiles
   geokit.core.geom.tileize
   geokit.core.geom.makeBox
   geokit.core.geom.polygon
   geokit.core.geom.makePolygon
   geokit.core.geom.line
   geokit.core.geom.makeLine
   geokit.core.geom.empty
   geokit.core.geom.makeEmpty
   geokit.core.geom.extractVerticies
   geokit.core.geom.convertWKT
   geokit.core.geom.convertGeoJson
   geokit.core.geom.polygonizeMatrix
   geokit.core.geom.polygonizeMask
   geokit.core.geom.transform
   geokit.core.geom.boundsToBounds
   geokit.core.geom.flatten
   geokit.core.geom.drawPoint
   geokit.core.geom.drawMultiPoint
   geokit.core.geom.drawLine
   geokit.core.geom.drawMultiLine
   geokit.core.geom.drawLinearRing
   geokit.core.geom.drawPolygon
   geokit.core.geom.drawMultiPolygon
   geokit.core.geom.drawGeoms
   geokit.core.geom.partition
   geokit.core.geom.shift
   geokit.core.geom.divideMultipolygonIntoEasternAndWesternPart
   geokit.core.geom.applyBuffer
   geokit.core.geom.fixOutOfBoundsGeoms


Module Contents
---------------

.. py:exception:: GeoKitGeomError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:data:: POINT

.. py:data:: MULTIPOINT

.. py:data:: LINE

.. py:data:: MULTILINE

.. py:data:: POLYGON

.. py:data:: MULTIPOLYGON

.. py:function:: point(*args, srs='latlon')

   Make a simple point geometry

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



.. py:function:: makePoint(*args, **kwargs)

   alias for geokit.geom.point(...)


.. py:function:: box(*args, srs=4326)

   Make an ogr polygon object from extents

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


.. py:function:: tile(xi, yi, zoom)

   Generates a box corresponding to a tile used for "slippy maps"

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



.. py:function:: tileAt(x, y, zoom, srs)

   Generates a box corresponding to a tile at the coordinates 'x' and 'y'
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



.. py:class:: Tile

   Bases: :py:obj:`tuple`


   .. py:attribute:: xi


   .. py:attribute:: yi


   .. py:attribute:: zoom


.. py:function:: subTiles(geom, zoom, checkIntersect=True, asGeom=False)

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


.. py:function:: tileize(geom, zoom)

   Deconstruct a given geometry into a set of tiled geometries

   Returns: Generator of ogr.Geometry objects


.. py:function:: makeBox(*args, **kwargs)

   alias for geokit.geom.box(...)


.. py:function:: polygon(outerRing, *args, srs='default')

   Creates an OGR Polygon obect from a given set of points

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


.. py:function:: makePolygon(*args, **kwargs)

   alias for geokit.geom.polygon(...)


.. py:function:: line(points, srs=4326)

   Creates an OGR Line obect from a given set of points

   Parameters:
   -----------
   Points : [(x,y), ], Nx2 numpy.ndarray or list of osgeo.ogr.Geometry points.
       The points defining the line

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the line to create

   Returns:
   --------
   ogr.Geometry


.. py:function:: makeLine(*args, **kwargs)

   alias for geokit.geom.line(...)


.. py:function:: empty(gtype, srs=None)

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


.. py:function:: makeEmpty(*args, **kwargs)

   alias for geokit.geom.empty(...)


.. py:function:: extractVerticies(geom)

   Get all verticies found on the geometry as a Nx2 numpy.ndarray


.. py:function:: convertWKT(wkt, srs=None)

   Make a geometry from a well known text (WKT) string

   Parameters:
   -----------
   wkt : str
       The WKT string to convert

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the geometry to create


.. py:function:: convertGeoJson(geojson, srs=3857)

   Make a geometry from a well known text (WKT) string
   TODO: UPDATE!!!
   Parameters:
   -----------
   wkt : str
       The WKT string to convert

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the geometry to create


.. py:function:: polygonizeMatrix(matrix, bounds=None, srs=None, flat=False, shrink=True, _raw=False)

   Create a geometry set from a matrix of integer values

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



.. py:function:: polygonizeMask(mask, bounds=None, srs=None, flat=True, shrink=True)

   Create a geometry set from a matrix mask

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



.. py:function:: transform(geoms, toSRS='europe_m', fromSRS=None, segment=None)

   Transform a geometry, or a list of geometries, from one SRS to another

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



.. py:function:: boundsToBounds(bounds, boundsSRS, outputSRS)

.. py:function:: flatten(geoms)

   Flatten a list of geometries into a single geometry object

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



.. py:function:: drawPoint(g, plotargs, ax, colorVal=None)

.. py:function:: drawMultiPoint(g, plotargs, ax, colorVal=None, skip=False)

.. py:function:: drawLine(g, plotargs, ax, colorVal=None, skip=False)

.. py:function:: drawMultiLine(g, plotargs, ax, colorVal=None)

.. py:function:: drawLinearRing(g, plotargs, ax, colorVal=None)

.. py:function:: drawPolygon(g, plotargs, ax, colorVal=None, skip=False)

.. py:function:: drawMultiPolygon(g, plotargs, ax, colorVal=None)

.. py:function:: drawGeoms(geoms, srs=4326, ax=None, simplificationFactor=5000, colorBy=None, figsize=(12, 12), xlim=None, ylim=None, fontsize=16, hideAxis=False, cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap='viridis', cbar=True, cbax=None, cbargs=None, leftMargin=0.01, rightMargin=0.01, topMargin=0.01, bottomMargin=0.01, **mplArgs)

   Draw geometries onto a matplotlib figure

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



.. py:function:: partition(geom, targetArea, growStep=None, _startPoint=0)

   Partition a Polygon into some number of pieces whose areas should be close
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


.. py:function:: shift(geom, lonShift=0, latShift=0)

   Shift a polygon in longitudinal and/or latitudinal direction.

   Inputs:
       geom : The geometry to be shifted
           - a single ogr Geometry object of POINT, LINESTRING, POLYGON or MULTIPOLYGON type
           - NOTE: Accepts only 2D geometries, z value must be zero.

       lonShift - (int, float) : The shift in longitudinal direction in units of the geom srs, may be positive or negative

       latShift - (int, float) : The shift in latitudinal direction in units of the geom srs, may be positive or negative

   Returns :
   --------
   osgeo.ogr.Geometry object of the input type with shifted coordinates


.. py:function:: divideMultipolygonIntoEasternAndWesternPart(geom, side='both')

   Multipolygons spanning the antimeridian (this includes polygons that are
   split at the antimeridian, with the Western half shifted Eastwards by 360°
   longitude) are separated into a part East and West of the antimeridian by
   identifying the largest longitudinal gap between any of the sub polygons and
   dividing the sub polys into one Eastern and one Western polygon list which
   is returned as multipolygons.
   NOTE: This function only works for already shifted subpolygons with an
   overall envelope betweeen -180° and +180° longitude.

   :param geom: The geometry to split. Must be a MultiPolygon.
   :type geom: ogr.Geometry
   :param side: 'left' or 'right' to return the left or right side of the antimeridian
                'main' to return the side with the largest area
                'both' to return both sides as a tuple (left, right)
   :type side: str, optional


.. py:function:: applyBuffer(geom, buffer, applyBufferInSRS=False, split='shift', tol=1e-06, verbose=False)

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


.. py:function:: fixOutOfBoundsGeoms(geom, how='shift')

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


