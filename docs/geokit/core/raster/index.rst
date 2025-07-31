geokit.core.raster
==================

.. py:module:: geokit.core.raster


Attributes
----------

.. autoapisummary::

   geokit.core.raster.COMPRESSION_OPTION
   geokit.core.raster._gdalIntToType
   geokit.core.raster._gdalType


Exceptions
----------

.. autoapisummary::

   geokit.core.raster.GeoKitRasterError


Classes
-------

.. autoapisummary::

   geokit.core.raster.RasterInfo
   geokit.core.raster.value


Functions
---------

.. autoapisummary::

   geokit.core.raster.loadRaster
   geokit.core.raster.gdalType
   geokit.core.raster.createRaster
   geokit.core.raster.createRasterLike
   geokit.core.raster.saveRasterAsTif
   geokit.core.raster.extractMatrix
   geokit.core.raster.rasterStats
   geokit.core.raster.gradient
   geokit.core.raster.isFlipped
   geokit.core.raster.rasterInfo
   geokit.core.raster.extractValues
   geokit.core.raster.interpolateValues
   geokit.core.raster.mutateRaster
   geokit.core.raster.indexToCoord
   geokit.core.raster.drawSmopyMap
   geokit.core.raster.drawRaster
   geokit.core.raster.polygonizeRaster
   geokit.core.raster.contours
   geokit.core.raster.warp
   geokit.core.raster.sieve
   geokit.core.raster.rasterCellNo


Module Contents
---------------

.. py:exception:: GeoKitRasterError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:data:: COMPRESSION_OPTION
   :value: ['COMPRESS=LZW']


.. py:function:: loadRaster(source: str | osgeo.gdal.Dataset, mode=0) -> osgeo.gdal.Dataset

   Load a raster dataset from a path to a file on disc

   Parameters:
   -----------
   source : str or gdal.Dataset
       * If a string is given, it is assumed as a path to a raster file on disc
       * If a gdal.Dataset is given, it is assumed to already be an open raster
         and is returned immediately

   Returns:
   --------
   gdal.Dataset



.. py:data:: _gdalIntToType

.. py:data:: _gdalType

.. py:function:: gdalType(s)

   Tries to determine gdal datatype from the given input type


.. py:function:: createRaster(bounds, output: None | str | pathlib.Path = None, pixelWidth=100, pixelHeight=100, dtype=None, srs='europe_m', compress=True, noData=None, overwrite: bool = True, fill=None, data=None, meta=None, scale=1, offset=0, creationOptions=dict(), **kwargs)

   Create a raster file

   NOTE:
   -----
   Raster datasets are always written in the 'yAtTop' orientation. Meaning that
   the first row of data values (either written to or read from the dataset) will
   refer to the TOP of the defined boundary, and will then move downward from
   there

   If a data matrix is given, and a negative pixelWidth is defined, the data
   will be flipped automatically

   Parameters:
   -----------
   bounds : (xMin, yMix, xMax, yMax) or Extent
       The geographic extents spanned by the raster

   pixelWidth : numeric
       The pixel width of the raster in units of the input srs
       * The keyword 'dx' can be used as well and will override anything given
       assigned to 'pixelWidth'

   pixelHeight : numeric
       The pixel height of the raster in units of the input srs
       * The keyword 'dy' can be used as well and will override anything given
         assigned to 'pixelHeight'

   output : str; optional
       A path to an output file
       * If output is None, the raster will be created in memory and a dataset
         handel will be returned
       * If output is given, the raster will be written to disk and nothing will
         be returned

   dtype : str; optional
       The datatype of the represented by the created raster's band
       * Options are: Byte, Int16, Int32, Int64, Float32, Float64
       * If dtype is None and data is None, the assumed datatype is a 'Byte'
       * If dtype is None and data is not None, the datatype will be inferred
         from the given data

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the point to create
         * If not given, longitude/latitude is assumed
         * srs MUST be given as a keyword argument
       * If 'bounds' is an Extent object, the bounds' internal srs will override
         this input

   compress : bool
       A flag instructing the output raster to use a compression algorithm
       * only useful if 'output' has been defined
       * "DEFLATE" used for Linux/Mac, "LZW" used for Windows

   noData : numeric; optional
       Specifies which value should be considered as 'no data' in the created
       raster
       * Must be the same datatye as the 'dtype' input (or that which is derived)

   fill : numeric; optional
       The initial value given to all pixels in the created raster band
       - numeric
       * Must be the same datatye as the 'dtype' input (or that which is derived)

   overwrite : bool
       A flag to overwrite a pre-existing output file
       * If set to False and an 'output' is specified which already exists,
         an error will be raised

   data : matrix_like
       A 2D matrix to write into the resulting raster
       * array dimensions must fit raster dimensions as calculated by the bounds
         and the pixel resolution

   scale : numeric; optional
       The scaling value given to apply to all values
       - numeric
       * Must be the same datatye as the 'dtype' input (or that which is derived)

   offset : numeric; optional
       The offset value given to apply to all values
       - numeric
       * Must be the same datatye as the 'dtype' input (or that which is derived)

   Returns:
   --------
   * If 'output' is None: gdal.Dataset
   * If 'output' is a string: The path to the output is returned (for easy opening)



.. py:function:: createRasterLike(source, copyMetadata=True, metadata=None, **kwargs)

   Create a raster described by the given raster info (as returned from a
   call to rasterInfo() ).

   * This copies all characteristics of the given raster, including: bounds,
     pixelWidth, pixelHeight, dtype, srs, noData, and meta.
   * Any keyword argument which is given will override values found in the
     source



.. py:function:: saveRasterAsTif(source, output, **kwargs)

   Write a osgeo.gdal.Dataset in memory to a GeoTiff file to disk.

   :param source:
   :type source: osgeo.gdal.Dataset
   :param output: A path to an output file
   :type output: str

   :returns: Path to the saved file on disk.
   :rtype: str


.. py:function:: extractMatrix(source, bounds=None, boundsSRS='latlon', maskBand: bool = False, autocorrect: bool = False, returnBounds: bool = False) -> numpy.ndarray | tuple[numpy.ndarray, tuple[float, float, float, float] | None]

   extract all or part of a raster's band as a numpy matrix

   Note:
   -----
   Unless one is trying to get the entire matrix from the raster dataset, usage
   of this function requires intimate knowledge of the raster's characteristics.
   In such a case it is probably easier to use Extent.extractMatrix

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource

   bounds: tuple or Extent
       The boundary to clip the raster to before mutating
       * If given as an Extent, the extent is always cast to the source's
           - native srs before mutating
       * If given as a tuple, (xMin, yMin, xMax, yMax) is expected
           - Units must be in the srs specified by 'boundsSRS'
       * This boundary must fit within the boundary of the rasters source
       * The boundary is always fitted to the source's grid, so the returned
         values do not necessarily match to the boundary which is provided

   boundsSRS: Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the 'bounds' argument
       * This is ignored if the 'bounds' argument is an Extent object or is None

   autocorrect : bool; optional
       If True, the matrix will search for no data values and change them to
       numpy.nan
       * Data type will always result in a float, so be careful with large
         matricies

   returnBounds : bool; optional
       If True, return the computed bounds along with the matrix data

   Returns:
   --------
   * If returnBounds is False: numpy.ndarray -> Two dimensional matrix
   * If returnBounds is True: (numpy.ndarray, tuple)
       - ndarray is matrix data
       - tuple is the (xMin, yMin, xMax, yMax) of the computed bounds



.. py:function:: rasterStats(source, cutline=None, ignoreValue=None, **kwargs)

   Compute basic statistics of the values contained in a raster dataset.

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource

   cutline : ogr.Geometry; optional
       The geometry over which to cut out the raster's data
       * Must be a Polygon or MultiPolygon

   ignoreValue : numeric
       A value to ignore when computing the statistics
       * If the raster source has a 'no Data' value, it is automatically
         ignored

   **kwargs
       * All kwargs are passed on to warp() when 'geom' is given
       * See gdal.WarpOptions for more details
       * For example, 'allTouched' may be useful

   Returns:
   --------
   Results from a call to scipy.stats.describe



.. py:function:: gradient(source, mode='total', factor=1, asMatrix=False, **kwargs)

   Calculate a raster's gradient and return as a new dataset or simply a matrix

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource

   mode : str; optional
       Determines the type of gradient to compute
       * Options are....
         "total" : Calculates the absolute gradient as a ratio

         "slope" : Same as 'total'

         "north-south" : Calculates the "north-facing" gradient as a ratio where
                         negative numbers indicate a south facing gradient

         "east-west" : Calculates the "east-facing" gradient as a ratio where
                       negative numbers indicate a west facing gradient

         "aspect" : calculates the gradient's direction in radians (0 is east)

         "dir" : same as 'aspect'

   factor : numeric or 'latlonToM'
       The scaling factor relating the units of the x & y dimensions to the z
       dimension
       * If factor is 'latlonToM', the x & y units are assumed to be degrees
         (lat & lon) and the z units are assumed to be meters. A factor is then
         computed for coordinates at the source's center.
       * Example: If x,y units are meters and z units are feet, factor should
         be 0.3048

   asMatrix : bool
       If True, makes the returned value a matrix
       If False, makes the returned value a raster dataset

   **kwargs : All extra key word arguments are passed on to a final call to
       'createRaster'
       * Only useful when 'asMatrix' is True

   Returns:
   --------
   * If 'asMatrix' is True: numpy.ndarray
   * If 'asMatrix' is False: gdal.Dataset



.. py:function:: isFlipped(source)

.. py:class:: RasterInfo

   Bases: :py:obj:`tuple`


   .. py:attribute:: srs


   .. py:attribute:: dtype


   .. py:attribute:: flipY


   .. py:attribute:: yAtTop


   .. py:attribute:: bounds


   .. py:attribute:: xMin


   .. py:attribute:: yMin


   .. py:attribute:: xMax


   .. py:attribute:: yMax


   .. py:attribute:: dx


   .. py:attribute:: dy


   .. py:attribute:: pixelWidth


   .. py:attribute:: pixelHeight


   .. py:attribute:: noData


   .. py:attribute:: xWinSize


   .. py:attribute:: yWinSize


   .. py:attribute:: meta


   .. py:attribute:: source


   .. py:attribute:: scale


   .. py:attribute:: offset


.. py:function:: rasterInfo(sourceDS) -> RasterInfo

   Returns a named tuple containing information relating to the input raster

   Returns:
   --------
   namedtuple -> ( srs: The spatial reference system (as an OGR object)
                   dtype: The datatype
                   flipY: A flag which indicates that the raster starts at the
                            'bottom' as opposed to at the 'top'
                   bounds: The (xMin, yMin, xMax, and yMax) values as a tuple
                   xMin: The minimal X boundary
                   yMin:The minimal Y boundary
                   xMax:The maximal X boundary
                   yMax: The maximal Y boundary
                   pixelWidth: The raster's pixelWidth
                   pixelHeight: The raster's pixelHeight
                   dx:The raster's pixelWidth
                   dy: The raster's pixelHeight
                   noData: The noData value used by the raster
                   scale: The scale value used by the raster
                   offset: The offset value used by the raster
                   xWinSize: The width of the raster is pixels
                   yWinSize: The height of the raster is pixels
                   meta: The raster's meta data )


.. py:class:: value

   Bases: :py:obj:`tuple`


   .. py:attribute:: data


   .. py:attribute:: xOffset


   .. py:attribute:: yOffset


   .. py:attribute:: inBounds


.. py:function:: extractValues(source, points, pointSRS='latlon', winRange=0, noDataOkay=True, _onlyValues=False)

   Extracts the value of a raster at a given point or collection of points.
      Can also extract a window of values if desired

   * If the given raster is not in the 'flipped-y' orientation, the result will
     be automatically flipped

   Notes:
   ------
   Generally speaking, interpolateValues() should be used instead of this function

   Parameters:
   -----------
   source : Anything acceptable by loadRaster() or list
       The raster datasource, can be a filepath, a raster dataset etc., see
       RASTER.loadRaster() for details. Alternatively, a list of multiple
       such raster datasources.

   points : (X,Y) or [(X1,Y1), (X2,Y2), ...] or Location or LocationSet()
       Coordinates for the points to extract
       * All points must be in the same SRS
       * !REMEMBER! For lat and lon coordinates, X is lon and Y is lat
         (opposite of what you may think...)

   pointSRS : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the point to create
         * If not given, longitude/latitude is assumed
         * Only useful when 'points' is not a LocationSet

   winRange : int
       The window range (in pixels) to extract the values centered around the
       closest raster index to the indicated locations.
       * A winRange of 0 will only extract the closest raster value
       * A winRange of 1 will extract a window of shape (3,3)
       * A winRange of 3 will extract a window of shape (7,7)

   noDataOkay: bool
       If True, an error is raised if a 'noData' value is extracted
       If False, numpy.nan is inserted whenever a 'noData' value is extracted

   Returns:
   --------
   * If only a single location is given:
       namedtuple -> (data : The extracted data at the location
                      xOffset : The X index distance from the location to the
                                center of the closest raster pixel
                      yOffset : The Y index distance from the location to the
                                center of the closest raster pixel
                      inBounds: Flag for whether or not the location is within
                                The raster's bounds
                       )
   * If Multiple locations are given:
       pandas.DataFrame
           * Columns are (data, xOffset, yOffset, inBounds)
               - See above for column descriptions
           * Index is 0...N if 'points' input is not a LocationSet
           * Index is the LocationSet is if 'points' input is a LocationSet



.. py:function:: interpolateValues(source, points, pointSRS='latlon', mode='near', func=None, winRange=None, **kwargs)

   Interpolates the value of a raster at a given point or collection of points.

   Supports various interpolation schemes:
       'near', 'linear-spline', 'cubic-spline', 'average', or user-defined


   Parameters:
   -----------
   source : Anything acceptable by loadRaster() or list
       The raster datasource, can be a filepath, a raster dataset etc., see
       RASTER.loadRaster() for details. Alternatively, a list of multiple
       such raster datasources.

   points : (X,Y) or [(X1,Y1), (X2,Y2), ...] or Location or LocationSet()
       Coordinates for the points to extract
       * All points must be in the same SRS
       * !REMEMBER! For lat and lon coordinates, X is lon and Y is lat
         (opposite of what you may think...)

   pointSRS : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the point to create
         * If not given, longitude/latitude is assumed
         * Only useful when 'points' is not a LocationSet

   mode : str; optional
       The interpolation scheme to use
       * options are...
         "near" - Just gets the nearest value (this is default)
         "linear-spline" - calculates a linear spline in between points
         "cubic-spline" - calculates a cubic spline in between points
         "average" - calculates average across a window
         "func" - uses user-provided calculator

   func - function
       A user defined interpolation function
       * Only utilized when 'mode' equals "func"
       * The function must take three arguments in this order...
         - A 2 dimensional data matrix
         - A x-index-offset
         - A y-index-offset
       * See the example below for more information

   winRange : int
       The window range (in pixels) to extract the values centered around the
       closest raster index to the indicated locations.
       * A winRange of 0 will only extract the closest raster value
       * A winRange of 1 will extract a window of shape (3,3)
       * A winRange of 3 will extract a window of shape (7,7)
       * Only utilized when 'mode' equals "func"
       * All interpolation schemes have a predefined window range which is
         appropriate to their use
           - near -> 0
           - linear-spline -> 2
           - cubic-spline -> 4
           - average -> 3
           - func -> 3


   Returns:
   --------
   * If only a single location is given: numeric
       namedtuple -> (data : The extracted data at the location
                      xOffset : The X index distance from the location to the
                                center of the closest raster pixel
                      yOffset : The Y index distance from the location to the
                                center of the closest raster pixel
                      inBounds: Flag for whether or not the location is within
                                The raster's bounds
                       )
   * If Multiple locations are given: numpy.ndrray -> (N,)
       - where N is the number of locations

   Example:
   --------
   "Interpolate" according to the median value in a 5x5 window

   >>> def medianFinder( data, xOff, yOff ):
   >>>     return numpy.median(data)
   >>>
   >>> result = interpolateValues( <source>, <points>, mode='func',
   >>>                             func=medianFinder, winRange=2)



.. py:function:: mutateRaster(source, processor=None, bounds=None, boundsSRS='latlon', autocorrect=False, output=None, dtype=None, **kwargs)

   Process all pixels in a raster according to a given function. The boundaries
   of the resulting raster can be changed as long as the new boundaries are within
   the scope of the original raster, but the resolution cannot

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource

   processor: function; optional
       The function performing the mutation of the raster's data
       * The function will take single argument (a 2D numpy.ndarray)
       * The function must return a numpy.ndarray of the same size as the input
       * The return type must also be containable within a Float32 (int and
         boolean is okay)
       * See example below for more info

   bounds: tuple or Extent
       The boundary to clip the raster to before mutating
       * If given as an Extent, the extent is always cast to the source's native
         srs before mutating
       * If given as a tuple, (xMin, yMin, xMax, yMax) is expected
           - Units must be in the srs specified by 'boundsSRS'
       * This boundary must fit within the boundary of the rasters source
       * The boundary is always fitted to the source's grid, so the returned
         values do not necessarily match to the boundary which is provided

   boundsSRS: Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the 'bounds' argument
       * This is ignored if the 'bounds' argument is an Extent object or is None

   autocorrect : bool; optional
       If True, then before mutating the matrix extracted from the source will have
       pixels equal to its 'noData' value converted to numpy.nan
       * Data type will always result in a float, so be careful with large
         matricies

   output : str; optional
       A path to an output file
       * If output is None, the raster will be created in memory and a dataset
         handel will be returned
       * If output is given, the raster will be written to disk and nothing will
         be returned

   dtype : Type, str, or numpy-dtype; optional
       If given, forces the processed data to be a particular datatype
       * Example
         - A python numeric type  such as bool, int, or float
         - A Numpy datatype such as numpy.uint8 or numpy.float64
         - a String such as "Byte", "UInt16", or "Double"

   **kwargs:
       * All kwargs are passed on to a call to createRaster()

   Example:
   --------
   If you wanted to assign suitability factors based on a raster containing
   integer identifiers

   >>> def calcSuitability( data ):
   >>>     # create an ouptut matrix
   >>>     outputMatrix = numpy.zeros( data.shape )
   >>>
   >>>     # do the processing
   >>>     outputMatrix[ data == 1 ] = 0.1
   >>>     outputMatrix[ data == 2 ] = 0.2
   >>>     outputMatrix[ data == 10] = 0.4
   >>>     outputMatrix[ np.logical_and(data > 15, data < 20)  ] = 0.5
   >>>
   >>>     # return the output matrix
   >>>     return outputMatrix
   >>>
   >>> result = processRaster( <source-path>, processor=calcSuitability )


.. py:function:: indexToCoord(yi, xi, source=None, asPoint=False, bounds=None, dx=None, dy=None, yAtTop=True, srs=None)

   Convert the index of a raster to coordinate values.

   Parameters:
   -----------
   xi : int
       The x index
       * a numpy array of ints is also acceptable

   yi : int
       The y index
       * a numpy array of ints is also acceptable

   source : Anything acceptable by loadRaster()
       The contentual raster datasource

   asPoint : bool
       Instruct program to return point geometries instead of x,y coordinates

   Returns:
   --------
   * If 'asPoint' is True: ogr.Geometry
   * If 'asPoint' is False: tuple -> (x,y) coordinates



.. py:function:: drawSmopyMap(bounds, zoom, tileserver='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png', tilesize=256, maxtiles=100, ax=None, attribution='© OpenStreetMap contributors', attribution_size=12, **kwargs)

   Draws a basemap using the "smopy" python package

   NOTE:
   * The basemap is drawn using the Smopy python package. See here: https://github.com/rossant/smopy
   * Be careful to adhere to the usage guidelines of the chosen tile source
       - By default, this source is OSM. See here: https://wiki.openstreetmap.org/wiki/Tile_servers

   !IMPORTANT! If you will publish any images drawn with this method, it's likely that the tile source
   will require an attribution to be written on the image. For example, if using OSM tile (the default),
   you have to write "© OpenStreetMap contributors" clearly on the map. But this is different for each
   tile source!

   Parameters:
   -----------

       bounds : (xMin, yMix, xMax, yMax) or Extent
           The geographic extent to be drawn

       zoom : int
           The zoom level to draw (between 1-20)
           * I suggest starting low (e.g. 4), and zooming in until you find a level that suits your needs

       tileserver : string
           The tile server to use

       tilesize : int
           The pixel size of the tiles from 'tileserver'

       maxtiles : int
           The maximum tiles to use when drawing an image
           * Be careful to adhere to the usage conditions stated by your selected tileserver!

       ax : matplotlib.axes
           The matplotlib axes to draw on
           * If 'None', then one will be generated automatically

       kwargs
           All extra keyword arguments are passed on to matplotlib.ax.imshow


   Returns:
   --------

       namedtuple
           * .ax     -> The axes draw on
           * .srs    -> The SRS used when drawing (will always be EPSG 3857)
           * .bounds -> The boundaries of the drawn map



.. py:function:: drawRaster(source, srs=None, ax=None, resolution=None, cutline=None, figsize=(12, 12), xlim=None, ylim=None, fontsize=16, hideAxis=False, cbar=True, cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap='viridis', cbax=None, cbargs=None, cutlineFillValue=-9999, leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0, zorder=0, resampleAlg='med', **kwargs)

   Draw a raster as an image on a matplotlib canvas

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource to draw

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the drawn raster data
         * If not given, the raster's internal srs is assumed
         * If the drawing resolution does not match the source's inherent
           resolution, the source will be warped to the correct format

   ax : matplotlib axis; optional
       The axis to draw the geometries on
         * If not given, a new axis is generated and returned

   resolution : numeric or tuple; optional
       The resolution of the plotted raster data
       * Lower resolution means more pixels to draw and can be a burden on
         memory
       * If a tuple is given, resolutions in the X and Y direction are expected
       * Changing the resolution fron the inherent resolution requires a warp

   cutline : str or ogr.Geometry; optional
       The cutline to limit the drawn data too
       * If a string is given, it must be a path to a vector file
       * Values outside of the cutline are given the value 'cutlineFillValue'
       * Requires a warp

   cutlineFillValue : numeric; optional
       The value to give to values outside a cutline
       * Has no effect when cutline is not given

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

   resampleAlg : str, optional
       The resampleAlg passed on to a call of warp() if needed, by default "med"

   **kwargs : Passed on to a call to warp()
       * Determines how the warping is carried out
       * Consider using 'resampleAlg' or 'workingType' for finer control


   Returns:
   --------
   A namedtuple containing:
      'ax' -> The map axis
      'handles' -> All geometry handles which were created in the order they were
                   drawn
      'cbar' -> The colorbar handle if it was drawn



.. py:function:: polygonizeRaster(source, srs=None, flat=False, shrink=True)

   Polygonize a raster or an integer-valued data matrix

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource to polygonize
       * The Datatype MUST be of boolean of integer type

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the polygons to create
         * If not given, the raster's internal srs is assumed

   flat : bool
       If True, flattens the resulting geometries which share a contiguous
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



.. py:function:: contours(source, contourEdges, polygonize=True, unpack=True, **kwargs) -> pandas.DataFrame

   Create contour geometries at specified edges for the given raster data

   Notes:
   ======
   This function is similar to geokit.geom.polygonizeMatrix, although it only
   operates on the user-specified edges AND applies the 'Marching Squares'
   algorithm

   See the gdal function "GDALContourGenerateEx" for mor information on the
   specifics of this algorithm

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource to operate on

   contourEdges : [float,]
       The edges to search for withing the raster dataset
         * This parameter can be set as "None", in which case an additional
           argument should be given to specify how the edges should be determined
           - See the documentation of "GDALContourGenerateEx"
           - Ex. "LEVEL_INTERVAL=10", contourEdges=None

   polygons : bool
       If true, contours are returned as polygons instead of linstrings

   unpack : bool
       If True, Multipolygon/MultiLinestring objects are decomposed

   **kwargs:
       * All keyword arguments are passed on to a call to gdal.ContourGenerateEx
       * They are used to construct the 'options' parameter
       * Example keys include: LEVEL_INTERVAL, LEVEL_BASE, LEVEL_EXP_BASE, NODATA
       * Do not use the key "ID_FIELD", since this is employed already

   Returns:
   --------
   pandas.DataFrame

   * The column 'geom' corresponds to generated geometry objects
   * The columns 'ID' corresponds to the associated contour edge for each object


.. py:function:: warp(source, resampleAlg: Literal['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'rms', 'mode', 'max', 'min', 'med', 'Q1', 'Q3', 'sum'] = 'bilinear', cutline=None, output: str | None = None, pixelHeight=None, pixelWidth=None, srs=None, bounds: tuple | None = None, dtype=None, noData=None, fill=None, overwrite=True, meta=None, **kwargs) -> osgeo.gdal.Dataset | str

   Warps a given raster source to another context

   * Can be used to 'warp' a raster in memory to a raster on disk

   Note:
   -----
   Unless manually altered as keyword arguments, the gdal.Warp options
   'targetAlignedPixels' and 'copyMetadata' are both set to True

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()
       The raster datasource to draw

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the resulting raster
         * If not given, the raster's internal srs is assumed

   resampleAlg : str; optional
       The resampling algorithm to use when translating pixel values
       * Knowing which option to use can have significant impacts!
       * Options are: near , bilinear, cubic,
       cubicspline, lanczos, average, rms, mode,
       max, min, med, Q1, Q3, sum

   cutline : str or ogr.Geometry; optional
       The cutline to limit the drawn data too
       * If a string is given, it must be a path to a vector file
       * Values outside of the cutline are given the value 'cutlineFillValue'
       * Requires a warp

   output : str; optional
       The path on disk where the new raster should be created

   pixelHeight : numeric; optional
       The pixel height (y-resolution) of the output raster
       * Only required if this value should be changed

   pixelWidth : numeric; optional
       The pixel width (x-resolution) of the output raster
       * Only required if this value should be changed

   bounds : tuple; optional
       The (xMin, yMin, xMax, yMax) limits of the output raster
       * Only required if this value should be changed

   dtype : Type, str, or numpy-dtype; optional
       If given, forces the processed data to be a particular datatype
       * Only required if this value should be changed
       * Example
         - A python numeric type  such as bool, int, or float
         - A Numpy datatype such as numpy.uint8 or numpy.float64
         - a String such as "Byte", "UInt16", or "Double"

   noData : numeric; optional
       The no-data value to apply to the output raster

   fill : numeric; optional
       The fill data to place into the new raster before warping occurs
       * Does not play a role when writing a file to disk

   **kwargs:
       * All keyword arguments are passed on to a call to gdal.WarpOptions
       * Use these to fine-tune the warping procedure
       * Key Options are (from gdal.WarpOptions):
           format --- output format ("GTiff", etc...)
           targetAlignedPixels --- whether to force output bounds to be multiple
                                   of output resolution
           workingType --- working type (gdal.GDT_Byte, etc...)
           warpMemoryLimit --- size of working buffer in bytes
           creationOptions --- list of creation options
           srcNodata --- source nodata value(s)
           dstNodata --- output nodata value(s)
           multithread --- whether to multithread computation and I/O operations
           cutlineWhere --- cutline WHERE clause
           cropToCutline --- whether to use cutline extent for output bounds
           setColorInterpretation --- whether to force color interpretation of
                                      input bands to output bands

   Returns:
   --------
   * If 'output' is None: gdal.Dataset
   * If 'output' is a string: The path to the output is returned (for easy opening)



.. py:function:: sieve(source, threshold=100, connectedness=8, mask='none', quiet_flag=False, output=None, **kwargs)

   Removes raster polygons smaller than a provided threshold size (in pixels) and
   replaces them with the pixel value of the largest neighbour polygon.
   It is useful if you have a large amount of small areas on your raster map.

   Parameters:
   -----------
   source : Anything acceptable by loadRaster()

   threshold (int): minimum polygon size (number of pixels) to retain.

   connectedness (int): either 4 indicating that diagonal pixels are not considered directly
                        adjacent for polygon membership purposes or 8 indicating they are.

   mask (str): 'none' or 'default'. An optional mask band. All pixels in the mask band with a
               value other than zero will be considered suitable for inclusion in polygons.

   quiet_flag (bool): 0 or 1. Callback for reporting algorithm progress

   output : str; optional
       The path on disk where the new raster should be created

   **kwargs:
       * All kwargs are passed on to SieveFilter()
       * See gdal.SieveFilter for more details

   :returns: * **\* If 'output' is None** (*gdal.Dataset*)
             * **\* If 'output' is a string** (*The path to the output is returned (for easy opening)*)


.. py:function:: rasterCellNo(points, source=None, bounds=None, cellWidth=None, cellHeight=None)

   Returns the raster cell number for one or multiple points defined by geometry or lon/lat. Cell numeration
   starting in the top left corner cell of the raster with (0,0). Cells with (-1,-1) are out of bounds.

   :param points: Can be an osgeo.ogr.Geometry point or an iterable thereof, else a (lon, lat) tuple (in EPSG:4326) or an iterable thereof.
   :type points: osgeo.ogr.Geometry, tuple, iterable
   :param source: A gdal.Dataset type raster or a str formatted path to a raster file. Defaults to None.
   :type source: gdal.Dataset, str, optional
   :param bounds: Raster boundaries in EPSG:4326 in the form of (minX, minY, maxX, maxY). Defaults to None.
   :type bounds: tuple, optional
   :param cellWidth: The cell width in EPSG:4326 units. Defaults to None.
   :type cellWidth: int, float, optional
   :param cellHeight: The cell height in EPSG:4326 units. Defaults to None.
   :type cellHeight: int, float, optional

   NOTE: If source is given, all of the others must be None, else they must be given.

   :returns: tuple with (X, Y) cell No or an iterable thereof if multiple points were given.
   :rtype: tuple or iterable


