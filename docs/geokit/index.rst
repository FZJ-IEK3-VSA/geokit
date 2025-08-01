geokit
======

.. py:module:: geokit

.. autoapi-nested-parse::

   The GeoKit library is a collection of general geospatial operations



Submodules
----------

.. toctree::
   :maxdepth: 1

   /docs/geokit/_algorithms/index
   /docs/geokit/algorithms/index
   /docs/geokit/core/index
   /docs/geokit/error/index
   /docs/geokit/geom/index
   /docs/geokit/get_test_data/index
   /docs/geokit/gk/index
   /docs/geokit/raster/index
   /docs/geokit/srs/index
   /docs/geokit/util/index
   /docs/geokit/vector/index


Classes
-------

.. autoapisummary::

   geokit.Extent
   geokit.Location
   geokit.LocationSet
   geokit.RegionMask


Functions
---------

.. autoapisummary::

   geokit.drawGeoms
   geokit.drawRaster
   geokit.drawSmopyMap
   geokit.drawImage


Package Contents
----------------

.. py:class:: Extent(*args, srs='latlon')

   Bases: :py:obj:`object`


   Geographic extent

   The Extent object represents geographic extents of an area and exposes useful
   methods which depend on those extents. This includes:
       - Easily representing the boundaries as (xMin, xMax, yMin, yMax) or
         (xMin, yMin, xMax, yMax)
       - Casting to another projection system
       - Padding and shifting the boundaries
       - "Fitting" the boundaries onto a given resolution
       - Clipping a given raster file

   Initialization:
   ---------------
   * Extent(xMin, yMin, xMax, yMax [, srs])
   * Extent.from_xyXY( (xMin, yMin, xMax, yMax) [, srs])
   * Extent.from_xXyY( (xMin, xMax, yMin, yMax) [, srs])
   * Extent.fromGeom( geom [, srs] )
   * Extent.fromVector( vector-file-path )
   * Extent.fromRaster( raster-file-path )
   * Extent.load( args )

   Create extent from explicitly defined boundaries

   Usage:
   ------
   Extent(xMin, yMin, xMax, yMax [, srs=<srs>])
   Extent( (xMin, yMin, xMax, yMax) [, srs=<srs>])

   Where:
       xMin - The minimal x value in the respective SRS
       yMin - The minimal y value in the respective SRS
       xMax - The maximal x value in the respective SRS
       yMax - The maximal y value in the respective SRS
       srs - The Spatial Reference system to use



   .. py:attribute:: _whatami
      :value: 'Extent'



   .. py:attribute:: xMin


   .. py:attribute:: xMax


   .. py:attribute:: yMin


   .. py:attribute:: yMax


   .. py:attribute:: srs
      :value: 'latlon'



   .. py:attribute:: _box


   .. py:method:: from_xXyY(bounds, srs='latlon')
      :staticmethod:


      Create an Extent from explicitly defined boundaries

      Parameters:
      -----------
      bounds : tuple
          The (xMin, xMax, yMin, yMax) values for the extent

      srs : Anything acceptable to geokit.srs.loadSRS(); optional
          The srs of the input coordinates
            * If not given, lat/lon coordinates are assumed

      Returns:
      --------
      Extent




   .. py:method:: fromGeom(geom)
      :staticmethod:


      Create extent around a given geometry

      Parameters:
      -----------
      geom : ogr.Geometry
          The geometry from which to extract the extent

      Returns:
      --------
      Extent




   .. py:method:: fromTile(xi, yi, zoom)
      :staticmethod:


      Generates an Extent corresponding to tiles used for "slippy maps"

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
      geokit.Extent




   .. py:method:: fromTileAt(x, y, zoom, srs)
      :staticmethod:


      Generates an Extent corresponding to tiles used for "slippy maps"
      at the coordinates ('x','y') in the 'srs' reference system

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
      geokit.Extent




   .. py:method:: fromVector(source, where=None, geom=None)
      :staticmethod:


      Create extent around the contemts of a vector source

      Parameters:
      -----------
      source : Anything acceptable by loadVector()
          The vector datasource to read from

      where : str; optional
          An SQL-style filtering string
          * Can be used to filter the input source according to their attributes
          * For tips, see "http://www.gdal.org/ogr_sql.html"
          Ex:
            where="eye_color='Green' AND IQ>90"

      geom : ogr.Geometry; optional
          The geometry to search within
          * All features are extracted which touch this Geometry

      Returns:
      --------
      Extent




   .. py:method:: fromRaster(source)
      :staticmethod:


      Create extent around the contents of a raster source

      Parameters:
      -----------
      source : Anything acceptable by loadRaster()
          The vector datasource to read from

      Returns:
      --------
      Extent




   .. py:method:: fromLocationSet(locs)
      :staticmethod:


      Create extent around the contents of a LocationSet object

      Parameters:
      -----------
      locs : LocationSet

      Returns:
      --------
      Extent




   .. py:method:: fromWKT(wkt, delimiter='|')
      :staticmethod:


      Create extent from a Well-Known_Text string

      * Actually the input should be two WKT strings seperated by a "|" character
      * These correspond to "<A Geometry WKT>|<an SRS WKT>"

      Parameters:
      -----------
      wkt : The string to be processed

      delimiter : The delimiter which seperates the two WKT sections

      Returns:
      --------
      Extent




   .. py:method:: load(source, **kwargs)
      :staticmethod:


      Attempts to load an Extent from a variety of inputs in the most
      appropriate manner

      One Extent initializer (.fromXXX) is called depending on the inputs

          source is LocationSet -> Extent.fromLocationSet( source )
          source is ogr.Geometry -> Extent.fromGeom( source )
          source is not a string: -> Extent(*source, **kwargs)
          source is a string:
              First try: Extent.fromVector(source)
              Then try: Extent.fromRaster(source)

      If none of the above works, an error is raised

      Returns:
      --------
      Extent




   .. py:method:: _fromInfo(info)
      :staticmethod:


      GeoKit internal

      Creates an Extent from rasterInfo's returned value



   .. py:property:: xyXY

      Returns a tuple of the extent boundaries in order:
      xMin, yMin, xMax, yMax


   .. py:property:: xXyY

      Returns a tuple of the extent boundaries in order:
      xMin, xMax, yMin, yMax


   .. py:property:: xYXy

      Returns a tuple of the extent boundaries in order:
      xMin, yMax, xMax, yMin


   .. py:property:: yxYX

      Returns a tuple of the extent boundaries in order:
      yMin, xMin, yMax, xMax


   .. py:property:: YxyX

      Returns a tuple of the extent boundaries in order:
      yMax, xMin, yMin, xMax


   .. py:property:: ylim

      Returns a tuple of the y-axis extent boundaries in order:
      yMin, yMax


   .. py:property:: xlim

      Returns a tuple of the x-axis extent boundaries in order:
      xMin, xMax


   .. py:property:: box

      Returns a rectangular ogr.Geometry object representing the extent


   .. py:method:: __eq__(o)


   .. py:method:: __add__(o)


   .. py:method:: __repr__()


   .. py:method:: __str__()


   .. py:method:: exportWKT(delimiter='|')

      Export the extent to a Well-Known_Text string

      * Actually the will be two WKT strings seperated by a "|" character
      * These correspond to "<A Geometry WKT>|<an SRS WKT>"

      Parameters:
      -----------
      delimiter : The delimiter which seperates the two WKT sections

      Returns:
      --------
      string




   .. py:method:: pad(pad, percent=False)

      Pad the extent in all directions

      Parameters:
      -----------
      pad : float
          The amount to pad in all directions
          * In units of the extent's srs
          * Can also accept a negative padding

      percent : bool, optional
          If True, the padding values are understood to be a percentage of the
          unpadded extent

      Returns:
      --------
      Extent




   .. py:method:: shift(dx=0, dy=0)

      Shift the extent in the X and/or Y dimensions

      Parameters:
      -----------
      dx : float
          The amount to shift in the x dimension
          * In units of the extent's srs

      dy : float
          The amount to shift in the y dimension
          * In units of the extent's srs

      Returns:
      --------
      Extent




   .. py:method:: fitsResolution(unit, tolerance=1e-06)

      Test if calling Extent first around the given unit(s) (at least within
      an error defined by 'tolerance')

      Parameters:
      -----------
      unit : numeric or tuple
          The unit value(s) to check
          * If float, a single resolution value is assumed for both X and Y dim
          * If tuple, resolutions for both dimensions (x, y)

      tolerance : float
          The tolerance to allow when comparing float values

      Returns:
      --------
      Extent

      Examples:
      ---------
      >>> ex = Extent( 100, 100, 300, 500)
      >>> ex.fitsResolution(25) # True!
      >>> ex.fitsResolution( (25, 10) ) # True!
      >>> ex.fitsResolution(33) # False!
      >>> ex.fitsResolution( (25, 33) ) # False!




   .. py:method:: fit(unit, dtype=None, start_raster=None)

      Fit the extent to a given pixel resolution

      Note:
      -----
      The extent is always expanded to fit onto the given unit


      Parameters:
      -----------
      unit : numeric or tuple
          The unit value(s) to check
          * If numeric, a single value is assumed for both X and Y dim
          * If tuple, resolutions for both dimensions (x, y)

      start_raster : str ('left' or 'right')
          If None passed, the extent will be centered on the shape with overlaps left and right
          depending on individual shape width and raster cell width. If 'left'/'right' passed,
          extent edges will be matching min/max longitude of shape.

      dtype : Type or np.dtype
          The final data type of the boundary values

      Returns:
      --------
      Extent




   .. py:method:: corners(asPoints=False)

      Returns the four corners of the extent as ogr.gGometry points or as (x,y)
      coordinates in the extent's srs




   .. py:method:: center(srs=None)

      Get the Extent's center



   .. py:method:: castTo(srs, segments=100)

      Creates a new Extent by transforming an extent from the original Extent's
      srs to a target SRS.

      Note:
      -----
      The resulting region spanned by the extent will be equal-to or (almost
      certainly) larger than the original

      Parameters:
      -----------
      srs : Anything acceptable to geokit.srs.loadSRS()
          The srs to cast the Extent object to

      Returns:
      --------
      Extent




   .. py:method:: inSourceExtent(source)

      Tests if the extent box is at least partially contained in the extent-box
      of the given vector or raster source

      Parameters:
      -----------
      sources : str
          The sources to test




   .. py:method:: filterSources(sources, error_on_missing=True)

      Filter a list of sources by those whose's envelope overlaps the Extent.

      Note:
      -----
      Creates a filter object which can be immediately iterated over, or else
      can be cast as a list

      Parameters:
      -----------
      sources : list or str
          The sources to filter
          * An iterable of vector/raster sources
          * An iterable of paths pointing to vector/raster sources
          * A glob string which will generate a list of source paths
              - see glob.glob for more info

      error_on_missing : bool, optional
          If True, then if a file path is given which does not exist, a RunTime
              error is raised. Otherwise a warning is given
          Only performs check when input is a string

      Returns:
      --------
      filter




   .. py:method:: containsLoc(locs, srs=None)

      Test if the extent contains a location or an iterable of locations

      Parameters:
      -----------
      locs : Anything acceptable to LocationSet()
          The locations to be checked


      srs : Anything acceptable to geokit.srs.loadSRS()
          The srs to cast the Extent object to

      Returns:
      --------
      * If a single location is checker: bool
      * If multiple locations are checked: numpy.ndarray




   .. py:method:: overlaps(extent, referenceSRS=SRS.EPSG4326)

      Tests if the extent overlaps with another given extent

      Note:
      -----
      If an optional resolution ('res') is given, the containment value is also
      dependent on whether or not the given extent fits within the larger extent
      AND is situated along the given resolution

      Parameters:
      -----------
      extent : Extent
          The Extent object to test for containment

      referenceSRS
          The spatial reference frame to do the comparison in
          * Can be 'self'

      Returns:
      --------
      bool




   .. py:method:: contains(extent, res=None)

      Tests if the extent contains another given extent

      Note:
      -----
      If an optional resolution ('res') is given, the containment value is also
      dependent on whether or not the given extent fits within the larger extent
      AND is situated along the given resolution

      Parameters:
      -----------
      extent : Extent
          The Extent object to test for containment

      res : numeric or tuple
          The X & Y resolution to enforce

      Returns:
      --------
      bool




   .. py:method:: findWithin(extent, res=100, yAtTop=True)

      Finds the indexes of the given extent within the main extent according
      to the given resolution.

      Note:
      -----
      * Use this to compute the index offsets and window sizes of a window
        within a raster dataset
      * The two extents MUST share the same SRS

      Parameters:
      -----------
      extent : Extent
          The extent to find within the calling extent

      res : numeric or tuple
          A resolution to check containment on

      yAtTop : bool; optional
          Instructs the offsetting to begin from yMax instead of from yMin

      :returns: tuple -> (xOffset, yOffset, xWindowSize, yWindowSize)



   .. py:method:: computePixelSize(*args)

      Finds the pixel resolution which fits to the Extent for a given pixel count.

      Note:
      -----
      * If only one integer argument is given, it is assumed to fit to both the X and Y dimensions
      * If two integer arguments are given, it is assumed to be in the order X then Y


      :returns: tuple -> (pixelWidth, pixelHeight)



   .. py:method:: createRaster(pixelWidth, pixelHeight, **kwargs)

      Convenience function for geokit.raster.createRaster which sets 'bounds'
      and 'srs' inputs

      * The input resolution MUST fit within the extent

      Parameters:
      -----------
      pixelWidth : numeric
          The pixel width of the raster in units of the input srs
          * The keyword 'dx' can be used as well and will override anything given
          assigned to 'pixelWidth'

      pixelHeight : numeric
          The pixel height of the raster in units of the input srs
          * The keyword 'dy' can be used as well and will override anything given
            assigned to 'pixelHeight'

      **kwargs:
          All other keyword arguments are passed on to geokit.raster.createRaster()

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: _quickRaster(pixelWidth, pixelHeight, **kwargs)

      Convenience function for geokit.raster.createRaster which sets 'bounds'
      and 'srs' inputs

      * The input resolution MUST fit within the extent

      Parameters:
      -----------
      pixelWidth : numeric
          The pixel width of the raster in units of the input srs
          * The keyword 'dx' can be used as well and will override anything given
          assigned to 'pixelWidth'

      pixelHeight : numeric
          The pixel height of the raster in units of the input srs
          * The keyword 'dy' can be used as well and will override anything given
            assigned to 'pixelHeight'

      **kwargs:
          All other keyword arguments are passed on to geokit.raster.createRaster()

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: extractMatrix(source, strict=True, **kwargs)

      Convenience wrapper around geokit.raster.extractMatrix(). Extracts the
      extent directly from the given raster source as a matrix around the Extent

      Note:
      -----
      The called extent must fit somewhere within the raster's grid

      Parameters:
      -----------
      source: gdal.Dataset or str
          The raster source to be read

      strict: bool; optional
          Whether or not to allow a returned value which does not fit to the
          given extent
          !! If this is set to False, it is STRONGLY recommended to also set the
             argument 'returnBounds' as True so that the new computed boundary
             can be known

      **kwargs
          All keyword arguments are passed to geokit.raster.extractMatrix

      Returns:
      --------
      numpy.ndarray or tuple
          * See geokit.raster.extractMatrix




   .. py:method:: warp(source, pixelWidth, pixelHeight, strict=True, **kwargs)

      Convenience function for geokit.raster.warp() which automatically sets the
      'srs' and 'bounds' input.

      Note:
      -----
      When creating an 'in memory' raster vs one which is saved to disk, a slightly
      different algorithm is used which can sometimes add an extra row of pixels. Be
      aware of this if you intend to compare value-matricies directly from rasters
      generated with this function.

      Parameters:
      -----------
      source : str
          The path to the vector file to load

      pixelHeight : numeric; optional
          The pixel height (y-resolution) of the output raster
          * Only required if this value should be changed

      pixelWidth : numeric; optional
          The pixel width (x-resolution) of the output raster
          * Only required if this value should be changed

      strict : bool
          If True, raise an error if trying to warp to a pixelWidth and
          pixelHeight which does not fit into the Extent

      **kwargs:
          All other keyword arguments are passed on to geokit.raster.warp()

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: rasterize(source, pixelWidth, pixelHeight, strict=True, **kwargs)

      Convenience function for geokit.vector.rasterize() which automatically
      sets the 'srs' and 'bounds' input.

      Note:
      -----
      When creating an 'in memory' raster vs one which is saved to disk, a slightly
      different algorithm is used which can sometimes add an extra row of pixels. Be
      aware of this if you intend to compare value-matricies directly from rasters
      generated with this function.

      Parameters:
      -----------
      source : str
          The path to the vector file to load

      pixelHeight : numeric; optional
          The pixel height (y-resolution) of the output raster
          * Only required if this value should be changed

      pixelWidth : numeric; optional
          The pixel width (x-resolution) of the output raster
          * Only required if this value should be changed

      strict : bool
          If True, raise an error if trying to rasterize to a pixelWidth and
          pixelHeight which does not fit into the Extent

      **kwargs:
          All other keyword arguments are passed on to geokit.raster.warp()

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: extractFeatures(source, **kwargs)

      Convenience wrapper for geokit.vector.extractFeatures() by setting the
      'geom' input to the extent's box

      Parameters:
      -----------
      source : str
          The path to the vector file to load

      **kwargs:
          All other keyword arguments are passed on to vector.extractFeatures()

      Returns:
      --------
      * If asPandas is True: pandas.DataFrame or pandas.Series
      * If asPandas is False: generator




   .. py:method:: mutateVector(source, matchContext=False, **kwargs)

      Convenience function for geokit.vector.mutateVector which automatically
      sets 'srs' and 'geom' input to the Extent's srs and geometry

      Note:
      -----
      If this is called without any arguments except for a source, it serves
      to clip the vector source around the extent

      Parameters:
      -----------
      source : Anything acceptable to geokit.vector.loadVector()
          The source to clip

      matchContext : bool; optional
          * If True, transforms all geometries to the Extent's srs before
            mutating
          * If False, the Extent is cast to the source's srs, and all filtering
            and mutating happens in that context

      **kwargs:
          All other keyword arguments are passed to geokit.vector.mutateVector


      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: mutateRaster(source, pixelWidth=None, pixelHeight=None, matchContext=False, warpArgs=None, processor=None, resampleAlg='bilinear', **mutateArgs)

      Convenience function for geokit.raster.mutateRaster which automatically
      warps the raster to the extent's area and srs before mutating

      Note:
      -----
      If this is called without any arguments except for a source, it serves
      to clip the raster source around the Extent, therefore performing
      the same function as Extent.warp(...) on an Extent which has been cast
      to the source's srs

      Parameters:
      -----------
      source : Anything acceptable to geokit.raster.loadRaster()
          The source to mutate

      pixelHeight : numeric
          The pixel height (y-resolution) of the output raster

      pixelWidth : numeric
          The pixel width (x-resolution) of the output raster

      matchContext : bool; optional
          * If True, Warp to the Extent's boundaries and srs before mutating
              - pixelHeight and pixelWidth MUST be provided in this case
          * If False, only warp to the Extent's boundaries, but keep its
            srs and resolution intact

      warpArgs : dict; optional
          Arguments to apply to the warping step
          * See geokit.raster.warp()

      processor - function; optional
          The function performing the mutation of the raster's data
          * The function will take single argument (a 2D numpy.ndarray)
          * The function must return a numpy.ndarray of the same size as the input
          * The return type must also be containable within a Float32 (int and
            boolean is okay)
          * See example in geokit.raster.mutateRaster for more info

      resampleAlg : str; optional
          The resampling algorithm to use while warping
          * Knowing which option to use can have significant impacts!
          * Options are: 'near', 'bilinear', 'cubic', 'average'

      **kwargs:
          All other keyword arguments are passed to geokit.vector.mutateVector

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: clipRaster(source, output=None, **kwargs)

      Clip a given raster source to the caling Extent

      Parameters:
      -----------
      source : Anything acceptable to geokit.raster.loadRaster()
          The source to clip

      **kwargs:
          All other keyword arguments are passed to gdal.Translate

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: contoursFromRaster(raster, contourEdges, transformGeoms=True, **kwargs)

      Convenience wrapper for geokit.raster.contours which autmatically
      clips a raster to the invoked Extent

      Parameters:
      -----------
      raster : The raster datasource to warp from

      contourEdges : [float,]
          The edges to search for withing the raster dataset
          * This parameter can be set as "None", in which case an additional
              argument should be given to specify how the edges should be determined
              - See the documentation of "GDALContourGenerateEx"
              - Ex. "LEVEL_INTERVAL=10", contourEdges=None

      transformGeoms : bool
          If True, geometries are transformed to the Extent's SRS, otehrwise they
          are left in their native SRS

      kwargs
          Keyword arguments to pass on to the contours function
          * See geokit.raster.contours

      Returns:
      --------
      pandas.DataFrame

      With columns:
          'geom' -> The contiguous-valued geometries
          'ID' -> The associated contour edge for each object




   .. py:method:: tileIndexBox(zoom)

      Determine the tile indexes at a given zoom level which surround the invoked Extent

      Parameters:
      -----------
      zoom : int
          The zoom level of the expected tile source

      Returns:
      --------
      namedtuple:
          - xi_start: int - The starting x index
          - xi_stop:  int - The ending x index
          - yi_start: int - The starting y index
          - yi_stop:  int - The ending y index




   .. py:method:: tileSources(zoom, source=None)

      Get the tiles sources which contribute to the invoking Extent

      Parameters:
      -----------
      zoom : int
          The zoom level of the expected tile source

      source : str
          The source to fetch tiles from
          * Must include indicators for:
            {z} -> The tile's zoom level
            {x} -> The tile's x-index
            {y} -> The tile's y-index
          * Ex:
            File on disk     : "/path/to/tile/directory/{z}/{x}/{y}/filename.tif"
            Remote HTTP file : "/vsicurl_streaming/http://path/to/resource/{z}/{x}/{y}/filename.tif"
          * Find more info at https://gdal.org/user/virtual_file_systems.html


      Yields:
      --------
      if source is given:     str
      if source is not given: (xi,yi,zoom)




   .. py:method:: subTiles(zoom, asGeom=False)

      Generates tile Extents at a given zoom level which encompass the envoking Extent.

      Parameters:
      -----------
      zoom : int
          The zoom level of the expected tile source

      asGeom : bool
          If True, returns tuple of ogr.Geometries in stead of (xi,yi,zoom) tuples

      Returns:
      --------
      Generator of Geometries or (xi,yi,zoom) tuples




   .. py:method:: tileBox(zoom, return_index_box=False)

      Determine the tile Extent at a given zoom level which surround the invoked Extent

      Parameters:
      -----------
      zoom : int
          The zoom level of the expected tile source

      return_index_box : bool
          If true, also return the index box at the specified zoom level (from self.tileIndexBox)

      Returns:
      --------
      if return_index_box is False: geokit.Extent

      if return_index_box is True: Tuple
          - Item 0: geokit.Extent
          - Item 1: namedtuple(xi_start, xi_stop, yi_start, yi_stop)




   .. py:method:: tileMosaic(source, zoom, **kwargs)

      Create a raster source surrounding the Extent from a collection of tiles

      Parameters:
      -----------
      source : str
          The source to fetch tiles from
          * Must include indicators for:
            {z} -> The tile's zoom level
            {x} -> The tile's x-index
            {y} -> The tile's y-index
          * Ex:
            File on disk     : "/path/to/tile/directory/{z}/{x}/{y}/filename.tif"
            Remote HTTP file : "/vsicurl_streaming/http://path/to/resource/{z}/{x}/{y}/filename.tif"
          * Find more info at https://gdal.org/user/virtual_file_systems.html


      zoom : int
          The zoom level of the expected tile source

      pixelsPerTile : int, (int,int)
          The number of pixels found in each tile

      workingType : np.dtype
          The datatype of the working matrix (should match the raster source)

      noData : numeric
          The value to treat as 'no data'

      output : str
          An optional path for an output raster (.tif) file

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: rasterMosaic(sources, _warpKwargs={}, _skipFiltering=False, **kwargs)

      Create a raster source surrounding the Extent from a collection of other rasters

      Parameters:
      -----------
      sources : list, or something acceptable to gk.Extent.filterSources
          The sources to add together over the invoking Extent

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: drawSmopyMap(zoom, tileserver='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png', tilesize=256, maxtiles=100, ax=None, **kwargs)

      Draws a basemap using the "smopy" python package

      * See more details about smopy here: https://github.com/rossant/smopy

      Parameters:
      -----------

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



.. py:class:: Location(lon, lat)

   Bases: :py:obj:`object`


   Represents a single location using lat/lon as a base coordinate system

   Initializations:
   ----------------

   # If you trust my programming skills and have any of the argument types listed
   below:
   >>> Location.load( args, srs=SRS )

   # If you have a latitude and longitude value
   >>> Location( latitude, longitude )

   # If you have an X and a Y coordinate in any arbitrary SRS
   >>> Location.fromXY( X, Y, srs=SRS)

   # If you have a string structured like such: "(5.12243,52,11342)"
   >>> Location.fromString( string, srs=SRS )

   # If you have a point geometry
   >>> Location.fromPointGeom( pointGeometryObject )


   Initialize a Location Object by explicitly providing lat/lon coordinates

   :param lon: The location's longitude value
   :type lon: numeric
   :param lat: The location's latitude value
   :type lat: numeric


   .. py:attribute:: _TYPE_KEY_
      :value: 'Location'



   .. py:attribute:: _e
      :value: 1e-05



   .. py:attribute:: lat


   .. py:attribute:: lon


   .. py:attribute:: _geom
      :value: None



   .. py:method:: __hash__()


   .. py:method:: __eq__(o)


   .. py:method:: __ne__(o)


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: fromString(self, srs=None)
      :staticmethod:


      Initialize a Location Object by providing a string

      * Must be formated like such: "(5.12243,52,11342)"
      * Whitespace is okay
      * Will only take the FIRST match it finds

      :param s: The string to parse
      :type s: string
      :param srs: The srs for input coordinates
      :type srs: Anything acceptable to gk.srs.loadSRS; optional
      :param Returns:
      :param --------:
      :param Locations:



   .. py:method:: fromPointGeom(g)
      :staticmethod:


      Initialize a Location Object by providing an OGR Point Object

      * Must have an SRS within the object

      :param g: The string to parse
      :type g: ogr.Geometry
      :param Returns:
      :param --------:
      :param Locations:



   .. py:method:: fromXY(x, y, srs=3035)
      :staticmethod:


      Initialize a Location Object by providing a n X and Y coordinate

      :param x: The location's x value
      :type x: numeric
      :param y: The location's y value
      :type y: numeric
      :param srs: The srs for input coordinates
      :type srs: Anything acceptable to gk.srs.loadSRS
      :param Returns:
      :param --------:
      :param Locations:



   .. py:property:: latlon


   .. py:method:: asGeom(srs='latlon')

      Extract the Location as an ogr.Geometry object in an arbitrary SRS

      :param srs: The srs for the created object
      :type srs: Anything acceptable to gk.srs.loadSRS
      :param Returns:
      :param --------:
      :param ogr.Geometry:



   .. py:method:: asXY(srs=3035)

      Extract the Location as an (X,Y) tuple in an arbitrary SRS

      :param srs: The srs for the created tuple
      :type srs: Anything acceptable to gk.srs.loadSRS
      :param Returns:
      :param --------:
      :param tuple -> (X:
      :param Y):



   .. py:property:: geom


   .. py:method:: makePickleable()

      Clears OGR objects from the Location's internals so that it becomes
      "pickleable"



   .. py:method:: load(loc, srs=4326)
      :staticmethod:


      Tries to load a Location object in the correct manner by inferring
      from the input type

      * Ends up calling one of the Location.from??? initializers

      :param loc: The location data to interpret
      :type loc: Location or ogr.Geometry or str or tuple
      :param srs: The srs for input coordinates
                  * If not given, latitude and longitude coordinates are expected
      :type srs: Anything acceptable to gk.srs.loadSRS
      :param Returns:
      :param --------:
      :param Locations:



.. py:class:: LocationSet(locations, srs=4326, _skip_check=False)

   Bases: :py:obj:`object`


   Represents a collection of location using lat/lon as a base coordinate
   system

   Note:
   -----
   When initializing, an iterable of anything acceptable by Location.load is
   expected

   Initializations:
   ----------------
   >>> LocationSet( iterable )

   Initialize a LocationSet Object

   * If only a single location is given, a set is still created

   :param locations:
                     The locations to collect
                       * Can be anything acceptable by Location.load()
   :type locations: iterable
   :param srs: The srs for input coordinates
               * if not given, lat/lon coordinates are expected
   :type srs: Anything acceptable to gk.srs.loadSRS; optional


   .. py:attribute:: _TYPE_KEY_
      :value: 'LocationSet'



   .. py:attribute:: _lons
      :value: None



   .. py:attribute:: _lats
      :value: None



   .. py:attribute:: _bounds4326
      :value: None



   .. py:attribute:: count


   .. py:attribute:: shape


   .. py:method:: __len__()


   .. py:method:: __getitem__(i)


   .. py:method:: __repr__()


   .. py:method:: getBounds(srs=4326)

      Returns the bounding box of all locations in the set in an arbitrary
      SRS

      :param srs: The srs for output coordinates
                  * if not given, lat/lon coordinates are expected
      :type srs: Anything acceptable to gk.srs.loadSRS; optional
      :param Returns:
      :param --------:
      :param tuple -> (xMin:
      :param yMin:
      :param xMax:
      :param yMax):



   .. py:property:: lats


   .. py:property:: lons


   .. py:method:: asString()

      Create a list of string representations of all locations in the set

      Returns:
      --------
      list -> [ '(lon1,lat1)', (lon2,lat2)', ... ]




   .. py:method:: makePickleable()

      Clears OGR objects from all individual Location's internals so that
      they become "pickleable"



   .. py:method:: asGeom(srs=4326)

      Create a list of ogr.Geometry representations of all locations in the
      set

      :param srs: The srs for output coordinates
                  * if not given, lat/lon coordinates are expected
      :type srs: Anything acceptable to gk.srs.loadSRS; optional
      :param Returns:
      :param --------:
      :param list -> [ Geometry1:
      :param Geometry1:
      :param ... ]:



   .. py:method:: asXY(srs=3035)

      Create an Nx2 array of x and y coordinates for all locations in the set

      :param srs: The srs for output coordinates
                  * if not given, EPSG3035 coordinates are assumed
      :type srs: Anything acceptable to gk.srs.loadSRS; optional
      :param Returns:
      :param --------:
      :param numpy.ndarray -> Nx2:



   .. py:method:: asHash()


   .. py:method:: splitKMeans(groups=2, **kwargs)

      Split the locations into groups according to KMEans clustering

      * An equal count of locations in each group is not guaranteed

      :param groups: The number of groups to split the locations into
      :type groups: int
      :param kwargs: All other keyword arguments are passed on to sklearn.cluster.KMeans
      :param Yields:
      :param --------:
      :param LocationSet -> A location set of each clustered group:



   .. py:method:: bisect(lon=True, lat=True, delta=0.005)

      Cluster the locations by finding a bisecting line in lat/lon
      coordinates in either (or both) directions

      * An equal count of locations in each group is not guaranteed
      * Will always either return 2 or 4 cluster groups

      :param lon: Split locations in the longitude direction
      :type lon: bool
      :param lat: Split locations in the latitude direction
      :type lat: bool
      :param delta: The search speed
                    * Smaller values will take longer to converge on the true bisector
      :type delta: float
      :param Yields:
      :param --------:
      :param LocationSet -> A location set of each clustered group:



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



.. py:class:: RegionMask(extent, pixelRes, mask=None, geom=None, attributes=None, **kwargs)

   Bases: :py:obj:`object`


   The RegionMask object represents a given region and exposes methods allowing
   for easy manipulation of geospatial data around that region.

   RegionMask objects are defined by providing a polygon (either via a vector
   file, an ogr.Geometry, or a Well-Known-Text (WKT) string), a projection system
   to work in, and an extent and pixel resolution to create a matrix mask (i.e.
   boolean values) of.

   * The extent of the generated mask matrix is the tightest fit around the region
     in units of the pixel resolution. However, the extenT can be defined explcitly
     if desired
   * The region can be manipulated as a vector polygon via the ".geometry"
     attribute, which exposes the geometry as an ogr.Geometry. To incoporate this
     into other vector-handeling libraries it is suggested to use the
     ".ExportToWkt()" method available via OGR.
   * The region can be manipulated as a raster matrix via the ".mask" attribute
     which exposes the mask as a boolean numpy.ndarray
   * Any raster source can be easily warped onto the region-mask's extent,
     projection, and resolution via the ".warp" method
   * Any vector source can be rasterized onto the region-mask's extent, projection,
     and resolution via the ".rasterize" method
   * The default mask set-up is defined by the constant members: DEFAULT_SRS,
     DEFAULT_RES, and DEFAULT_PAD

   Initializers:
   -------------
   * RegionMask(...)
       - This is not the preferred way

   * RegionMask.fromVector( ... )

   * RegionMask.fromVectorFeature( ... )

   * RegionMask.fromGeom( ... )

   * RegionMask.fromMask( ... )

   * RegionMask.load( ... )
       - This function tries to determine which of the other initializers
         should be used based off the input


   The default constructor for RegionMask objects. Creates a RegionMask
   directly from a matrix mask and a given extent (and optionally a geometry).
   Pixel resolution is calculated in accordance with the shape of the mask
   mask and the provided extent

   * Generally one should use the '.load' or else one of the '.fromXXX'
     methods to create RegionMasks

   Parameters:
   -----------
   extent : Extent object
       The geospatial context of the region mask
       * The extent must fit the given pixel sizes
       * All computations using the RegionMask will be evaluated within this
         spatial context

   pixelRes : float or tuple
       The RegionMask's native pixel size(s)
       * If float : A pixel size to apply to both the X and Y dimension
       * If (float float) : An X-dimension and Y-dimension pixel size
       * All computations using the RegionMask will generate results in
         reference to these pixel sizes (i.e. either at this resolution or
         at some scaling of this resolution)

   mask : numpy-ndarray
       A mask over the context area defining which pixel as inside the region
       and which are outside
       * Must be a 2-Dimensional bool-matrix describing the region, where:
           - 0/False -> "not in the region"
           - 1/True  -> "Inside the region"
       * Either a mask or a geometry must be given, but not both

   geom : ogr-Geomertry
       A geometric representation of the RegionMask's region
       * Either a mask or a geometry must be given, but not both

   attributes : dict
       Keyword attributes and values to carry along with the RegionMask



   .. py:attribute:: DEFAULT_SRS
      :value: 'europe_m'



   .. py:attribute:: DEFAULT_RES
      :value: 100



   .. py:attribute:: DEFAULT_PAD
      :value: None



   .. py:attribute:: extent


   .. py:attribute:: srs


   .. py:attribute:: pixelWidth


   .. py:attribute:: pixelHeight


   .. py:attribute:: width
      :value: None



   .. py:attribute:: height
      :value: None



   .. py:attribute:: _mask
      :value: None



   .. py:attribute:: _vector
      :value: None



   .. py:attribute:: _vectorPath
      :value: None



   .. py:attribute:: attributes


   .. py:method:: fromMask(extent, mask, attributes=None)
      :staticmethod:


      Make a RegionMask directly from amask matrix and extent

      Note:
      -----
      Pixel sizes are calculated from the extent boundaries and mask dimensional
      sizes

      Parameters:
      -----------
      extent : Extent object
          The geospatial context of the region mask
          * The extent must fit the given pixel sizes
          * All computations using the RegionMask will be evaluated within this
            spatial context

      mask : numpy-ndarray
          A mask over the context area defining which pixel as inside the region
          and which are outside
          * Must be a 2-Dimensional bool-matrix describing the region, where:
              - 0/False -> "not in the region"
              - 1/True  -> "Inside the region"

      attributes : dict
          Keyword attributes and values to carry along with the RegionMask

      Returns:
      --------
      RegionMask




   .. py:method:: fromGeom(geom, pixelRes=DEFAULT_RES, srs=DEFAULT_SRS, start_raster=None, extent=None, padExtent=DEFAULT_PAD, attributes=None, **k)
      :staticmethod:


      Make a RasterMask from a given geometry

      Parameters:
      -----------
      geom : ogr-Geomertry or str
          A geometric representation of the RegionMask's region
          * If a string is given, geokit.geom.convertWKT(geom, srs) is called
            to convert it to an ogr.Geometry

      pixelRes : float or tuple
          The RegionMask's native pixel resolution(s)
          * If float : A pixel size to apply to both the X and Y dimension
          * If (float float) : An X-dimension and Y-dimension pixel size

      srs : Anything acceptable to geokit.srs.loadSRS()
          The srs context of the generated RegionMask object
          * This srs is superseded by the srs in an explicitly defined extent
          * The default srs EPSG3035 is only valid for a European context

      extent : Extent object
          The geospatial context of the generated region mask
          * The extent must fit the given pixel sizes

      padExtent : float; optional
          An amount by which to pad the extent before generating the RegionMask

      attributes : dict
          Keyword attributes and values to carry along with the RegionMask

      Returns:
      --------
      RegionMask




   .. py:method:: fromVector(source, where=None, geom=None, start_raster=None, pixelRes=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, limitOne=True, **kwargs)
      :staticmethod:


      Make a RasterMask from a given vector source

      Note:
      -----
      Be careful when creating a RegionMask over a large area (such as a country)!
      Using the default pixel size for a large area (such as a country) can
      easily consume your system's memory

      Parameters:
      -----------
      source : Anything acceptable by loadVector()
          The vector data source to read from

      where : str, int; optional
          If string -> An SQL-like where statement to apply to the source
          If int -> The feature's ID within the dataset
          * Feature attribute name do not need quotes
          * String values should be wrapped in 'single quotes'
          Example: If the source vector has a string attribute called "ISO" and
                   a integer attribute called "POP", you could use....

              where = "ISO='DEU' AND POP>1000"

      geom : ogr.Geometry; optional
          The geometry to search with
          * All features are extracted which touch this geometry

      pixelRes : float or tuple
          The RegionMask's native pixel resolution(s)
          * If float : A pixel size to apply to both the X and Y dimension
          * If (float float) : An X-dimension and Y-dimension pixel size

      srs : Anything acceptable to geokit.srs.loadSRS()
          The srs context of the generated RegionMask object
          * This srs is superseded by the srs in an explicitly defined extent
          * The default srs EPSG3035 is only valid for a European context

      extent : Extent object
          The geospatial context of the generated region mask
          * The extent must fit the given pixel sizes
          * If not specified, the entire extent of the vector file is assumed

      padExtent : float; optional
          An amount by which to pad the extent before generating the RegionMask

      limitOne : bool; optional
          Whether or not to allow more than one feature to be extracted

      Returns:
      --------
      RegionMask




   .. py:method:: load(region, start_raster=None, **kwargs)
      :staticmethod:


      Tries to initialize and return a RegionMask in the most appropriate way.

      Note:
      -----
      If 'region' input is...
          * Already a RegionMask, simply return it
          * A file path, use RegionMask.fromVector
          * An OGR Geometry object, assume is it to be loaded by RegionMask.fromGeom
          * A NumPy array, assume is it to be loaded by RegionMask.fromMask
              - An 'extent' input must also be given

      Parameters:
      -----------
      region : Can be RegionMask, str, ogr.Geometry, numpy.ndarray
          The shape  defining the region over which to build the RegionMask
          * See the note above

      where : str, int; optional
          If string -> An SQL-like where statement to apply to the source
          If int -> The feature's ID within the dataset
          * Feature attribute name do not need quotes
          * String values should be wrapped in 'single quotes'
          Example: If the source vector has a string attribute called "ISO" and
                   a integer attribute called "POP", you could use....

              where = "ISO='DEU' AND POP>1000"

      geom : ogr.Geometry; optional
          The geometry to search with
          * All features are extracted which touch this geometry

      pixelRes : float or tuple
          The RegionMask's native pixel resolution(s)
          * If float : A pixel size to apply to both the X and Y dimension
          * If (float float) : An X-dimension and Y-dimension pixel size

      srs : Anything acceptable to geokit.srs.loadSRS()
          The srs context of the generated RegionMask object
          * This srs is superseded by the srs in an explicitly defined extent
          * The default srs EPSG3035 is only valid for a European context

      extent : Extent object
          The geospatial context of the generated region mask
          * The extent must fit the given pixel sizes
          * If not specified, the entire extent of the vector file is assumed

      padExtent : float; optional
          An amount by which to pad the extent before generating the RegionMask




   .. py:property:: pixelRes

      The RegionMask's pixel size.

      !!Only available when pixelWidth equals pixelHeight!!


   .. py:method:: buildMask(**kwargs)

      Explicitly build the RegionMask's mask matrix.

      * The 'width' and 'height' attributes for the RegionMask are also set
        when this function is called
      * All kwargs are passed on to a call to geokit.vector.rasterize()




   .. py:property:: mask

      The RegionMask's mask array as an 2-dimensional boolean numpy array.

      * If no mask was given at the time of the RegionMask's creation, then a
        mask will be generated on first access to the 'mask' property
      * The mask can be rebuilt in a customized way using the
        RegionMask.buildMask() function


   .. py:property:: area


   .. py:method:: buildGeometry()

      Explicitly build the RegionMask's geometry



   .. py:property:: geometry

      Fetches a clone of the RegionMask's geometry as an OGR Geometry object

      * If a geometry was not provided when the RegionMask was initialized,
        then one will be generated from the RegionMask's mask matrix in the
        RegionMask's extent
      * The geometry can always be deleted and rebuild using the
        RegionMask.rebuildGeometry() function


   .. py:property:: vectorPath

      Returns a path to a vector path on disc which is built only once


   .. py:property:: vector

      Returns a vector saved in memory which is built only once


   .. py:method:: _repr_svg_()


   .. py:method:: _tempFile(head='tmp', ext='.tif')

      ***RM INTERNAL***

      Use this to create a temporary file associated with the RegionMask which
      will be deleted when the RM goes out of scope.

      !! BEWARE OF EXTERNAL DEPENDANCIES WHEN THE RM IS GOING OUT OF SCOPE,
      THIS WILL CAUSE A LOT OF ISSUES !!



   .. py:method:: __del__()


   .. py:method:: _resolve(div)


   .. py:method:: applyMask(mat, noData=0)

      Shortcut to apply the RegionMask's mask to an array. Mainly intended
      for internal use

      * When the passed matrix does not have the same extents of the given matrix,
        it is assumed that the RegionMask's mask needs to be scaled so that the
        matrix dimensions match

      * The RM's mask can only be scaled UP, and the given matrix's dimensions
        must be mutiples of the mask's dimensions

      Parameters:
      -----------
      mat : np.ndarray
          The matrix to apply the mask to
          * Must have dimensions equal, or are multiples of, the mask's

      noData : float
          The no-data value to set into matrix's values which are not within
          the region

      Returns:
      --------
      numpy.ndarray




   .. py:method:: _returnBlank(resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=None, **kwargs)


   .. py:method:: indicateValueToGeoms(source, value, contours=False, transformGeoms=True)

      TODO: UPDATE ME



   .. py:method:: indicateValues(source, value, buffer=None, resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=None, resampleAlg='bilinear', bufferMethod='area', preBufferSimplification=None, warpDType=None, prunePatchSize=0, threshold=0.5, multiProcess=True, **kwargs)

      Indicates those pixels in the RegionMask which correspond to a particular
      value, or range of values, from a given raster datasource

      Returns a matrix matching the RegionMask's mask dimensions wherein 0 means
      the pixels is not included in the indicated set, and 1 meaning the pixel
      is included in the indicated set. Intermediate values are also possible.
      This results from a scenario when the datasource's resolution does not
      line up perfectly with the RegionMask's resolution and, as a result, a
      RegionMask pixel overlaps multiple datasource pixels which are not all
      indicated (or not-indicated).

      * Value processing is performed BEFORE a warp takes place
      * Output from the warp is clipped to values between 0 and 1
      * If a boolean matrix is desired of the result, use "result > 0.5"

      Parameters:
      -----------
      source : str or gdal.Dataset
          The raster datasource to indicate from

      value : numeric, tuple, iterable or str
          The value, range, or set of values to indicate on
          * If float : The exact value to accept
          * If tuple : The inclusive range to accept. Given as (low,high)
            - Assumes exactly 2 values are present
            - If either value is "None", then the range is assumed to be unbounded on that side
          * If any other iterable : The list of exact values to accept
          * If str : The formatted set of elements to accept
            - Each element in the set is seperated by a ","
            - Each element must be either a singluar numeric value, or a range
            - A range element begins with either "[" or "(", and ends with either "]" or ")"
              and should have an '-' in between
              - "[" and "]" imply inclusivity
              - "(" and ")" imply exclusivity
              - Numbers on either side can be omitted, impling no limit on that side
              - Examples:
                - "[1-5]" -> Indicate values from 1 up to 5, inclusively
                - "[1-5)" -> Indicate values from 1 up to 5, but not including 5
                - "(1-]"  -> Indicate values above 1 (but not including 1) up to infinity
                - "[-5]"  -> Indicate values from negative infinity up to and including 5
                - "[-]"   -> Indicate values from negative infinity to positive infinity (dont do this..)
            - All whitespaces will be ignored (so feel free to use them as you wish)
            - Example:
              - "[-2),[5-7),12,(22-26],29,33,[40-]" will indicate all of the following:
                - Everything below 2, but not including 2
                - Values between 5 up to 7, but not including 7
                - 12
                - Values above 22 up to and including 26
                - 29
                - 33
                - Everything above 40, including 40


      buffer : float; optional
          A buffer region to add around the indicated pixels
          * Units are in the RegionMask's srs
          * The buffering occurs AFTER the indication and warping step and
            so it may not represent the original dataset exactly
            - Buffering can be made more accurate by increasing the
              'resolutionDiv' input

      resolutionDiv : int
          The factor by which to divide the RegionMask's native resolution
          * This is useful if you need to represent very fine details

      resampleAlg : str; optional
          The resampling algorithm to use when warping values
          * Options are: 'near', 'bilinear', 'cubic', 'average', 'mode', 'max', 'min'
          * Knowing which option to use can have significant impacts!
              When indicating from a low resolution raster (relative to the region mask),
              then it is best to use one of 'near', 'bilinear', or 'cubic'. However,
              when indicating from a high resolution raster file (again, relative to the region
              mask) then one of 'average', 'mode', 'max', or 'min' is likely better.

      warpDType : str or None; optional
          If given, this controls the raster datatype of the warped indication matrix.
          If not given, then a default datatype is assumed based off `resampleAlg`:
             reampleAlg : assumed dtype
             ----------   -------------
                 'near' : 'uint8'
             'bilinear' : 'float32'
                'cubic' : 'float32'
              'average' : 'float32'
                 'mode' : 'uint8'
                  'max' : 'uint8'
                  'min' : 'uint8'

      forceMaskShape : bool
          If True, forces the returned matrix to have the same dimension as
          the RegionMask's mask regardless of the 'resolutionDiv' argument

      applyMask : bool
          When True, the RegionMask's mask will be applied to the outputData
          as described by RegionMask.applyMask

      noData : numeric
          The noData value to use when applying the mask

      geomsFromContours: bool
          If True, then geometries will be constructed from the function
          geokit.RegionMask.contoursFromMatrix, as opposed to using
          geokit.RegionMask.polygonizeMask.
          - This will result in simpler geometries which are easier to grow,
            but which do not strictly follow the edges of the indicated pixels


      bufferMethod : str; optional
          An indicator determining the method to use when buffereing
          * Options are: 'area' and 'contour'
          * If 'area', the function will first rasterize the raw geometries and
            will then apply the buffer to the indicated pixels
            - Uses geokit.RegionMask.polygonizeMask
            - This is the safer option although is not as accurate as the 'geom'
              option since it does not capture the exact edges of the geometries
            - This method can be made more accurate by increasing the
              'resolutionDiv' input
          * If 'contour', the function will still rasterize the raw geometries,
            but will then create geometries via mask contours (not the explicit
            pixel edges)
            - Uses geokit.RegionMask.contoursFromMatrix
            - This option will recreate geometries which are more similar to the
              original geometries compared to the 'area' method
            - This method can be made more accurate by increasing the
              'resolutionDiv' input

      preBufferSimplification: numeric; optional
          If given, then geometries will be simplified (using ogr.Geometry.Simplify)
          using the specified value before being buffered
          - Using this can drastically decrease the time it takes to perform the
            bufferring procedure, but can decrease accuracy if it is too high

      prunePatchSize: numeric; optional
          If given, then isolated non-indicated patches below the given size
          will be removed. The given value corresponds to the minimum area in
          the unit of the regionmask SRS that will not be removed. Defaults
          to 0, i.e. no patches will be removed.
          Note: This is applied to the geoms after buffer application and can
          deviate from the patch size after final rasterization.

      threshold: float; optional
          The cell value ABOVE which cells count as positively indicated,
          relevant for partial overlaps with buffer method 'area'. Defaults to 0.5.

      multiProcess: boolean, optional
          If True, multiple parallel processes will be spawned within the function to
          improve RAM efficiency, else it will fall back on linear execution. By default True.
          Only works on Linux and will be deactivated on Windows and Mac.

      kwargs -- Passed on to RegionMask.warp()
          * Most notably: 'resampleAlg'


      Returns:
      --------
      numpy.ndarray



   .. py:method:: indicateFeatures(source, where=None, buffer=None, bufferMethod='geom', resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=0, preBufferSimplification=None, multiProcess=True, **kwargs)

      Indicates the RegionMask pixels which are found within the features (or
      a subset of the features) contained in a given vector datasource

      * A Rasterization is performed from the input data set to the
      RegionMask's mask.
      -See geokit.vector.rasterize or, more specifically gdal.RasterizeOptions
      kwargs for more info on how to control the rasterization step

      Parameters:
      -----------
      source : str or gdal.Dataset
          The vector datasource to indicate from

      where : str; optional
          An SQL-style filtering string
          * Can be used to filter the input source according to their attributes
          * For tips, see "http://www.gdal.org/ogr_sql.html"
          Ex:
          where="eye_color='Green' AND IQ>90"

      buffer : float; optional
          A buffer region to add around the indicated pixels
          * Units are in the RegionMask's srs

      bufferMethod : str; optional
          An indicator determining the method to use when buffereing
          * Options are: 'geom', 'area', and 'contour'
          * If 'geom', the function will attempt to grow each of the geometries
          directly using the ogr library
          - This can fail sometimes when the geometries are particularly
              complex or if some of the geometries are not valid (as in, they
              have self-intersections)
          * If 'area', the function will first rasterize the raw geometries and
          will then apply the buffer to the indicated pixels
          - This is the safer option although is not as accurate as the 'geom'
              option since it does not capture the exact edges of the geometries
          - This method can be made more accurate by increasing the
              'resolutionDiv' input
          * If 'contour', the function will still rasterize the raw geometries,
          but will then create geometries via mask contours (not the explicit
          pixel edges)
          - This option will recreate geometries which are more similar to the
              original geometries compared to the 'area' method
          - This method can be made more accurate by increasing the
              'resolutionDiv' input

      resolutionDiv : int; optional
          The factor by which to divide the RegionMask's native resolution
          * This is useful if you need to represent very fine details

      forceMaskShape : bool; optional
          If True, forces the returned matrix to have the same dimension as
          the RegionMask's mask regardless of the 'resolutionDiv' argument

      applyMask : bool; optional
          When True, the RegionMask's mask will be applied to the outputData
          as described by RegionMask.applyMask

      noData : numeric
          The noData value to use when applying the mask

      preBufferSimplification: numeric
          If given, then geometries will be simplified (using ogr.Geometry.Simplify)
          using the specified value before being buffered
          - Using this can drastically decrease the time it takes to perform the
          bufferring procedure, but can decrease accuracy if it is too high

      multiProcess: boolean, optional
          If True, multiple parallel processes will be spawned within the function to
          improve RAM efficiency, else it will fall back on linear execution. By default True.

      kwargs -- Passed on to RegionMask.rasterize()
          * Most notably: 'allTouched'

      Returns:
      --------
      numpy.ndarray




   .. py:method:: indicateGeoms(geom, **kwargs)

      Convenience wrapper to indicate values found within a geometry (or a
      list of geometries)

      * Simply creates a new vector source from the given geometry and then
        calls RegionMask.indicateFeatures
      * All keywords are passed on to RegionMask.indicateFeatures



   .. py:method:: subRegions(gridSize, asMaskAndExtent=False)

      Generate a number of sub regions on a grid which combine into the total
      RegionMask area



   .. py:method:: subTiles(zoom, checkIntersect=True, asGeom=False)

      Generates tile Extents at a given zoom level which encompass the envoking Regionmask.

      Parameters:
      -----------
      zoom : int
          The zoom level of the expected tile source

      checkIntersect : bool
          If True, exclude tiles which do not intersect with the RegionMask's geometry

      asGeom : bool
          If True, returns tuple of ogr.Geometries in stead of (xi,yi,zoom) tuples

      Returns:
      --------
      Generator of Geometries or (xi,yi,zoom) tuples




   .. py:method:: drawMask(ax=None, **kwargs)

      Convenience wrapper around geokit.util.drawImage which plots the
      RegionMask's mask over the RegionMask's context.

      * See geokit.util.drawImage for more info on argument options
      * Unless specified, the plotting extent is set to the RegionMask's extent
          - This only plays a role when generating a new axis




   .. py:method:: drawImage(matrix, ax=None, drawSelf=True, **kwargs)

      Convenience wrapper around geokit.util.drawImage which plots matrix data
      which is assumed to match the boundaries of the RegionMask

      * See geokit.util.drawImage for more info on argument options
      * Unless specified, the plotting extent is set to the RegionMask's extent
          - This only plays a role when generating a new axis




   .. py:method:: drawGeoms(geoms, ax=None, drawSelf=True, **kwargs)

      Convenience wrapper around geokit.geom.drawGeoms which plots geometries
      which are then plotted within the context of the RegionMask

      * See geokit.geom.drawGeoms for more info on argument options
      * Geometries are always plotted in the RegionMask's SRS
      * Unless specified, x and y limits are set to the RegionMask's extent
          - This only plays a role when generating a new axis



   .. py:method:: drawSelf(ax=None, **kwargs)

      Convenience wrapper around geokit.geom.drawGeoms which plots the
      RegionMask's geometry

      * See geokit.geom.drawGeoms for more info on argument options
      * Geometry are always plotted in the RegionMask's SRS
      * Unless specified, x and y limits are set to the RegionMask's extent
          - This only plays a role when generating a new axis



   .. py:method:: drawRaster(source, ax=None, drawSelf=True, **kwargs)

      Convenience wrapper around geokit.raster.drawRaster which plots a raster
      dataset within the context of the RegionMask

      * See geokit.raster.drawRaster for more info on argument options
      * The raster is always warped to the RegionMask's SRS
      * Unless specified, x and y limits are set to the RegionMask's extent
          - This only plays a role when generating a new axis



   .. py:method:: createRaster(output=None, resolutionDiv=1, **kwargs)

      Convenience wrapper for geokit.raster.createRaster which sets 'srs',
      'bounds', 'pixelWidth', and 'pixelHeight' inputs

      Parameters:
      -----------
      output : str; optional
          A path to an output file to write to

      resolutionDiv : int
          The factor by which to divide the RegionMask's native resolution
          * This is useful if you need to represent very fine details

      **kwargs:
          All other keywargs are passed on to geokit.raster.createRaster()
          * See below for argument descriptions

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: warp(source, output=None, resolutionDiv=1, returnMatrix=True, applyMask=True, noData=None, resampleAlg='bilinear', **kwargs)

      Convenience wrapper for geokit.raster.warp() which automatically sets
      'srs', 'bounds', 'pixelWidth', and 'pixelHeight' inputs

      Note:
      -----
      When creating an 'in memory' raster vs one which is saved to disk, a slightly
      different algorithm is used which can sometimes add an extra row of pixels. Be
      aware of this if you intend to compare value-matricies directly from rasters
      generated with this function.

      Parameters:
      -----------
      source : str
          The path to the raster file to warp

      output : str; optional
          A path to an output file to write to

      resampleAlg : str; optional
          The resampling algorithm to use when warping values
          * Knowing which option to use can have significant impacts!
          * Options are: 'nearesampleAlg=resampleAlg, r', 'bilinear', 'cubic',
            'average'

      resolutionDiv : int
          The factor by which to divide the RegionMask's native resolution
          * This is useful if you need to represent very fine details

      returnAsMatrix : bool
          When True, the resulting raster's matrix is return
          * Should have the same dimensions as the RegionMask's mask matrix

      applyMask : bool
          When True, the RegionMask's mask will be applied to the outputData
          as described by RegionMask.applyMask

      noData : numeric
          The noData value to use when applying the mask

      **kwargs:
          All other keywargs are passed on to geokit.raster.warp()

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: rasterize(source, output=None, resolutionDiv=1, returnMatrix=True, applyMask=True, noData=None, **kwargs)

      Convenience wrapper for geokit.vector.rasterize() which automatically
      sets the 'srs', 'bounds', 'pixelWidth', and 'pixelHeight' inputs

      Note:
      -----
      When creating an 'in memory' raster vs one which is saved to disk, a slightly
      different algorithm is used which can sometimes add an extra row of pixels. Be
      aware of this if you intend to compare value-matricies directly from rasters
      generated with this function.

      Parameters:
      -----------
      source : str
          The path to the vector file to load

      output : str; optional
          A path to an output file to write to

      resolutionDiv : int; optional
          The factor by which to divide the RegionMask's native resolution
          * This is useful if you need to represent very fine details

      returnAsMatrix : bool; optional
          When True, the resulting raster's matrix is return
          * Should have the same dimensions as the RegionMask's mask matrix

      applyMask : bool; optional
          When True, the RegionMask's mask will be applied to the outputData
          as described by RegionMask.applyMask

      noData : numeric; optional
          The noData value to use when applying the mask

      **kwargs:
          All other keywargs are passed on to geokit.vector.rasterize()

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: extractFeatures(source, **kwargs)

      Convenience wrapper for geokit.vector.extractFeatures() by setting the
      'geom' input to the RegionMask's geometry

      Parameters:
      -----------
      source : str
          The path to the vector file to load

      **kwargs:
          All other keyword arguments are passed on to vector.extractFeatures()

      Returns:
      --------
      * If asPandas is True: pandas.DataFrame or pandas.Series
      * If asPandas is False: generator




   .. py:method:: mutateVector(source, matchContext=False, regionPad=0, **kwargs)

      Convenience wrapper for geokit.vector.mutateVector which automatically
      sets 'srs' and 'geom' inputs to the RegionMask's srs and geometry

      * The RegionMask's geometry is always used to select features within the
      source. If you need a broader scope, try using the RegionMask's extent's
      version of this function

      Note:
      -----
      If this is called without any arguments except for a source, it serves
      to clip the vector source around the RegionMask

      Parameters:
      -----------
      source : Anything acceptable to geokit.vector.loadVector()
          The source to clip

      matchContext : bool; optional
          * If True, transforms all geometries to the RegionMask's srs before
            mutating
          * If False, only selects the geometries which touch the RegionMask

      regionPad: int, optional
          Will buffer the regionmask geometry by a value (in the unit of the
          regionmask srs) before mutating the source vector onto it. Defaults
          to 0.

      **kwargs:
          All other keyword arguments are passed to geokit.vector.mutateVector

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: mutateRaster(source, matchContext=True, warpArgs=None, applyMask=True, processor=None, resampleAlg='bilinear', **mutateArgs)

      Convenience wrapper for geokit.vector.mutateRaster which automatically
      sets 'bounds'. It also warps the raster to the RegionMask's area
      and srs before mutating

      Note:
      -----
      If this is called without any arguments except for a source, it serves
      to clip the raster source around the RegionMask, therefore performing
      the same function as RegionMask.warp(..., returnMatrix=False)

      Parameters:
      -----------
      source : Anything acceptable to geokit.raster.loadRaster()
          The source to mutate

      matchContext : bool; optional
          * If True, Warp to the RegionMask's boundaries, srs and pixel size
            before mutating
          * If False, only warp to the RegionMask's boundaries, but keep its
            srs and resolution intact

      resampleAlg : str; optional
          The resampling algorithm to use when warping values
          * Knowing which option to use can have significant impacts!
          * Options are: 'nearesampleAlg=resampleAlg, r', 'bilinear', 'cubic',
            'average'

      warpArgs : dict; optional
          Arguments to apply to the warping step
          * See geokit.raster.warp()

      processor - function; optional
          The function performing the mutation of the raster's data
          * The function will take single argument (a 2D numpy.ndarray)
          * The function must return a numpy.ndarray of the same size as the input
          * The return type must also be containable within a Float32 (int and
            boolean is okay)
          * See example in geokit.raster.mutateRaster for more info

      applyMask : bool; optional
          When True, the RegionMask's mask will be applied to the outputData
          as described by RegionMask.applyMask

      **mutateArgs:
          All other keyword arguments are passed to geokit.vector.mutateVector

      Returns:
      --------
      * If 'output' is None: gdal.Dataset
      * If 'output' is a string: None




   .. py:method:: polygonizeMatrix(matrix, flat=False, shrink=True, _raw=False)

      Convenience wrapper for geokit.geom.polygonizeMatrix which autmatically
      sets the 'bounds' and 'srs' inputs. The matrix data is assumed to span the
      RegionMask exactly.

      Each unique-valued group of pixels will be converted to a geometry

      Parameters:
      -----------
      matrix : matrix_like
          The matrix which will be turned into a geometry set
            * Must be 2 dimensional
            * Must be integer or boolean type

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




   .. py:method:: polygonizeMask(mask, bounds=None, srs=None, flat=True, shrink=True)

      Convenience wrapper for geokit.geom.polygonizeMask which autmatically
      sets the 'bounds' and 'srs' inputs. The mask data is assumed to span the
      RegionMask exactly

      Each True-valued group of pixels will be converted to a geometry

      Parameters:
      -----------
      mask : matrix_like
          The mask which will be turned into a geometry set
            * Must be 2 dimensional
            * Must be boolean type
            * True values are interpreted as 'in the geometry'

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




   .. py:method:: contoursFromRaster(raster, contourEdges, applyMask=True, contoursKwargs={}, warpKwargs={})

      Convenience wrapper for geokit.raster.contours which automatically
      warps a raster to the invoking RegioNmask

      NOTE:
      -----
      * The raster is first warped to the RegionMask before the contours are
        determined. If this behavior is not desired, consider using the function
        Extent.contoursFromRaster

      Parameters:
      -----------
      raster : The raster datasource to warp from

      contourEdges : [float,]
          The edges to search for withing the raster dataset
          * This parameter can be set as "None", in which case an additional
              argument should be given to specify how the edges should be determined
              - See the documentation of "GDALContourGenerateEx"
              - Ex. "LEVEL_INTERVAL=10", contourEdges=None

      contoursKwargs : dict
          Keyword arguments to pass on to the contours function
          * See geokit.raster.contours

      warpKwargs : dict
          Keyword arguments to pass on to the raster warp function
          * See geokit.RegionMask.warp

      Returns:
      --------
      pandas.DataFrame

      With columns:
          'geom' -> The contiguous-valued geometries
          'ID' -> The associated contour edge for each object




   .. py:method:: contoursFromMatrix(matrix, contourEdges, contoursKwargs={}, createRasterKwargs={})

      Convenience wrapper for geokit.raster.contours which autmatically
      creates a raster for the given matrix (which is assumed to match the
      domain of the RegionMask)

      Parameters:
      -----------
      matrix : matrix_like
          The matrix which will be turned into a geometry set
            * Must be 2 dimensional

      contourEdges : [float,]
          The edges to search for withing the raster dataset
          * This parameter can be set as "None", in which case an additional
              argument should be given to specify how the edges should be determined
              - See the documentation of "GDALContourGenerateEx"
              - Ex. "LEVEL_INTERVAL=10", contourEdges=None

      contoursKwargs : dict
          Keyword arguments to pass on to the contours function
          * See geokit.raster.contours

      createRasterKwargs : dict
          Keyword arguments to pass on to the raster creation function
          * See geokit.RegionMask.createRaster

      Returns:
      --------
      pandas.DataFrame

      With columns:
          'geom' -> The contiguous-valued geometries
          'ID' -> The associated contour edge for each object




   .. py:method:: contoursFromMask(mask, truthThreshold=0.5, trueAboveThreshold=True, contoursKwargs={}, createRasterKwargs={})

      Convenience wrapper for geokit.raster.contours which autmatically
      creates a raster for the given mask (which is assumed to match the
      domain of the RegionMask), and extracts the geometries which are indicated
      in the mask as "True"

      Parameters:
      -----------
      mask : matrix_like
          The mask which will be turned into a geometry set
            * Must be 2 dimensional

      truthThreshold : [float,]
          The value which separates "True" from "False" values
          * Values are True when they are above the threshold unless
            trueAboveThreshold is set as False

      trueAboveThreshold: bool
          If true, then pixels with values above the threshold are identified
          as "True"

      contoursKwargs : dict
          Keyword arguments to pass on to the contours function
          * See geokit.raster.contours

      createRasterKwargs : dict
          Keyword arguments to pass on to the raster creation function
          * See geokit.RegionMask.createRaster

      Returns:
      --------
      pandas.DataFrame

      With columns:
          'geom' -> The contiguous-valued geometries
          'ID' -> The associated contour edge for each object




.. py:function:: drawImage(matrix, ax=None, xlim=None, ylim=None, yAtTop=True, scaling=1, fontsize=16, hideAxis=False, figsize=(12, 12), cbar=True, cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap='viridis', cbax=None, cbargs=None, leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0, **kwargs)

   Draw a matrix as an image on a matplotlib canvas

   Parameters:
   -----------
   matrix : numpy.ndarray
       The matrix data to draw

   ax : matplotlib axis; optional
       The axis to draw the geometries on
         * If not given, a new axis is generated and returned

   xlim : (float, float); optional
       The x-axis limits to draw the marix on

   ylim : (float, float); optional
       The y-axis limits to draw the marix on

   yAtTop : bool; optional
       If True, the first row of data should be plotted at the top of the image

   scaling : numeric; optional
       An integer factor by which to scale the matrix before plotting

   figsize : (int, int); optional
       The figure size to create when generating a new axis
         * If resultign figure looks wierd, altering the figure size is your best
           bet to make it look nicer

   fontsize : int; optional
       A base font size to apply to tick marks which appear
         * Titles and labels are given a size of 'fontsize' + 2

   cbar : bool; optional
       If True, a color bar will be drawn

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

   vmax : float; optional
       The maximum value to color

   cmap : str or matplotlib ColorMap; optional
       The colormap to use when coloring

   cbax : matplotlib axis; optional
       An explicitly given axis to use for drawing the colorbar

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

   Returns:
   --------
   A namedtuple containing:
      'ax' -> The map axis
      'handles' -> All geometry handles which were created in the order they were
                   drawn
      'cbar' -> The colorbar handle if it was drawn



