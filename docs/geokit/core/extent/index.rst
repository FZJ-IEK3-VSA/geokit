geokit.core.extent
==================

.. py:module:: geokit.core.extent


Exceptions
----------

.. autoapisummary::

   geokit.core.extent.GeoKitExtentError


Classes
-------

.. autoapisummary::

   geokit.core.extent.IndexSet
   geokit.core.extent.tileBox
   geokit.core.extent.Extent


Module Contents
---------------

.. py:exception:: GeoKitExtentError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: IndexSet

   Bases: :py:obj:`tuple`


   .. py:attribute:: xStart


   .. py:attribute:: yStart


   .. py:attribute:: xWin


   .. py:attribute:: yWin


   .. py:attribute:: xEnd


   .. py:attribute:: yEnd


.. py:class:: tileBox

   Bases: :py:obj:`tuple`


   .. py:attribute:: xi_start


   .. py:attribute:: xi_stop


   .. py:attribute:: yi_start


   .. py:attribute:: yi_stop


   .. py:attribute:: zoom


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




