geokit.core.regionmask
======================

.. py:module:: geokit.core.regionmask


Exceptions
----------

.. autoapisummary::

   geokit.core.regionmask.GeoKitRegionMaskError


Classes
-------

.. autoapisummary::

   geokit.core.regionmask.MaskAndExtent
   geokit.core.regionmask.RegionMask


Functions
---------

.. autoapisummary::

   geokit.core.regionmask.usage


Module Contents
---------------

.. py:function:: usage()

.. py:exception:: GeoKitRegionMaskError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: MaskAndExtent

   Bases: :py:obj:`tuple`


   .. py:attribute:: mask


   .. py:attribute:: extent


   .. py:attribute:: id


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




