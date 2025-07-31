geokit.core.vector
==================

.. py:module:: geokit.core.vector


Attributes
----------

.. autoapisummary::

   geokit.core.vector._ogrIntToType
   geokit.core.vector._ogrStrToType


Exceptions
----------

.. autoapisummary::

   geokit.core.vector.GeoKitVectorError


Classes
-------

.. autoapisummary::

   geokit.core.vector.vecInfo


Functions
---------

.. autoapisummary::

   geokit.core.vector.loadVector
   geokit.core.vector.loopFeatures
   geokit.core.vector.ogrType
   geokit.core.vector.filterLayer
   geokit.core.vector.countFeatures
   geokit.core.vector.vectorInfo
   geokit.core.vector.listLayers
   geokit.core.vector._extractFeatures
   geokit.core.vector.extractFeatures
   geokit.core.vector.extractFeature
   geokit.core.vector.extractAsDataFrame
   geokit.core.vector.extractAndClipFeatures
   geokit.core.vector.createVector
   geokit.core.vector.createGeoJson
   geokit.core.vector.createGeoDataFrame
   geokit.core.vector.createDataFrameFromGeoDataFrame
   geokit.core.vector.mutateVector
   geokit.core.vector.rasterize
   geokit.core.vector.applyGeopandasMethod


Module Contents
---------------

.. py:exception:: GeoKitVectorError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:function:: loadVector(x)

   Load a vector dataset from a path to a file on disc

   Parameters:
   -----------
   source : str or gdal.Dataset
       * If a string is given, it is assumed as a path to a vector file on disc
       * If a gdal.Dataset is given, it is assumed to already be an open vector
         and is returned immediately

   Returns:
   --------
   gdal.Dataset



.. py:function:: loopFeatures(source)

   Geokit internal

   *Loops over an input layer's features
   * Will reset the reading counter before looping is initiated

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector datasource to read from



.. py:data:: _ogrIntToType

.. py:data:: _ogrStrToType

.. py:function:: ogrType(s)

   Tries to determine the corresponding OGR type according to the input


.. py:function:: filterLayer(layer, geom=None, where=None)

   GeoKit internal

   Filters an ogr Layer object accordint to a geometry and where statement


.. py:function:: countFeatures(source, geom=None, where=None)

   Returns the number of features found in the given source and within a
   given geometry and/or where-statement

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector datasource to read from

   geom : ogr.Geometry; optional
       The geometry to search within
       * All features are extracted which touch this geometry

   where : str; optional
       An SQL-like where statement to apply to the source
       * Feature attribute name do not need quotes
       * String values should be wrapped in 'single quotes'
       Example: If the source vector has a string attribute called "ISO" and
                a integer attribute called "POP", you could use....

           where = "ISO='DEU' AND POP>1000"

   Returns:
   --------
   int



.. py:class:: vecInfo

   Bases: :py:obj:`tuple`


   .. py:attribute:: srs


   .. py:attribute:: bounds


   .. py:attribute:: xMin


   .. py:attribute:: yMin


   .. py:attribute:: xMax


   .. py:attribute:: yMax


   .. py:attribute:: count


   .. py:attribute:: attributes


   .. py:attribute:: source


.. py:function:: vectorInfo(source)

   Extract general information about a vector source

   Determines:

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector datasource to read from

   Returns:
   --------
   namedtuple -> (srs : The source's SRS system,
                  bounds : The source's boundaries (in the srs's units),
                  xMin : The source's xMin boundaries (in the srs's units),
                  yMin : The source's xMax boundaries (in the srs's units),
                  xMax : The source's yMin boundaries (in the srs's units),
                  yMax : The source's yMax boundaries (in the srs's units),
                  count : The number of features in the source,
                  attributes : The attribute titles for the source's features,)



.. py:function:: listLayers(source)

   Returns the layer names for each layer that is stored in a geopackage.

   :param source: The vector datasource to read from
   :type source: Anything acceptable by loadVector()

   :returns: A list of layer names for the source geopackage.
   :rtype: list


.. py:function:: _extractFeatures(source, geom, where, srs, onlyGeom, onlyAttr, skipMissingGeoms, layerName=None, spatialPredicate='Touches')

.. py:function:: extractFeatures(source, where=None, geom=None, srs=None, onlyGeom=False, onlyAttr=False, asPandas=True, indexCol=None, skipMissingGeoms=True, layerName=None, spatialPredicate='Touches', **kwargs)

   Creates a generator which extract the features contained within the source

   * Iteratively returns (feature-geometry, feature-fields)

   Note:
   -----
   Be careful when filtering by a geometry as the extracted features may not
   necessarily be IN the given shape
   * Sometimes they may only overlap
   * Sometimes they are only in the geometry's envelope
   * To be sure an extracted geometry fits the selection criteria, you may
     still need to do further processing or use extractAndClipFeatures()

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector data source to read from

   geom : ogr.Geometry; optional
       The geometry to search with
       * All features are extracted which touch this geometry

   where : str; optional
       An SQL-like where statement to apply to the source
       * Feature attribute name do not need quotes
       * String values should be wrapped in 'single quotes'
       Example: If the source vector has a string attribute called "ISO" and
                a integer attribute called "POP", you could use....

           where = "ISO='DEU' AND POP>1000"

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the geometries to extract
         * If not given, the source's inherent srs is used
         * If srs does not match the inherent srs, all geometries will be
           transformed

   onlyGeom : bool; optional
       If True, only feature geometries will be returned

   onlyAttr : bool; optional
       If True, only feature attributes will be returned

   asPandas : bool; optional
       Whether or not the result should be returned as a pandas.DataFrame (when
       onlyGeom is False) or pandas.Series (when onlyGeom is True)

   indexCol : str; optional
       The feature identifier to use as the DataFrams's index
       * Only useful when as DataFrame is True

   skipMissingGeoms : bool; optional
       If True, then the parser will not read a feature which are missing a geometry

   layerName : str; optional
       The name of the layer to extract from the source vector dataset (only applicable in case of a geopackage).

   spatialPredicate : str, optional
       Applies only in combination with given 'geom' filter. If "Touches",
       all geometries will be extracted that simply touch the filter
       geom. If "Overlaps", geometries to be extracted must overlap (for
       lines, this represents an "Intersect") either partially or
       completely (i.e. it includes "Within"), and if "CentroidWithin"
       the centroid of the extracted geom must be within or on the
       filter geom. By default "Touches".
       NOTE: When filter geom is a polygon, centroids exactly on the
       filter geom boundary will NOT be extracted.

   Returns:
   --------
   * If asPandas is True: pandas.DataFrame or pandas.Series
   * If asPandas is False: generator



.. py:function:: extractFeature(source, where=None, geom=None, srs=None, onlyGeom=False, onlyAttr=False, **kwargs)

   Convenience function calling extractFeatures which assumes there is only
   one feature to extract

   * Will raise an error if multiple features are found

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector datasource to read from

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

   outputSRS : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the geometries to extract
         * If not given, the source's inherent srs is used
         * If srs does not match the inherent srs, all geometries will be
           transformed

   onlyGeom : bool; optional
       If True, only feature geometries will be returned

   onlyAttr : bool; optional
       If True, only feature attributes will be returned

   Returns:
   --------
   * If onlyGeom and onlyAttr are False: namedtuple -> (geom, attr)
   * If onlyGeom is True: ogr.Geometry
   * If onlyAttr is True: dict



.. py:function:: extractAsDataFrame(source, indexCol=None, geom=None, where=None, srs=None, **kwargs)

   Convenience function calling extractFeatures and structuring the output as
   a pandas DataFrame

   * Geometries are written to a column called 'geom'
   * Attributes are written to a column of the same name

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector datasource to read from

   indexCol : str; optional
       The feature identifier to use as the DataFrams's index

   geom : ogr.Geometry; optional
       The geometry to search within
       * All features are extracted which touch this geometry

   where : str; optional
       An SQL-like where statement to apply to the source
       * Feature attribute name do not need quotes
       * String values should be wrapped in 'single quotes'
       Example: If the source vector has a string attribute called "ISO" and
                a integer attribute called "POP", you could use....

           where = "ISO='DEU' AND POP>1000"

   outputSRS : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the geometries to extract
         * If not given, the source's inherent srs is used
         * If srs does not match the inherent srs, all geometries will be
           transformed

   Returns:
   --------
   pandas.DataFrame



.. py:function:: extractAndClipFeatures(source, geom, where=None, srs=None, onlyGeom=False, indexCol=None, skipMissingGeoms=True, layerName=None, scaleAttrs=None, minShare=0.001, **kwargs)

   Extracts features from a source and clips them to the boundaries of a given geom.
   Optionally scales numeric attribute values linearly to the overlapping area share.

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector data source to read from

   geom : ogr.Geometry
       The geometry to search with
       * All features touching this geometry are extracted and clipped to the geometry boundaries.

   where : str; optional
       An SQL-like where statement to apply to the source
       * Feature attribute name do not need quotes
       * String values should be wrapped in 'single quotes'
       Example: If the source vector has a string attribute called "ISO" and
                a integer attribute called "POP", you could use....

           where = "ISO='DEU' AND POP>1000"

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the geometries to extract
         * If not given, the geom's inherent srs is used
         * If srs does not match the inherent srs, all geometries will be
           transformed

   onlyGeom : bool; optional
       If True, only feature geometries will be returned

   indexCol : str; optional
       The feature identifier to use as the DataFrams's index
       * Only useful when as DataFrame is True

   skipMissingGeoms : bool; optional
       If True, then the parser will not read a feature which are missing a geometry

   layerName : str; optional
       The name of the layer to extract from the source vector dataset (only applicable in case of a geopackage).

   scaleAttrs : str or list, optional
       attribute names of the source with numeric values. The values will be scaled linearly with the
       area share of the feature overlapping the geom.

   minShare : float
       The min. area share of a polygon that falls either inside or
       outside the clipping geom. Allows to deal with imperfect boundary
       alignments. 0 means that all clipped geoms, however small they
       may be, are considered. Example: If minShare=0.001 (0.1%), a
       polygon that overlaps with the clipping geom by 99.92% of its
       area will NOT be clipped. If another polygon also at
       minShare=0.001 overlaps by only 0.06% of its area, it will be
       disregarded. By default 0.001.

   Returns:
   --------
   * pandas.DataFrame or pandas.Series


.. py:function:: createVector(geoms, output=None, srs=None, driverName='ESRI Shapefile', layerName='default', fieldVals=None, fieldDef=None, checkAllGeoms=False, overwrite=True)

   Create a vector on disk from geometries or a DataFrame with 'geom' column

   Parameters:
   -----------
   geoms : ogr.Geometry or [ogr.Geometry, ] or pandas.DataFrane
       The geometries to write into the vector file
       * If a DataFrame is given, it must have a column called 'geom'
       * All geometries must share the same type (point, line, polygon, ect...)
       * All geometries must share the same SRS
       * If geometry SRS differs from the 'srs' input, then all geometries will
         be projected to the input srs

   output : str; optional
       A path on disk to create the output vector
       * If output is None, the vector dataset will be created in memory
       * Assumed to be of "ESRI Shapefile" format
       * Will create a number of files with different extensions

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the vector to create
         * If not given, the geometries' inherent srs is used
         * If srs does not match the inherent srs, all geometries will be
           transformed

   driverName : str; optional
       The name of the driver to use when creating the vector.
       Currently supported options are:
           - ESRI Shapefile
           - GPKG

       For a list of all supported vector drivers by OGR, see: https://gdal.org/drivers/vector/index.html

   layerName : str; optional
       The name of the layer to create within the vector. (Only applicable for GeoPackages)
       * If the layer already exists, it will be overwritten

   fieldVals : dict of lists or pandas.DataFrame; optional
       Explicit attribute values to assign to each geometry
       * If a dict is given, must be a dictionary of the attribute names and
         associated lists of the attribute values
       * If a DataFrame is given, the field names are taken from column names
         and attribute values from the corresponding column data
       * The order of each column/list will correspond to the order geometries
         are written into the dataset
       * The length of each column/list MUST match the number of geometries
       * All values in a single column/list must share the same type
           - Options are int, float, or str

   fieldDef : dict; optional
       A dictionary specifying the datatype of each attribute when written into
       the final dataset
       * Options are defined from ogr.OFT[...]
         - ex. Integer, Real, String
       * The ogrType() function can be used to map typical python and numpy types
         to appropriate ogr types

   checkAllGeoms : bool, optional
       If True, all geoms will be asserted in object type and exact srs. Else, only
       the first geom in the geom column/iterable will be checked fpr performance reasons.
       By default False.

   overwrite : bool; optional
       Determines whether the prexisting files should be overwritten
       * Only used when output is not None

   Returns:
   --------
   * If 'output' is None: gdal.Dataset
   * If 'output' is given: None



.. py:function:: createGeoJson(geoms, output=None, srs=4326, topo=False, fill='')

   Convert a set of geometries to a geoJSON object


.. py:function:: createGeoDataFrame(dfGeokit: pandas.DataFrame)

   Creates a gdf from an Reskit shape pd.DataFrame

   :param dfGeokit: Reskit shape pd.DataFrame, need a 'geom' column.
   :type dfGeokit: pd.DataFrame

   :returns: Same as the previos, just as an GeodataFrame
   :rtype: gpd.GeoDataFrame


.. py:function:: createDataFrameFromGeoDataFrame(gdf: pandas.DataFrame)

   Creates a geokit-style dataframe from a geopandas geodataframe

   :param gdf: geopandas-style pd.DataFrame, need a 'geometry' column.
   :type gdf: pd.DataFrame

   :returns: Same as the previos, just as an geokit-style dataframe with 'geom'
             column with osgeo.ogr.Geometry objects.
   :rtype: pd.DataFrame


.. py:function:: mutateVector(source, processor=None, srs=None, geom=None, where=None, fieldDef=None, output=None, keepAttributes=True, _slim=False, **kwargs)

   Process a vector dataset according to an arbitrary function

   Note:
   -----
   If this is called without a processor, it simply clips the vector source to
   the selection criteria given by 'geom' and 'where' as well as translates
   all geometries to 'srs'

   Parameters:
   -----------
   source : Anything acceptable by loadVector()
       The vector datasource to read from

   processor : function; optional
       The function which mutates each feature
       * If no function is given, this stage is skipped
       * The function will take 1 arguments: a pandas.Series object containing
         one 'geom' key indicating the geometry and the other keys indicating
         attributes
       * The function must return something understandable by pd.Series
         containing the geometry under the index 'geom' and any other keys
           - These will be used to update the old geometries and attributes
       * The attribute dictionary's values should only be numerics and strings
       * See example below for more info

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the vector to create
         * If not given, the source's inherent srs is used
         * If the given SRS is different from the source's SRS, all feature
           geometries will be cast to the given SRS before processing

   geom : ogr.Geometry; optional
       The geometry to search within
       * All features are extracted which touch this geometry

   where : str; optional
       An SQL-like where statement to apply to the source
       * Feature attribute name do not need quotes
       * String values should be wrapped in 'single quotes'
       Example: If the source vector has a string attribute called "ISO" and
                a integer attribute called "POP", you could use....

           where = "ISO='DEU' AND POP>1000"

   fieldDef : dict; optional
       A dictionary specifying the datatype of each attribute when written into
       the final dataset
       * Options are defined from ogr.OFT[...]
         - ex. Integer, Real, String
       * The ogrType() function can be used to map typical python and numpy types
         to appropriate ogr types

   output : str; optional
       A path on disk to create the output vector
       * If output is None, the vector dataset will be created in memory
       * Assumed to be of "ESRI Shapefile" format
       * Will create a number of files with different extensions

   keepAttributes : bool; optional
       If True, the old attributes will be kept in the output vector
           * Unless they are over written by the processor
       If False, only the newly specified attributes are kept

   Returns:
   --------
   * If 'output' is None: gdal.Dataset
   * If 'output' is given: None

   Example:
   --------
   Say you had a vector source which contains point geometries, and where each
   feature also had an float-attribute called "value". You want to create a new
   vector set wherein you have circle geometries at the same locations as the
   original points and whose radius is equal to the original features' "value"
   attribute. Furthermore, let's say you only want to do this for feature's who's
   "value" attribute is greater than zero and less than 10. Do as such...

   >>> def growPoints( row ):
   >>>     # Create a new geom
   >>>     newGeom = row.geom.Buffer(row.radius)
   >>>
   >>>     # Return the new geometry/attribute set
   >>>     return { 'geom':newGeom }
   >>>
   >>> result = processVector( <source-path>, where="value>0 AND value<10",
   >>>                         processor=growPoints )
   >>>



.. py:function:: rasterize(source, pixelWidth, pixelHeight, srs=None, bounds=None, where=None, value=1, output=None, dtype=None, compress=True, noData=None, overwrite=True, fill=None, **kwargs)

   Rasterize a vector datasource onto a raster context

   Note:
   -----
   When creating an 'in memory' raster vs one which is saved to disk, a slightly
   different algorithm is used which can sometimes add an extra row of pixels. Be
   aware of this if you intend to compare value-matricies directly from rasters
   generated with this function.

   Parameters:
   -----------
   source : str or ogr.Geometry
       If str, the path to the vector file to load
       If ogr.Geometry, an Polygon geometry
           - Will be immediately turned into a vector

   pixelWidth : numeric
       The pixel width of the raster in the working srs
       * Is 'srs' is not given, these are the units of the source's inherent srs

   pixelHeight : numeric
       The pixel height of the raster in the working srs
       * Is 'srs' is not given, these are the units of the source's inherent srs

   srs : Anything acceptable to geokit.srs.loadSRS(); optional
       The srs of the point to create
       * If 'bounds' is an Extent object, the bounds' internal srs will override
         this input

   bounds : (xMin, yMix, xMax, yMax) or Extent; optional
       The geographic extents spanned by the raster
       * If not given, the whole bounds spanned by the input is used

   where : str; optional
       An SQL-like where statement to use to filter the vector before rasterizing

   value : numeric, str
       The values to burn into the raster
       * If a numeric is given, all pixels are burned with the specified value
       * If a string is given, then one the feature attribute names is expected

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

   Returns:
   --------
   * If 'output' is None: gdal.Dataset
   * If 'output' is a string: The path to the output is returned (for easy opening)



.. py:function:: applyGeopandasMethod(geopandasMethod, *dfs, **kwargs)

   Convenience function to apply geopandas methods to a geokit-style
   dataframe with 'geom' column with osgeo.ogr.Geometry objects.
   NOTE: All arguments besides **kwargs must be passed as positional
   arguments.

   geopandasMethod : str, executable
       Geopandas method to apply to the dataframe, either str-formatted
       method name or method as a callable function.
   *dfs : pd.DataFrames
       One or multiple comma-separated pd.DataFrames with 'geom' column
       with osgeo.ogr.Geometry objects. Will be passed to the geopandas
       method as positional arguments, starting from the first position.
   **kwargs
       Will be passed on to the geopandas function.


