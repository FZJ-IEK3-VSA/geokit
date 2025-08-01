geokit.core.srs
===============

.. py:module:: geokit.core.srs


Attributes
----------

.. autoapisummary::

   geokit.core.srs.EPSG3035
   geokit.core.srs.EPSG4326
   geokit.core.srs.EPSG3857
   geokit.core.srs.SRSCOMMON


Exceptions
----------

.. autoapisummary::

   geokit.core.srs.GeoKitSRSError


Classes
-------

.. autoapisummary::

   geokit.core.srs.Tile
   geokit.core.srs._SRSCOMMON


Functions
---------

.. autoapisummary::

   geokit.core.srs.loadSRS
   geokit.core.srs.centeredLAEA
   geokit.core.srs.xyTransform
   geokit.core.srs.tileIndexAt


Module Contents
---------------

.. py:exception:: GeoKitSRSError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:function:: loadSRS(source) -> osgeo.osr.SpatialReference

   Load a spatial reference system (SRS) from various sources.

   Parameters:
   -----------
   source : Many things....
       The SRS to load

       Example of acceptable objects are...
         * osr.SpatialReference object
         * An EPSG integer ID
         * A standardized srs str definition such as 'EPSG:4326' or 'ESRI:53003'
         * a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
         * a WKT string

   Returns:
   --------
   osr.SpatialReference



.. py:data:: EPSG3035
   :value: 3035


.. py:data:: EPSG4326
   :value: 4326


.. py:data:: EPSG3857
   :value: 3857


.. py:function:: centeredLAEA(lon=None, lat=None, name='unnamed_m', geom=None)

   Load a Lambert-Azimuthal-Equal_Area spatial reference system (SRS) centered
   on a given set of latitude and longitude coordinates. Alternatively, a geom
   can be passed to center the LAEA on.

   Parameters:
   -----------
   lon : float
       The longitude of the projection's center. Required if no geom is given.

   lat : float
       The latitude of the projection's center. Required if no geom is given.

   geom: osgeo.ogr.Geometry
       The region shape to center the LAEA in. If given, lat and lon must not
       be given, instead they will be defined automatically as the coordinates
       of the region centroid.

   Returns:
   --------
   osr.SpatialReference



.. py:function:: xyTransform(*args, fromSRS='latlon', toSRS='europe_m', outputFormat='raw')

   Transform xy points between coordinate systems

   Parameters:
   -----------
       xy : A single, or an iterable of (x,y) tuples
           The coordinates to transform

       toSRS : Anything acceptable by geokit.srs.loadSRS
           The srs of the output points

       fromSRS : Anything acceptable by geokit.srs.loadSRS
           The srs of the input points

       outputFormat : str
           Determine return value format
           * if 'raw', the raw output from osr.TransformPoints is given
           * if 'xy', or 'xyz' the points are given as named tuples

   Returns:
   --------

   list of tuples, or namedtuple
     * See the point for the 'outputFormat' argument



.. py:class:: Tile

   Bases: :py:obj:`tuple`


   .. py:attribute:: xi


   .. py:attribute:: yi


   .. py:attribute:: zoom


.. py:function:: tileIndexAt(x, y, zoom, srs)

   Get the "slippy tile" index at the given zoom, around the
   coordinates ('x', 'y') within the specified 'srs'


.. py:class:: _SRSCOMMON

   The SRSCOMMON library contains shortcuts and contextual information for various commonly used projection systems

   * You can access an srs in two ways (where <srs> is replaced with the SRS's name):
       1: SRSCOMMON.<srs>
       2: SRSCOMMON["<srs>"]


   .. py:attribute:: _latlon


   .. py:property:: latlon

      Basic SRS for unprojected latitude and longitude coordinates

      Units: Degrees


   .. py:attribute:: _europe_laea


   .. py:attribute:: _europe_m


   .. py:property:: europe_m


   .. py:property:: europe_laea

      Equal-Area projection centered around Europe.

      * Good for relational operations within Europe

      Units: Meters


   .. py:attribute:: _ecowas_laea


   .. py:property:: ecowas_laea

      Equal-Area projection centered around ECOWAS (Western Africa).

      * Good for relational operations within Western Africa

      Units: Meters


   .. py:attribute:: _sadc_laea


   .. py:property:: sadc_laea

      Equal-Area projection centered around ECOWAS (Western Africa).

      * Good for relational operations within Western Africa

      Units: Meters


   .. py:method:: __getitem__(name)


.. py:data:: SRSCOMMON

