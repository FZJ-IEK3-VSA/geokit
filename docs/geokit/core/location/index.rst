geokit.core.location
====================

.. py:module:: geokit.core.location


Attributes
----------

.. autoapisummary::

   geokit.core.location.LocationMatcher


Exceptions
----------

.. autoapisummary::

   geokit.core.location.GeoKitLocationError


Classes
-------

.. autoapisummary::

   geokit.core.location.Location
   geokit.core.location.LocationSet


Module Contents
---------------

.. py:exception:: GeoKitLocationError

   Bases: :py:obj:`geokit.core.util.GeoKitError`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:data:: LocationMatcher

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



