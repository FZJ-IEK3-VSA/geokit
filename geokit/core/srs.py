import os
import numpy as np
from osgeo import osr, gdal, ogr
import warnings
from collections import namedtuple
import smopy
from typing import Iterable

from . import util as UTIL
from . import geom as GEOM

class GeoKitSRSError(UTIL.GeoKitError):
    pass


# Basic loader

def loadSRS(source):
    """
    Load a spatial reference system (SRS) from various sources.

    Parameters:
    -----------
    source : Many things....
        The SRS to load

        Example of acceptable objects are...
          * osr.SpatialReference object
          * An EPSG integer ID
          * a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
          * a WKT string

    Returns:
    --------
    osr.SpatialReference

    """
    # Do initial check of source
    if(isinstance(source, osr.SpatialReference)):
        return source
    elif source is None:
        return None

    # Create an empty SRS
    srs = osr.SpatialReference()

    # Check if source is a string
    if(isinstance(source, str)):
        if hasattr(SRSCOMMON, source):
            # assume a name for one of the common SRS's was given
            srs = SRSCOMMON[source]
        else:
            srs.ImportFromWkt(source)  # assume a Wkt string was input
    elif(isinstance(source, int)):
        srs.ImportFromEPSG(source)
    else:
        raise GeoKitSRSError("Unknown srs source type: ", type(source))

    if gdal.__version__ >= '3.0.0':
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # assert that the srs is valid (may be invalid if e.g. wrong integer codes were passed)
    assert srs.Validate()==0, f"Created srs is invalid."

    return srs


# Load a few typical constants
EPSG3035 = loadSRS(3035)
EPSG4326 = loadSRS(4326)
EPSG3857 = loadSRS(3857)


def centeredLAEA(lon=None, lat=None, name="unnamed_m", geom=None):
    """
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

    """
    if geom is None:
        assert isinstance(lat, float) and isinstance(lon, float), "If geom is not passed, lat and lon must be given as float values."
    else:
        assert isinstance(geom, ogr.Geometry), "geom must be given as osgeo.ogr.Geometry class object if not None."
        assert lat is None and lon is None, "If geom is given, lat and lon must not be given."
    
    # check if lat/lon can be used or if it must be extracted from geom first
    if not geom is None:
        # transform to EPSG:4326 in case it not already is lat/lon projection
        geom = GEOM.transform(geom, toSRS=4326)
        # extract lat/lon centroid coordinates to center LAEA upon
        lon = geom.Centroid().GetX()
        lat = geom.Centroid().GetY()

    srs = osr.SpatialReference()
    srs.ImportFromWkt('PROJCS["{}",GEOGCS["GRS 1980(IUGG, 1980)",DATUM["unknown",SPHEROID["GRS80",6378137,298.257222101],TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",{}],PARAMETER["longitude_of_center",{}],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'.format(name, lat, lon))
    
    if gdal.__version__ >= '3.0.0':
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # assert that the srs is valid (may be invalid if e.g. wrong integer codes were passed)
    assert srs.Validate()==0, f"Created srs is invalid."
        
    return srs

####################################################################
# point transformer


def xyTransform(*args, fromSRS='latlon', toSRS='europe_m', outputFormat="raw"):
    """Transform xy points between coordinate systems

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

    """
    # load srs's
    fromSRS = loadSRS(fromSRS)
    toSRS = loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    if len(args) == 0:
        raise GeoKitSRSError("no positional inputs given")
    elif len(args) == 1:
        xy = args[0]
        if isinstance(xy, tuple):
            x, y = xy
            out = [trx.TransformPoint(x, y), ]
        else:
            out = trx.TransformPoints(xy)
    elif len(args) == 2:
        x = np.array(args[0])
        y = np.array(args[1])
        xy = np.column_stack([x, y])

        out = trx.TransformPoints(xy)

    else:
        raise GeoKitSRSError("Too many positional inputs")
    # Done!
    if outputFormat == "raw":
        return out
    elif outputFormat == "xy":
        x = np.array([o[0] for o in out])
        y = np.array([o[1] for o in out])

        TransformedPoints = namedtuple("TransformedPoints", "x y")
        return TransformedPoints(x, y)

    elif outputFormat == "xyz":
        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]

        TransformedPoints = namedtuple("TransformedPoints", "x y z")
        return TransformedPoints(x, y, z)


Tile = namedtuple("Tile", "xi yi zoom")
def tileIndexAt(x, y, zoom, srs):
    """Get the "slippy tile" index at the given zoom, around the 
    coordinates ('x', 'y') within the specified 'srs'
    """
    srs = loadSRS(srs)

    iterable_input = isinstance(x,Iterable) or isinstance(y,Iterable)

    if not srs.IsSame(EPSG4326):
        pt = xyTransform(x,y, 
                fromSRS=srs,
                toSRS=EPSG4326, 
                outputFormat="xy")
        
        if iterable_input:
            x, y = pt.x, pt.y
        else:
            x, y = pt.x[0], pt.y[0]

    if iterable_input:
        x = np.array(x)
        y = np.array(y)

    xi, yi = smopy.deg2num(y, x, zoom)

    return Tile(xi, yi, zoom)

######################################################################################
# Common SRS library

# Some other srs shortcuts


class _SRSCOMMON:
    """The SRSCOMMON library contains shortcuts and contextual information for various commonly used projection systems

    * You can access an srs in two ways (where <srs> is replaced with the SRS's name):
        1: SRSCOMMON.<srs>
        2: SRSCOMMON["<srs>"]
    """
    # basic latitude and longitude
    _latlon = osr.SpatialReference()
    _latlon.ImportFromEPSG(4326)

    @property
    def latlon(self):
        """Basic SRS for unprojected latitude and longitude coordinates

        Units: Degrees"""
        return self._latlon
    # basic latitude and longitude
    _europe_m = osr.SpatialReference()
    _europe_m.ImportFromEPSG(3035)

    @property
    def europe_m(self):
        """Equal-Area projection centered around Europe.

        * Good for relational operations within Europe

        Units: Meters"""
        return self._europe_m

    # define a centered LAEA on the centroid lat/lon of ECOWAS region
    _ecowas_m = centeredLAEA(lon=0.782051665138668, lat=13.564515698612, name="ecowas_m")
    
    @property
    def ecowas_m(self):
        """Equal-Area projection centered around ECOWAS (Western Africa).

        * Good for relational operations within Western Africa

        Units: Meters"""
        return self._ecowas_m
    
    # define a centered LAEA on the centroid lat/lon of SADC region
    _sadc_m = centeredLAEA(lon=26.6605715570689, lat=-14.5952938182064, name="sadc_m")
    
    @property
    def sadc_m(self):
        """Equal-Area projection centered around ECOWAS (Western Africa).

        * Good for relational operations within Western Africa

        Units: Meters"""
        return self._sadc_m

    # basic getter
    def __getitem__(self, name):
        if not hasattr(self, name):
            raise ValueError("SRS \"%s\" not found" % name)
        return getattr(self, "_"+name)


# Initialize
SRSCOMMON = _SRSCOMMON()

################################################