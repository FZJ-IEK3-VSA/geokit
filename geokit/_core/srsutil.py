from .util import *

######################################################################################
# Common SRS library

# Some other srs shortcuts
class _SRSCOMMON:
    # basic lattitude and longitude
    _latlon = osr.SpatialReference()
    _latlon.ImportFromEPSG(4326)

    @property
    def latlon(s):
        """Basic SRS for unprojected latitude and longitude coordinates

        Units: Degrees"""
        return s._latlon
    # basic lattitude and longitude
    _europe_m = osr.SpatialReference()
    _europe_m.ImportFromEPSG(3035)

    @property
    def europe_m(s):
        """Equal-Area projection centered around Europe.

        * Good for relational operations within Europe

        Units: Meters"""
        return s._europe_m
    
    # basic getter
    def __getitem__(s, name):
        if not hasattr( s, name ):
            raise ValueError("SRS \"%s\" not found"%name)
        return getattr(s, "_"+name)

# Initialize
SRSCOMMON = _SRSCOMMON()

################################################
# Basic loader
def loadSRS(source):
    """
    Load a spatial reference system from various sources.
    """
    # Do initial check of source
    if(isinstance(source, osr.SpatialReference)):
        return source
    
    # Create an empty SRS
    srs = osr.SpatialReference()

    # Check if source is a string
    if( isinstance(source,str) ):
        if hasattr(SRSCOMMON, source): 
            srs = SRSCOMMON[source] # assume a name for one of the common SRS's was given
        else:
            srs.ImportFromWkt(source) # assume a Wkt string was input
    elif( isinstance(source,int) ):
        srs.ImportFromEPSG(source)
    else:
        raise GeoKitSRSError("Unknown srs source type: ", type(source))
        
    return srs

# Load a few typical constants
EPSG3035 = loadSRS(3035)
EPSG4326 = loadSRS(4326)

####################################################################
# point transformer
def xyTransform( xy, fromSRS='latlon', toSRS='europe_m'):
    """Transform points between coordinate systems

    Inputs:
        xy 
            (float,float) -- X and Y coordinates
            [ (x1,y1), (x2,y2), ] -- A list of XY coordinates

    """
    # load srs's
    fromSRS = loadSRS(fromSRS)
    toSRS = loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    if isinstance( xy, tuple):
        x, y = xy
        out = trx.TransformPoint(x,y)
    else:
        out = trx.TransformPoints(xy)
        
    # Done!
    return out

