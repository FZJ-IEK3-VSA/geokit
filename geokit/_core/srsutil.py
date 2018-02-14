from .util import *

######################################################################################
# Common SRS library

# Some other srs shortcuts
class _SRSCOMMON:
    """The SRSCOMMON library contains shortcuts and contextual information for various commonly used projection systems
    
    * You can access an srs in two ways (where <srs> is replaced with the SRS's name):
        1: SRSCOMMON.<srs>
        2: SRSCOMMON["<srs>"]
    """
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
def xyTransform( *args, fromSRS='latlon', toSRS='europe_m', outputFormat="raw"):
    """Transform xy points between coordinate systems

    Inputs:
        xy  : The coordinates to transform
            - (float,float) -- X and Y coordinates
            - [ (x1,y1), (x2,y2), ] -- A list of XY coordinates

        toSRS : The srs of the output points
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string

        fromSRS : The srs of the input points
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string

        outputFOrmat - str : Use this to control how the return value is given
            * if 'raw', the raw output from osr.TransformPoints is given
            * if 'xy', or 'xyz' the points are given as named tuples

    """
    # load srs's
    fromSRS = loadSRS(fromSRS)
    toSRS = loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    if len(args)==0: raise GeoKitSRSError("no positional inputs given")
    elif len(args)==1:
        xy = args[0]
        if isinstance( xy, tuple):
            x, y = xy
            out = [trx.TransformPoint(x,y), ]
        else:
            out = trx.TransformPoints(xy)
    elif len(args)==2:
        x = np.array(args[0])
        y = np.array(args[1])
        xy = np.column_stack( [x,y] )

        out = trx.TransformPoints(xy)

    else: 
        raise GeoKitSRSError("Too many positional inputs")
    # Done!
    if outputFormat=="raw":
        return out
    elif outputFormat=="xy":
        x = np.array([o[0] for o in out])
        y = np.array([o[1] for o in out])

        TransformedPoints = namedtuple("TransformedPoints", "x y")
        return TransformedPoints(x, y)
    
    elif outputFormat=="xyz":
        x = out[:,0]
        y = out[:,1]
        z = out[:,2]

        TransformedPoints = namedtuple("TransformedPoints", "x y z")
        return TransformedPoints(x, y, z)

