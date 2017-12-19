from .util import *
from .srsutil import *
from .geomutil import *
import types
import re

LocationMatcher = re.compile("\((?P<lon>[0-9.-]{1,}),(?P<lat>[0-9.-]{1,})\)")

class Location(object):
    _e = 1e-5
    def __init__(s, lon, lat): 
        if not (isinstance(lat,float) or isinstance(lat,int)): raise GeoKitLocationError("lat input is not a float")
        if not (isinstance(lon,float) or isinstance(lon,int)): raise GeoKitLocationError("lon input is not a float")
        s.lat=lat
        s.lon=lon
        s._geom=None

    def __hash__(s): # I need this to make pandas indexing work when location objects are used as columns and indexes
        return hash((int(s.lon/s._e), int(s.lat/s._e)))

    def __eq__(s,o):
        if isinstance(o,Location):
            return abs(s.lon-o.lon)<s._e and abs(s.lat-o.lat)<s._e
        elif isinstance(o,ogr.Geometry):
            return s==Location.fromPointGeom(o)
        elif isinstance(o, tuple) and len(o)==2:
            return abs(s.lon-o[0])<s._e and abs(s.lat-o[1])<s._e
        else:
            return False

    def __ne__(s,o):
        return not(s==o)

    def __str__(s):
        return "(%.5f,%.5f)"%(s.lon,s.lat)

    def __repr__(s):
        return s.__str__()

    @staticmethod
    def fromString(s, srs=None):
        m = LocationMatcher.search(s)
        if m is None: raise GeoKitLocationError("string does not match Location specification")

        lon,lat = m.groups()
        if srs is None:
            return Location(lon=float(lon),lat=float(lat))
        else:
            return Location.fromXY(lon=float(lon),lat=float(lat),srs=srs)


    @staticmethod
    def fromPointGeom(g):
        if g.GetGeometryName()!="POINT":
            raise GeoKitLocationError("Invalid geometry given")
        if not g.GetSpatialReference().IsSame(EPSG4326):
            g = g.Clone()
            g.TransformTo(EPSG4326)

        return Location(lon=g.GetX(), lat=g.GetY())

    @staticmethod
    def fromXY(x,y,srs=3035):
        g = point(x,y,srs=srs)
        return Location.fromPointGeom(g)
    @property
    def latlon(s): return s.lat,s.lon

    def asGeom(s,srs='latlon'):
        g = s.geom
        return transform(g, toSRS=srs)

    def asXY(s,srs=3035): 
        g = s.asGeom(srs=srs)
        return g.GetX(), g.GetY()

    @property
    def geom(s):
        if s._geom is None: 
            s._geom = point(s.lon,s.lat,srs=EPSG4326)
        return s._geom
            
    @staticmethod
    def makePickleable(loc):
        if isinstance(loc,Location):
            loc._geom = None
        else: 
            [Location.makePickleable(l) for l in loc]

    @staticmethod
    def ensureLocation(locations, srs=None, forceAsArray=False, isiterable=False):
        if isinstance(locations, Location): output = locations

        elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
            output = Location.fromPointGeom(locations)
        
        elif isinstance(locations, str):
            output = Location.fromString(locations)
        
        elif isinstance(locations, Feature):
            output = Location.ensureLocation(locations.geom)
        
        elif not isiterable and ((isinstance(locations, tuple) or isinstance(locations, list) or isinstance(locations, np.ndarray))) and len(locations)==2:
            try:
                if srs is None:
                    output = Location(lon=locations[0], lat=locations[1])
                else:
                    output = Location.fromXY(lon=locations[0], lat=locations[1], srs=srs)
            except:
                output = Location.ensureLocation(locations, srs=srs, forceAsArray=forceAsArray, isiterable=True)
        else: # Assume iteratable
            try:
                output = np.array([Location.ensureLocation(l, srs=srs) for l in locations])
            except:
                raise GeoKitLocationError("Could not understand location input")

        # Done!
        if forceAsArray:
            if isinstance(output, np.ndarray): pass
            else:
                output = np.array([output, ])
        return output