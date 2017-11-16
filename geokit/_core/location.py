from .util import *
from .srsutil import *
from .geomutil import *
import types
import re

LocationMatcher = re.compile("Location\((?P<lon>[0-9.]{1,}),(?P<lat>[0-9.]{1,})\)")

class Location(object):
    _e = 1e-5
    def __init__(s, lon, lat): 
        if not isinstance(lat,float): raise GeoKitLocationError("lat input is not a float")
        if not isinstance(lon,float): raise GeoKitLocationError("lon input is not a float")
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
        return "Location(%.5f,%.5f)"%(s.lon,s.lat)

    def __repr__(s):
        return "lat: %8f    lon: %8f"%(s.lat,s.lon)

    @staticmethod
    def fromString(s):
        m = LocationMatcher.search(s)
        if m is None: raise GeoKitLocationError("string does not match Location specification")

        lon,lat = m.groups()
        return Location(lon=float(lon),lat=float(lat))


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
    def ensureLocation(locations, srs=None, forceAsArray=False):
        if isinstance(locations, Location): output = locations
        elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
            output = Location.fromPointGeom(locations)
        elif isinstance(locations, Feature):
            output = Location.ensureLocation(locations.geom)
        elif (isinstance(locations, tuple) or isinstance(locations, list)) and len(locations)==2:
            if srs is None:
                output = Location(lon=locations[0], lat=locations[1])
            else:
                output = Location.fromXY(lon=locations[0], lat=locations[1], srs=srs)

        elif isinstance(locations, list) or ( isinstance(locations, np.ndarray) and len(locations.shape)==1):
            if isinstance(locations[0], Location): output = locations # if the first item is a Location, assume they all are
            else: output = np.array([Location.ensureLocation(l) for l in locations])

        elif isinstance(locations, np.ndarray) and locations.shape[1]==2:
            if srs is None:
                output = np.array([Location(lon=loc[0], lat=loc[1]) for loc in locations])
            else:
                output = np.array([Location.fromXY(lon=loc[0], lat=loc[1], srs=srs) for loc in locations])

            if output.shape[0]==1: output = output[0]

        elif isinstance(locations, types.GeneratorType):
            output = np.array([Location.ensureLocation(loc) for loc in locations])
            if output.shape[0]==1: output = output[0]
        else:
            print(locations)
            raise GeoKitLocationError("Cannot understand location input. Use either a Location, tuple, list, or an ogr.Geometry object")

        # Done!
        if forceAsArray:
            if isinstance(output, np.ndarray): return output
            
            if not isinstance(output, list): output = [output,]

            finalOut = np.ndarray(shape=len(output),dtype=np.object)
            for i,l in enumerate(output): finalOut[i]=l

            return finalOut
        else:
            return output