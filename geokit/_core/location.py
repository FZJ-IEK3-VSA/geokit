from .util import *
from .srsutil import *
from .geomutil import *
import types


LocationNT = namedtuple("Location", "lon lat")
class Location(object):
    _e = 1e-6
    def __init__(s, lon, lat): 
        s.lat=lat
        s.lon=lon

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
        return "%8f,%8f"%(s.lon,s.lat)

    def __repr__(s):
        return "lat: %8f    lon: %8f"%(s.lat,s.lon)

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
        return transform(s.geom.Clone(), toSRS=srs)

    def asXY(s,srs=3035): 
        g = s.asGeom(srs=srs)
        return g.GetX(), g.GetY()

    @property
    def geom(s):
        try: return s._geom
        except:
            s._geom = point(s.lon,s.lat,srs=EPSG4326)
            return s._geom

    @staticmethod
    def ensureLocation(locations, srs=None, forceAsArray=False):
        if isinstance(locations, Location): output = locations
        elif isinstance(locations, ogr.Geometry): # Check if loc is a single point
            output = Location.fromPointGeom(locations)
        elif isinstance(locations, tuple) and len(locations)==2:
            if srs is None:
                output = Location(lon=locations[0], lat=locations[1])
            else:
                output = Location.fromXY(locations[0], locations[1], srs=srs)

        elif isinstance(locations, list) or isinstance(locations, np.ndarray):
            if isinstance(locations[0], Location): output = locations
            else: output = [Location.ensureLocation(l) for l in locations]

        elif isinstance(locations, types.GeneratorType):
            output = Location.ensureLocation(list(locations))

        else:
            raise GeoKitLocationError("Cannot understand location input. Use either a Location or an ogr.Geometry object")

        # Done!
        if forceAsArray:
            if isinstance(output, np.ndarray): return output
            
            if not isinstance(output, list): output = [output,]

            finalOut = np.ndarray(shape=len(output),dtype=np.object)
            for i,l in enumerate(output): finalOut[i]=l

            return finalOut
        else:
            return output