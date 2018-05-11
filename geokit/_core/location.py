from .util import *
from .srsutil import *
from .geomutil import *
import types
import re

LocationMatcher = re.compile("\((?P<lon>[0-9.-]{1,}),(?P<lat>[0-9.-]{1,})\)")

class Location(object):
    _TYPE_KEY_="Location"
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
            
    def makePickleable(s): 
        s._geom = None
        
    @staticmethod
    def load(loc, srs=4326):
        if   isinstance(loc, Location): output = loc
        elif isinstance(loc, ogr.Geometry): output = Location.fromPointGeom(loc)
        elif isinstance(loc, str): output = Location.fromString(loc)
        elif isinstance(loc, Feature): output = Location.ensureLocation(loc.geom)
        
        elif ((isinstance(loc, tuple) or isinstance(loc, list) or isinstance(loc, np.ndarray))) and len(loc)==2:
            if srs is None or srs == 4326 or srs=='latlon' or EPSG4326.IsSame(srs):
                output = Location(lon=loc[0], lat=loc[1])
            else:
                output = Location.fromXY(x=loc[0], y=loc[1], srs=srs)
        else: # Assume iteratable
            raise GeoKitLocationError("Could not understand location input:", loc)

        return output

class LocationSet(object):
    _TYPE_KEY_="LocationSet"
    def __init__(s, locations, srs=4326, _skip_check=False):
        if not _skip_check:
            try: # Try loading all locations one at a time
                s._locations = np.array([Location.load(l, srs=srs) for l in locations])
            except GeoKitLocationError as err:
                try:
                    # Try loading the input as as single Location
                    s._locations = np.array([Location.load(locations, srs=srs), ])
                except GeoKitLocationError:
                    raise err
        else: 
            s._locations = locations

        s._lons = None
        s._lats = None
        s._bounds4326 = None
        s.count = len(s._locations)

    def __getitem__(s,i): return s._locations[i]
    def __repr__(s):
        out = " , Lon      , Lat\n"
        if s.count > 10:
            for i in range(5): out += "%d, %-9.5f, %-9.5f\n"%(i, s[i].lon, s[i].lat)
            out += "...\n"
            for i in range(5): out += "%d, %-9.5f, %-9.5f\n"%(s.count-6+i, s[-6+i].lon, s[-6+i].lat)
        else:
            for i in range(s.count): out += "%d, %-9.5f, %-9.5f\n"%(i, s[i].lon, s[i].lat)

        return out

    def getBounds(s, srs=4326):
        if srs==4326 and not s._bounds4326 is None:
            return s._bounds4326
        elif srs==4326:
            s._bounds4326 = (s.lons.min(), s.lats.min(), s.lons.max(), s.lats.max())
            return s._bounds4326
        else:
            geoms = transform( [l.geom for l in s._locations], fromSrs=EPSG4326, toSRS=srs )

            yVals = np.array([g.GetY() for g in geoms])
            xVals = np.array([g.GetX() for g in geoms])

            return (xVals.min(), yVals.min(), xVals.max(), yVals.max())

    @property
    def lats(s): 
        if s._lats is None: s._lats = np.array([l.lat for l in s._locations])
        return s._lats

    @property
    def lons(s): 
        if s._lons is None: s._lons = np.array([l.lon for l in s._locations])
        return s._lons

    def asString(s):
        return [str(l) for l in s._locations]

    def makePickleable(loc):
        for l in s._locations: l.makePickleable()

    def asGeom(s, srs=4326):
        srs=loadSRS(srs)
        geoms4326 = [l.geom for l in s._locations]
        if EPSG4326.IsSame(srs): return geoms4326
        else: return transform(geoms4326, fromSRS=EPSG4326, toSRS=srs)

    def asXY(s, srs=3035):
        srs=loadSRS(srs)
        if EPSG4326.IsSame(srs): return np.column_stack([s.lons, s.lats])
        else: 
            geoms4326 = [l.geom for l in s._locations]
            geomsSRS = transform(geoms4326, fromSRS=EPSG4326, toSRS=srs)
            return np.array([(g.GetX(), g.GetY()) for g in geomsSRS])

    def asHash(s): return [l.__hash__() for l in s._locations]

    def splitKMeans(s, groups=2, **kwargs):
        from sklearn.cluster import KMeans

        obs = np.column_stack([s.lons,s.lats])

        km = KMeans(n_clusters=groups, **kwargs).fit(obs)
        for i in range(groups):
            sel = km.labels_ == i
            yield LocationSet( s[sel], _skip_check=True)

    def bisect(s, lon=True, lat=True, delta=0.005):
        MAX_ATTEMPTS = 100

        lonDiv = np.median(s.lons)
        latDiv = np.median(s.lats)

        if lon and lat:
            yield LocationSet( s[(s.lons<lonDiv) & (s.lats<latDiv)], _skip_check=True )
            yield LocationSet( s[(s.lons>=lonDiv) & (s.lats<latDiv)], _skip_check=True )
            yield LocationSet( s[(s.lons<lonDiv) & (s.lats>=latDiv)], _skip_check=True )
            yield LocationSet( s[(s.lons>=lonDiv) & (s.lats>=latDiv)], _skip_check=True )

        elif lon and not lat:
            yield LocationSet( s[(s.lons<lonDiv)], _skip_check=True )
            yield LocationSet( s[(s.lons>=lonDiv)], _skip_check=True )

        elif lat and not lon:
            yield LocationSet( s[(s.lats<latDiv)], _skip_check=True )
            yield LocationSet( s[(s.lats>=latDiv)], _skip_check=True )

