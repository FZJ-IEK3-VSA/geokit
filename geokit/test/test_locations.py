from helpers import *
from geokit.geom import *
from geokit import Location, LocationSet

xy = (9,5)

def test_Location___init__():
    l = Location(*xy)
    print("Location___init__ passed")

def test_Location___hash__():
    l = Location(*xy)
    compare(hash(l), 4109769254643766781)
    print( "Location___hash__ passed")

def test_Location___eq__():
    # TEst against locations
    l1 = Location(*xy)
    l2 = Location(*xy)
    
    if not l1 == l2: error("Location matching")

    l3 = Location(xy[0], xy[1]+0.001)
    if l1 == l3: error("Location matching")

    # test against tuple
    if not l1 == xy: error("Location matching")

    # test against geometry
    pt = point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)

    if not l1 == pt: error("Location matching")
    
    print( "Location___eq__ passed")

def test_Location___ne__():
    print( "Location___ne__ is trivial")

def test_Location___str__():
    
    l1 = Location(*xy)
    if not str(l1)=='(9.00000,5.00000)': error("string representation")

    print( "Location___str__ passed")

def test_Location___repr__():
    print( "Location___repr__ is trivial")

def test_Location_fromString():
    l1 = Location(*xy)

    okay = ['( 9.00000,5.00000)',
            ' (9.00000,5.00000)',
            '(9.00000,5.00000) ',
            ' ( 9.00000,5.00000) ',
            ' qweqdada( 9.00000,5.00000)adfafdq ',
            ' ( 9.00,  5) ',
            ' ( 9.00000,  5.00000) ']
    
    for p in okay:
        if not l1==Location.fromString(p): error("fromString")
    
    print( "Location_fromString passed")

def test_Location_fromPointGeom():
    l1 = Location(*xy)

    pt = point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)
    l2 = Location.fromPointGeom(pt)
    if not l1==l2: error("fromPointGeom")

    print( "Location_fromPointGeom passed")

def test_Location_fromXY():
    l1 = Location(*xy)
    
    pt = point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)

    l2 = Location.fromXY(pt.GetX(), pt.GetY(), srs=EPSG3035)
    if not l1==l2: error("fromXY")

    print( "Location_fromXY passed")

def test_Location_latlon():
    print( "Location_latlon is trivial")

def test_Location_asGeom():
    print( "Location_asGeom is trivial")

def test_Location_asXY():
    print( "Location_asXY is trivial")

def test_Location_geom():
    print( "Location_geom is trivial")

def test_Location_makePickleable():
    print( "Location_makePickleable is trivial")

def test_Location_load():
    l1 = Location(*xy)

    if not l1==Location.load(l1): error("load")


    # From pt
    pt = point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)
    if not l1==Location.load(pt): error("load")

    # From xy
    if not l1==Location.load(xy): error("load")

    # From str
    if not l1==Location.load(' ( 9.00000,5.00000) ',): error("load")

    # From xy with srs
    xy_3035 = pt.GetX(), pt.GetY()

    if not l1==Location.load(xy_3035, srs=3035): error("load")

    if not l1==Location.load( list(xy_3035), srs=3035): error("load")

    if not l1==Location.load( np.array(xy_3035), srs=3035): error("load")

    print( "Location_load passed")

def test_LocationSet___init__():
    # From xy list
    ls = LocationSet(pointsInAachen4326)

    # From numpy array
    ls2 = LocationSet( np.array(pointsInAachen4326))
    if not ls[1] == ls2[1]: error("LocationSet init")

    # From numpy array with srs change
    ls2 = LocationSet( np.array(pointsInAachen3035), srs=3035)
    if not ls[1] == ls2[1]: error("LocationSet init")

    # From single pt
    ls3 = LocationSet(xy)
    if not ls3.count==1: error("LocationSet single xy")

    # From single geom
    pt = point(*xy, srs=4326)
    ls4 = LocationSet(pt)
    if not ls4.count==1: error("LocationSet single geom")
    if not ls3[0]==ls4[0]: error("LocationSet single xy")

    # From many geoms
    pts = [ point(x,y, srs=4326) for x,y in np.random.random(size=(10,2)) ]
    ls5 = LocationSet(pts)
    if not ls5.count==10: error("LocationSet single geom")
   
    print( "LocationSet___init__ passed")

def test_LocationSet___getitem__():
    print( "LocationSet___getitem__ is trivial")

def test_LocationSet___repr__():
    print( "LocationSet___repr__ is trivial")

def test_LocationSet_getBounds():
    ls = LocationSet(pointsInAachen4326)

    bounds = ls.getBounds(3035)

    compare(bounds[0], 4039553.1900635841)
    compare(bounds[1], 3052769.5385426758)
    compare(bounds[2], 4065568.4155270099)
    compare(bounds[3], 3087947.74365965)

    print( "LocationSet_getBounds passed")

def test_LocationSet_asString():
    pts = [(2,3), (4,2), (5,5)]
    ls = LocationSet(pts)

    s = ls.asString()
    if not s[1]=="(4.00000,2.00000)": error("asString")

    print( "LocationSet_asString passed")

def test_LocationSet_makePickleable():
    print( "LocationSet_makePickleable is trivial")

def test_LocationSet_asGeom():
    print( "LocationSet_asGeom is trivial")

def test_LocationSet_asXY():
    print( "LocationSet_asXY is trivial")

def test_LocationSet_asHash():
    print( "LocationSet_asHash is trivial")

def test_LocationSet_splitKMeans():
    print( "LocationSet_splitKMeans is not tested...")

def test_LocationSet_bisect():
    print( "LocationSet_bisect is not tested...")


if __name__ == "__main__":
    test_Location___init__()
    test_Location___hash__()
    test_Location___eq__()
    test_Location___ne__()
    test_Location___str__()
    test_Location___repr__()
    test_Location_fromString()
    test_Location_fromPointGeom()
    test_Location_fromXY()
    test_Location_latlon()
    test_Location_asGeom()
    test_Location_asXY()
    test_Location_geom()
    test_Location_makePickleable()
    test_Location_load()
    test_LocationSet___init__()
    test_LocationSet___getitem__()
    test_LocationSet___repr__()
    test_LocationSet_getBounds()
    test_LocationSet_asString()
    test_LocationSet_asGeom()
    test_LocationSet_asXY()
    test_LocationSet_asHash()
    test_LocationSet_splitKMeans()
    test_LocationSet_bisect()

