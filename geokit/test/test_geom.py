from helpers import *
from geokit.geom import *


## box
def test_makeBox():
    # fun func
    b1 = makeBox(0,0,5,10, srs=EPSG3035)
    
    # check results
    if( b1.Area() != 50 ): error("Box Creation")

def test_makePoint():
    x,y = pointInAachen3035

    # test separate input
    p1 = makePoint( x, y, srs=EPSG3035)
    if not isclose(p1.GetX(),x): error("makePoint - setting x")
    if not isclose(p1.GetY(),y): error("makePoint - setting y")
    if not p1.GetSpatialReference().IsSame(EPSG3035): error("makePoint - setting srs")

    # test tuple input
    p2 = makePoint( (x, y), srs=EPSG3035)
    if not isclose(p2.GetX(),x): error("makePoint - setting x")
    if not isclose(p2.GetY(),y): error("makePoint - setting y")
    if not p2.GetSpatialReference().IsSame(EPSG3035): error("makePoint - setting srs")

def test_makeEmptyGeom():
    print("MAKE 'makeEmptyGeom' TESTER!!!!!!!!!!!!")

def test_convertWKT():
    g1 = convertWKT(POLY, srs=EPSG4326)
    if not isclose(g1.Area(),7.8149999999999995): error("convertWKT - area")
    if not g1.GetSpatialReference().IsSame(EPSG4326): error("convertWKT - setting srs")

def test_vectorize():
    # test a simple box
    boxmatrix = np.array([[0,0,0,0,0],
                          [0,1,1,1,0],
                          [0,1,0,1,0],
                          [0,1,1,1,0],
                          [0,0,0,0,0]], dtype=np.int)

    g1 = vectorize(boxmatrix, shrink=None)
    if not isclose(g1[0].Area(),8.0): error("vectorize: simple area")
    if not g1[0].GetSpatialReference() is None: error("vectorize: empty srs")

    # test shrink
    g1b = vectorize(boxmatrix, shrink=0.0001)
    if not isclose(g1b[0].Area(), 7.98400085984): error("vectorize: shrunk area")

    # test a more complex area
    complexmatrix = np.array([[0,2,0,0,0],
                              [2,2,0,1,0],
                              [0,0,0,1,1],
                              [1,1,0,1,0],
                              [3,1,0,0,0]], dtype=np.int)

    g2 = vectorize(complexmatrix, shrink=None)
    if not len(g2)==4: error("vectorize: geometry count")
    if not isclose(sum([g.Area() for g in g2]),11.0): error("vectorize: area")

    # flatten the complex area
    g3 = vectorize(complexmatrix, flat=True, shrink=None)
    if not len(g3)==3: error("vectorize: geometry count")
    if not isclose(g3[0].Area(),7.0): error("vectorize: flattened area")
    
    # set a boundary and srs context
    g4 = vectorize(complexmatrix, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    if not isclose(g4[0].Area(),175.0): error("vectorize: contexted area")
    if not g4[0].GetSpatialReference().IsSame(EPSG3035): error("vectorize: contexted srs")

def test_convertMask():
    # test a simple box
    boxmask = np.array([[0,0,0,0,0],
                        [0,1,1,1,0],
                        [0,1,0,1,0],
                        [0,1,1,1,0],
                        [0,0,0,0,0]], dtype=np.bool)

    g1 = convertMask(boxmask, shrink=None)
    if not isclose(g1[0].Area(),8.0): error("convertMask: simple area")
    if not g1[0].GetSpatialReference() is None: error("convertMask: empty srs")

    # test shrink
    g1b = convertMask(boxmask, shrink=0.0001)
    if not isclose(g1b[0].Area(), 7.98400085984): error("convertMask: shrunk area")

    # test a more complex area
    complexmask = np.array([[0,1,0,0,0],
                            [1,1,0,1,0],
                            [0,0,0,1,1],
                            [1,1,0,1,0],
                            [0,1,0,0,0]], dtype=np.bool)

    g2 = convertMask(complexmask, shrink=None)
    if not len(g2)==3: error("convertMask: geometry count")
    if not isclose(sum([g.Area() for g in g2]),10.0): error("convertMask: area")

    # flatten the complex area
    g3 = convertMask(complexmask, flat=True, shrink=None)
    if not isclose(g3.Area(),10.0): error("convertMask: flattened area")
    
    # set a boundary and srs context
    g4 = convertMask(complexmask, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    if not isclose(g4.Area(),250.0): error("convertMask: contexted area")
    if not g4.GetSpatialReference().IsSame(EPSG3035): error("convertMask: contexted srs")

def test_flatten():

    # Overlapping polygons
    bounds = [ (i,i,i+2,i+2) for i in range(5) ]
    # test basic combination
    geomList = [makeBox(b, srs=EPSG4326) for b in bounds]

    f1 = flatten(geomList)

    if not isclose(f1.Area(),16.0): error( "flattened area" )
    env = f1.GetEnvelope()
    if not ( isclose(env[0],0) and 
             isclose(env[1],6) and 
             isclose(env[2],0) and 
             isclose(env[3],6)) : error( "flattened extents")

    if not (f1.GetSpatialReference().IsSame(EPSG4326)): error("flattened srs")

def test_transform():
    # test a single point
    pt = makePoint(7,48, srs=EPSG4326)
    t1 = transform(pt, toSRS=EPSG3035)
    if not ( isclose(t1.GetX(),4097075.016) and isclose(t1.GetY(),2769703.15423898)):
        error("Point transform")


    # make a collection of polygons using convertMask
    complexmask = np.array([[0,1,0,0,0],
                            [1,1,0,1,0],
                            [0,0,0,1,1],
                            [1,1,0,1,0],
                            [0,1,0,0,0]], dtype=np.bool)

    polygons = convertMask( complexmask, bounds=(6, 45, 11, 50), srs=EPSG4326, shrink=None)

    t2 = transform(polygons, toSRS='europe_m', segment=0.1)
    if not ( len(t2)==3): error("Transform Count")
    if not t2[0].GetSpatialReference().IsSame(EPSG3035): error("Transform srs")
    if not isclose( sum([t.Area() for t in t2]), 83747886418.48529 ): error("Transform Area")


if __name__ == '__main__':
    test_makeBox()
    test_makePoint()
    test_makeEmptyGeom()
    test_convertWKT()
    test_flatten()
    test_vectorize()
    test_convertMask()
    test_transform()
