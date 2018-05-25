from helpers import *
from geokit.geom import *


## box
def test_box():
    # fun func
    b1 = box(0,0,5,10, srs=EPSG3035)
    
    # check results
    if( b1.Area() != 50 ): error("Box Creation")

    print("box passed all tests")

def test_point():
    x,y = pointInAachen3035

    # test separate input
    p1 = point( x, y, srs=EPSG3035)
    if not isclose(p1.GetX(),x): error("point - setting x")
    if not isclose(p1.GetY(),y): error("point - setting y")
    if not p1.GetSpatialReference().IsSame(EPSG3035): error("point - setting srs")

    # test tuple input
    p2 = point( (x, y), srs=EPSG3035)
    if not isclose(p2.GetX(),x): error("point - setting x")
    if not isclose(p2.GetY(),y): error("point - setting y")
    if not p2.GetSpatialReference().IsSame(EPSG3035): error("point - setting srs")

    print("point passed all tests")

def test_empty():
    print("empty is not tested...")

def test_convertWKT():
    g1 = convertWKT(POLY, srs=EPSG4326)
    if not isclose(g1.Area(),7.8149999999999995): error("convertWKT - area")
    if not g1.GetSpatialReference().IsSame(EPSG4326): error("convertWKT - setting srs")

    print("convertWKT passed all tests")

def test_polygonizeMatrix():
    # test a simple box
    boxmatrix = np.array([[0,0,0,0,0],
                          [0,1,1,1,0],
                          [0,1,0,1,0],
                          [0,1,1,1,0],
                          [0,0,0,0,0]], dtype=np.int)

    g1 = polygonizeMatrix(boxmatrix, shrink=None)
    if not isclose(g1.geom[0].Area(),8.0): error("polygonizeMatrix: simple area")
    if not g1.geom[0].GetSpatialReference() is None: error("polygonizeMatrix: empty srs")
    if not g1.value[0] == 1: error("polygonizeMatrix: Value retention")

    # test shrink
    g1b = polygonizeMatrix(boxmatrix, shrink=0.0001)
    compare(g1b.geom[0].Area(), 7.99984000, "polygonizeMatrix: shrunk area")

    # test a more complex area
    complexmatrix = np.array([[0,2,0,0,0],
                              [2,2,0,1,0],
                              [0,0,0,1,1],
                              [1,1,0,1,0],
                              [3,1,0,0,0]], dtype=np.int)

    g2 = polygonizeMatrix(complexmatrix, shrink=None)
    compare(g2.shape[0], 4, "polygonizeMatrix: geometry count")
    compare(sum([g.Area() for g in g2.geom]), 11.0, "polygonizeMatrix: area")
    compare(g2.value[0], 2, "polygonizeMatrix: Value retention")

    # flatten the complex area
    g3 = polygonizeMatrix(complexmatrix, flat=True, shrink=None)
    compare(g3.shape[0], 3, "polygonizeMatrix: geometry count")
    compare(g3.geom[0].Area(),7.0, "polygonizeMatrix: flattened area")
    
    # set a boundary and srs context
    g4 = polygonizeMatrix(complexmatrix, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    compare(g4.geom[0].Area(), 175.0, "polygonizeMatrix: contexted area")
    if not g4.geom[0].GetSpatialReference().IsSame(EPSG3035): error("polygonizeMatrix: contexted srs")

    print("polygonizeMatrix passed all tests")

def test_polygonizeMask():
    # test a simple box
    boxmask = np.array([[0,0,0,0,0],
                        [0,1,1,1,0],
                        [0,1,0,1,0],
                        [0,1,1,1,0],
                        [0,0,0,0,0]], dtype=np.bool)

    g1 = polygonizeMask(boxmask, shrink=None)
    compare(g1.Area(),8.0, "polygonizeMask: simple area")
    if not g1.GetSpatialReference() is None: error("polygonizeMask: empty srs")

    # test shrink
    g1b = polygonizeMask(boxmask, shrink=0.0001)
    compare(g1b.Area(), 7.99984000, "polygonizeMask: shrunk area")

    # test a more complex area
    complexmask = np.array([[0,1,0,0,0],
                            [1,1,0,1,0],
                            [0,0,0,1,1],
                            [1,1,0,1,0],
                            [0,1,0,0,0]], dtype=np.bool)

    g2 = polygonizeMask(complexmask, shrink=None, flat=False)
    compare(len(g2), 3, "polygonizeMask: geometry count")
    compare(sum([g.Area() for g in g2]),10.0, "polygonizeMask: area")

    # flatten the complex area
    g3 = polygonizeMask(complexmask, flat=True, shrink=None)
    compare(g3.Area(), 10.0, "polygonizeMask: flattened area")
    
    # set a boundary and srs context
    g4 = polygonizeMask(complexmask, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    compare(g4.Area(), 250.0, "polygonizeMask: contexted area")
    if not g4.GetSpatialReference().IsSame(EPSG3035): error("polygonizeMask: contexted srs")

    print( "polygonizeMask passed")

def test_flatten():

    # Overlapping polygons
    bounds = [ (i,i,i+2,i+2) for i in range(5) ]
    # test basic combination
    geomList = [box(b, srs=EPSG4326) for b in bounds]

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
    pt = point(7,48, srs=EPSG4326)
    t1 = transform(pt, toSRS=EPSG3035)
    if not ( isclose(t1.GetX(),4097075.016) and isclose(t1.GetY(),2769703.15423898)):
        error("Point transform")


    # make a collection of polygons using polygonizeMask
    complexmask = np.array([[0,1,0,0,0],
                            [1,1,0,1,0],
                            [0,0,0,1,1],
                            [1,1,0,1,0],
                            [0,1,0,0,0]], dtype=np.bool)

    polygons = polygonizeMask( complexmask, bounds=(6, 45, 11, 50), flat=False, srs=EPSG4326, shrink=None)

    t2 = transform(polygons, toSRS='europe_m', segment=0.1)
    if not ( len(t2)==3): error("Transform Count")
    if not t2[0].GetSpatialReference().IsSame(EPSG3035): error("Transform srs")
    if not isclose( sum([t.Area() for t in t2]), 83747886418.48529 ): error("Transform Area")

def test_extractVerticies():
    # Test polygon
    pts1 = extractVerticies(GEOM)
    compare(pts1[5,1], 35.1, "extractVecticies")
    if not pts1.shape==(10,2): error( "extractVecticies")

    # Test multipolygon
    pts2 = extractVerticies(flatten(SUB_GEOMS))
    if not pts2.shape==(12,2): error( "extractVecticies")

    # Test linestring
    pts3 = extractVerticies(GEOM.Boundary())
    compare(pts3[5,1], 35.1, "extractVecticies")
    if not pts3.shape==(10,2): error( "extractVecticies")

    # Test multilinestring
    compare(pts3[5,1], 35.1, "extractVecticies")
    if not pts3.shape==(10,2): error( "extractVecticies")

    # Test Point
    pts5 = extractVerticies(point(5,20))
    compare(pts5[0,0], 5, "extractVecticies - value")
    if not pts5.shape==(1,2): error( "extractVecticies - shape")    
    
    print( "extractVerticies passed" )

def test_drawGeoms():
    #Draw single polygon
    r = drawGeoms(SUB_GEOM)
    plt.savefig(result("drawGeoms-1.png"), dpi=100)

    #Draw single linestring
    r = drawGeoms(SUB_GEOM.Boundary())
    plt.savefig(result("drawGeoms-2.png"), dpi=100)

    #Draw a multipolygon
    r = drawGeoms(flatten(SUB_GEOMS))
    plt.savefig(result("drawGeoms-3.png"), dpi=100)

    #Draw a list of polygons and set an MPL argument
    r = drawGeoms(SUB_GEOMS, fc='b')
    plt.savefig(result("drawGeoms-4.png"), dpi=100)

    # Change projection systems
    r = drawGeoms(SUB_GEOMS, fc='r', srs=3035)
    plt.savefig(result("drawGeoms-5.png"), dpi=100)

    # Draw from a dataframe
    df = pd.DataFrame(dict(geom=SUB_GEOMS, hats=[1,2,3]))
    
    r = drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-6.png"), dpi=100)

    # Set individual mpl args
    df["MPL:hatch"] = ["//","+",None]
    r = drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-7.png"), dpi=100)

    # Test colorby
    r = drawGeoms(df, srs=3035, colorBy="hats")
    plt.savefig(result("drawGeoms-8.png"), dpi=100)

    print("No errors in drawGeoms, but they should be checked manually!")

if __name__ == '__main__':
    # test_box()
    # test_point()
    # test_empty()
    # test_convertWKT()
    # test_flatten()
    # test_polygonizeMatrix()
    # test_polygonizeMask()
    # test_transform()
    # test_extractVerticies()
    test_drawGeoms()
 