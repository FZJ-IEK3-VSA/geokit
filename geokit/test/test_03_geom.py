from .helpers import MASK_DATA, np, pointInAachen3035, pointsInAachen4326, EPSG3035, EPSG4326, POLY, GEOM, SUB_GEOMS, SUB_GEOM, result
from geokit import geom
import matplotlib.pyplot as plt
import pytest 
import pandas as pd

## box
def test_box():
    # fun func
    b1 = geom.box(0,0,5,10, srs=EPSG3035)
    assert np.isclose(b1.Area(), 50 )

def test_point():
    x,y = pointInAachen3035

    # test separate input
    p1 = geom.point( x, y, srs=EPSG3035)
    assert np.isclose(p1.GetX(),x)
    assert np.isclose(p1.GetY(),y)
    assert p1.GetSpatialReference().IsSame(EPSG3035)

    # test tuple input
    p2 = geom.point( (x, y), srs=EPSG3035)
    assert np.isclose(p2.GetX(),x)
    assert np.isclose(p2.GetY(),y)
    assert p2.GetSpatialReference().IsSame(EPSG3035)

@pytest.mark.skip("No test implemented for: geom.empty")
def test_empty(): assert False

def test_convertWKT():
    g1 = geom.convertWKT(POLY, srs=EPSG4326)
    assert np.isclose(g1.Area(),7.8149999999999995)
    assert g1.GetSpatialReference().IsSame(EPSG4326)

def test_polygonizeMatrix():
    # test a simple box
    boxmatrix = np.array([[0,0,0,0,0],
                          [0,1,1,1,0],
                          [0,1,0,1,0],
                          [0,1,1,1,0],
                          [0,0,0,0,0]], dtype=np.int)

    g1 = geom.polygonizeMatrix(boxmatrix, shrink=None)
    assert np.isclose(g1.geom[0].Area(),8.0) # polygonizeMatrix: simple area
    assert g1.geom[0].GetSpatialReference() is None # polygonizeMatrix: empty srs
    assert g1.value[0] == 1 # polygonizeMatrix: Value retention

    # test shrink
    g1b = geom.polygonizeMatrix(boxmatrix, shrink=0.0001)
    assert np.isclose(g1b.geom[0].Area(), 7.99984000) # polygonizeMatrix: shrunk area

    # test a more complex area
    complexmatrix = np.array([[0,2,0,0,0],
                              [2,2,0,1,0],
                              [0,0,0,1,1],
                              [1,1,0,1,0],
                              [3,1,0,0,0]], dtype=np.int)

    g2 = geom.polygonizeMatrix(complexmatrix, shrink=None)
    assert np.isclose(g2.shape[0], 4) # polygonizeMatrix: geometry count
    assert np.isclose(sum([g.Area() for g in g2.geom]), 11.0) # polygonizeMatrix: area"
    assert np.isclose(g2.value[0], 2) # polygonizeMatrix: Value retention

    # flatten the complex area
    g3 = geom.polygonizeMatrix(complexmatrix, flat=True, shrink=None)
    assert np.isclose(g3.shape[0], 3) # polygonizeMatrix: geometry count
    assert np.isclose(g3.geom[0].Area(),7.0) # polygonizeMatrix: flattened area
    
    # set a boundary and srs context
    g4 = geom.polygonizeMatrix(complexmatrix, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    assert np.isclose(g4.geom[0].Area(), 175.0) # polygonizeMatrix: contexted area
    assert g4.geom[0].GetSpatialReference().IsSame(EPSG3035) # polygonizeMatrix: contexted srs


def test_polygonizeMask():
    # test a simple box
    boxmask = np.array([[0,0,0,0,0],
                        [0,1,1,1,0],
                        [0,1,0,1,0],
                        [0,1,1,1,0],
                        [0,0,0,0,0]], dtype=np.bool)

    g1 = geom.polygonizeMask(boxmask, shrink=None)
    assert np.isclose(g1.Area(), 8.0) # polygonizeMask: simple area
    assert g1.GetSpatialReference() is None # polygonizeMask: empty srs

    # test shrink
    g1b = geom.polygonizeMask(boxmask, shrink=0.0001)
    assert np.isclose(g1b.Area(), 7.99984000) # polygonizeMask: shrunk area

    # test a more complex area
    complexmask = np.array([[0,1,0,0,0],
                            [1,1,0,1,0],
                            [0,0,0,1,1],
                            [1,1,0,1,0],
                            [0,1,0,0,0]], dtype=np.bool)

    g2 = geom.polygonizeMask(complexmask, shrink=None, flat=False)
    assert np.isclose(len(g2), 3) # polygonizeMask: geometry count
    assert np.isclose(sum([g.Area() for g in g2]),10.0) # polygonizeMask: area

    # flatten the complex area
    g3 = geom.polygonizeMask(complexmask, flat=True, shrink=None)
    assert np.isclose(g3.Area(), 10.0) # polygonizeMask: flattened area
    
    # set a boundary and srs context
    g4 = geom.polygonizeMask(complexmask, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    assert np.isclose(g4.Area(), 250.0) # polygonizeMask: contexted area
    assert g4.GetSpatialReference().IsSame(EPSG3035) # error("polygonizeMask: contexted srs

def test_flatten():

    # Overlapping polygons
    bounds = [ (i,i,i+2,i+2) for i in range(5) ]
    # test basic combination
    geomList = [geom.box(b, srs=EPSG4326) for b in bounds]

    f1 = geom.flatten(geomList)

    assert np.isclose(f1.Area(),16.0) # flattened area

    env = f1.GetEnvelope()
    assert np.isclose(env[0],0)
    assert np.isclose(env[1],6)
    assert np.isclose(env[2],0)
    assert np.isclose(env[3],6)

    assert f1.GetSpatialReference().IsSame(EPSG4326) # flattened srs

def test_transform():
    # test a single point
    pt = geom.point(7,48, srs=EPSG4326)
    t1 = geom.transform(pt, toSRS=EPSG3035)
    assert np.isclose(t1.GetX(), 4097075.016)
    assert np.isclose(t1.GetY(), 2769703.15423898)

    # make a collection of polygons using polygonizeMask
    complexmask = np.array([[0,1,0,0,0],
                            [1,1,0,1,0],
                            [0,0,0,1,1],
                            [1,1,0,1,0],
                            [0,1,0,0,0]], dtype=np.bool)

    polygons = geom.polygonizeMask( complexmask, bounds=(6, 45, 11, 50), flat=False, srs=EPSG4326, shrink=None)

    t2 = geom.transform(polygons, toSRS='europe_m', segment=0.1)
    assert ( len(t2)==3) # "Transform Count
    assert t2[0].GetSpatialReference().IsSame(EPSG3035) # "Transform srs
    assert np.isclose( sum([t.Area() for t in t2]), 83747886418.48529 ) # "Transform Area

def test_extractVerticies():
    # Test polygon
    pts1 = geom.extractVerticies(GEOM)
    assert np.isclose(pts1[5,1], 35.1) 
    assert pts1.shape==(10,2) 

    # Test multipolygon
    pts2 = geom.extractVerticies(geom.flatten(SUB_GEOMS))
    assert pts2.shape==(12,2)

    # Test linestring
    pts3 = geom.extractVerticies(GEOM.Boundary())
    assert np.isclose(pts3[5,1], 35.1) 
    assert pts3.shape==(10,2) 

    # Test multilinestring
    assert np.isclose(pts3[5,1], 35.1) 
    assert pts3.shape==(10,2) 

    # Test Point
    pts5 = geom.extractVerticies(geom.point(5,20))
    assert np.isclose(pts5[0,0], 5) 
    assert pts5.shape==(1,2) 
    
def test_drawGeoms():
    #Draw single polygon
    r = geom.drawGeoms(SUB_GEOM)
    plt.savefig(result("drawGeoms-1.png"), dpi=100)

    #Draw single linestring
    r = geom.drawGeoms(SUB_GEOM.Boundary())
    plt.savefig(result("drawGeoms-2.png"), dpi=100)

    #Draw a multipolygon
    r = geom.drawGeoms(geom.flatten(SUB_GEOMS))
    plt.savefig(result("drawGeoms-3.png"), dpi=100)

    #Draw a list of polygons and set an MPL argument
    r = geom.drawGeoms(SUB_GEOMS, fc='b')
    plt.savefig(result("drawGeoms-4.png"), dpi=100)

    # Change projection systems
    r = geom.drawGeoms(SUB_GEOMS, fc='r', srs=3035)
    plt.savefig(result("drawGeoms-5.png"), dpi=100)

    # Draw from a dataframe
    df = pd.DataFrame(dict(geom=SUB_GEOMS, hats=[1,2,3]))
    
    r = geom.drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-6.png"), dpi=100)

    # Set individual mpl args
    df["MPL:hatch"] = ["//","+",None]
    r = geom.drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-7.png"), dpi=100)

    # Test colorby
    r = geom.drawGeoms(df, srs=3035, colorBy="hats")
    plt.savefig(result("drawGeoms-8.png"), dpi=100)

    assert True
