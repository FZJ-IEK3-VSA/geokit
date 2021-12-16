from .helpers import MASK_DATA, np, pointInAachen3035, pointsInAachen4326, EPSG3035, EPSG4326, POLY, GEOM, SUB_GEOMS, SUB_GEOM, result
from geokit import geom
import matplotlib.pyplot as plt
import pytest
import pandas as pd

# box


def test_box():
    # fun func
    b1 = geom.box(0, 0, 5, 10, srs=EPSG3035)
    assert np.isclose(b1.Area(), 50)


def test_tile():
    # fun func
    t1 = geom.tile(xi=4250, yi=2775, zoom=13)
    envelope = t1.GetEnvelope()

    assert np.isclose(envelope[0], 753363.3507786973)
    assert np.isclose(envelope[1], 758255.3205889486)
    assert np.isclose(envelope[2], 6457400.14953169)
    assert np.isclose(envelope[3], 6462292.119341941)

def test_tileAt():
    tile = geom.tileAt(x=6,y=50, zoom=7, srs=EPSG4326)

    a = np.array(tile.Boundary().GetPoints())
    assert np.isclose(a,
            np.array([[ 626172.13571216, 6261721.35712164,       0.        ],
                        [ 939258.20356825, 6261721.35712164,       0.        ],
                        [ 939258.20356825, 6574807.42497772,       0.        ],
                        [ 626172.13571216, 6574807.42497772,       0.        ],
                        [ 626172.13571216, 6261721.35712164,       0.        ]])).all()

    tile = geom.tileAt(x=4101103, y=2978620, zoom=7, srs=EPSG3035)

    a = np.array(tile.Boundary().GetPoints())
    assert np.isclose(a,
            np.array([[ 626172.13571216, 6261721.35712164,       0.        ],
                        [ 939258.20356825, 6261721.35712164,       0.        ],
                        [ 939258.20356825, 6574807.42497772,       0.        ],
                        [ 626172.13571216, 6574807.42497772,       0.        ],
                        [ 626172.13571216, 6261721.35712164,       0.        ]])).all()

def test_subTiles():
    tiles = list(geom.subTiles(GEOM,
                               zoom=5,
                               checkIntersect=False,
                               asGeom=False))
    assert len(tiles) == 4

    assert tiles[0] == (16, 12, 5)
    assert tiles[1] == (16, 13, 5)
    assert tiles[2] == (17, 12, 5)
    assert tiles[3] == (17, 13, 5)

    tiles = list(geom.subTiles(GEOM,
                               zoom=7,
                               checkIntersect=True,
                               asGeom=False))

    assert len(tiles) == 7

    assert tiles[0] == (67, 50, 7)
    assert tiles[1] == (67, 51, 7)
    assert tiles[2] == (67, 52, 7)
    assert tiles[3] == (68, 49, 7)
    assert tiles[4] == (68, 50, 7)
    assert tiles[5] == (68, 51, 7)
    assert tiles[6] == (69, 49, 7)


def test_tileize():
    geoms = list(geom.tileize(GEOM, zoom=7))

    assert np.isclose(geoms[0].Area(), 6185440214.480698)
    assert np.isclose(geoms[1].Area(), 22669806295.02369)
    assert np.isclose(geoms[2].Area(), 4971343426.690063)
    assert np.isclose(geoms[3].Area(), 11085156736.902699)
    assert np.isclose(geoms[4].Area(), 60694504952.24364)
    assert np.isclose(geoms[5].Area(), 8127832949.697159)
    assert np.isclose(geoms[6].Area(), 4469553269.708176)


def test_point():
    x, y = pointInAachen3035

    # test separate input
    p1 = geom.point(x, y, srs=EPSG3035)
    assert np.isclose(p1.GetX(), x)
    assert np.isclose(p1.GetY(), y)
    assert p1.GetSpatialReference().IsSame(EPSG3035)

    # test tuple input
    p2 = geom.point((x, y), srs=EPSG3035)
    assert np.isclose(p2.GetX(), x)
    assert np.isclose(p2.GetY(), y)
    assert p2.GetSpatialReference().IsSame(EPSG3035)


@pytest.mark.skip("No test implemented for: geom.empty")
def test_empty(): assert False


def test_convertWKT():
    g1 = geom.convertWKT(POLY, srs=EPSG4326)
    assert np.isclose(g1.Area(), 7.8149999999999995)
    assert g1.GetSpatialReference().IsSame(EPSG4326)


def test_polygonizeMatrix():
    # test a simple box
    boxmatrix = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 0, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]], dtype=np.int)

    g1 = geom.polygonizeMatrix(boxmatrix, shrink=None)
    assert np.isclose(g1.geom[0].Area(), 8.0)  # polygonizeMatrix: simple area
    # polygonizeMatrix: empty srs
    assert g1.geom[0].GetSpatialReference() is None
    assert g1.value[0] == 1  # polygonizeMatrix: Value retention

    # test shrink
    g1b = geom.polygonizeMatrix(boxmatrix, shrink=0.0001)
    # polygonizeMatrix: shrunk area
    assert np.isclose(g1b.geom[0].Area(), 7.99984000)

    # test a more complex area
    complexmatrix = np.array([[0, 2, 0, 0, 0],
                              [2, 2, 0, 1, 0],
                              [0, 0, 0, 1, 1],
                              [1, 1, 0, 1, 0],
                              [3, 1, 0, 0, 0]], dtype=np.int)

    g2 = geom.polygonizeMatrix(complexmatrix, shrink=None)
    assert np.isclose(g2.shape[0], 4)  # polygonizeMatrix: geometry count
    assert np.isclose(sum([g.Area() for g in g2.geom]),
                      11.0)  # polygonizeMatrix: area"
    assert np.isclose(g2.value[0], 2)  # polygonizeMatrix: Value retention

    # flatten the complex area
    g3 = geom.polygonizeMatrix(complexmatrix, flat=True, shrink=None)
    assert np.isclose(g3.shape[0], 3)  # polygonizeMatrix: geometry count
    # polygonizeMatrix: flattened area
    assert np.isclose(g3.geom[0].Area(), 7.0)

    # set a boundary and srs context
    g4 = geom.polygonizeMatrix(
        complexmatrix, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    # polygonizeMatrix: contexted area
    assert np.isclose(g4.geom[0].Area(), 175.0)
    assert g4.geom[0].GetSpatialReference().IsSame(
        EPSG3035)  # polygonizeMatrix: contexted srs


def test_polygonizeMask():
    # test a simple box
    boxmask = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0]], dtype=np.bool)

    g1 = geom.polygonizeMask(boxmask, shrink=None)
    assert np.isclose(g1.Area(), 8.0)  # polygonizeMask: simple area
    assert g1.GetSpatialReference() is None  # polygonizeMask: empty srs

    # test shrink
    g1b = geom.polygonizeMask(boxmask, shrink=0.0001)
    assert np.isclose(g1b.Area(), 7.99984000)  # polygonizeMask: shrunk area

    # test a more complex area
    complexmask = np.array([[0, 1, 0, 0, 0],
                            [1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 1],
                            [1, 1, 0, 1, 0],
                            [0, 1, 0, 0, 0]], dtype=np.bool)

    g2 = geom.polygonizeMask(complexmask, shrink=None, flat=False)
    assert np.isclose(len(g2), 3)  # polygonizeMask: geometry count
    assert np.isclose(sum([g.Area() for g in g2]),
                      10.0)  # polygonizeMask: area

    # flatten the complex area
    g3 = geom.polygonizeMask(complexmask, flat=True, shrink=None)
    assert np.isclose(g3.Area(), 10.0)  # polygonizeMask: flattened area

    # set a boundary and srs context
    g4 = geom.polygonizeMask(
        complexmask, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None)
    assert np.isclose(g4.Area(), 250.0)  # polygonizeMask: contexted area
    assert g4.GetSpatialReference().IsSame(
        EPSG3035)  # error("polygonizeMask: contexted srs


def test_flatten():

    # Overlapping polygons
    bounds = [(i, i, i + 2, i + 2) for i in range(5)]
    # test basic combination
    geomList = [geom.box(b, srs=EPSG4326) for b in bounds]

    f1 = geom.flatten(geomList)

    assert np.isclose(f1.Area(), 16.0)  # flattened area

    env = f1.GetEnvelope()
    assert np.isclose(env[0], 0)
    assert np.isclose(env[1], 6)
    assert np.isclose(env[2], 0)
    assert np.isclose(env[3], 6)

    assert f1.GetSpatialReference().IsSame(EPSG4326)  # flattened srs


def test_transform():
    # test a single point
    pt = geom.point(7, 48, srs=EPSG4326)
    t1 = geom.transform(pt, toSRS=EPSG3035)
    assert np.isclose(t1.GetX(), 4097075.016)
    assert np.isclose(t1.GetY(), 2769703.15423898)

    # make a collection of polygons using polygonizeMask
    complexmask = np.array([[0, 1, 0, 0, 0],
                            [1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 1],
                            [1, 1, 0, 1, 0],
                            [0, 1, 0, 0, 0]], dtype=np.bool)

    polygons = geom.polygonizeMask(complexmask, bounds=(
        6, 45, 11, 50), flat=False, srs=EPSG4326, shrink=None)

    t2 = geom.transform(polygons, toSRS='europe_m', segment=0.1)
    assert (len(t2) == 3)  # "Transform Count
    assert t2[0].GetSpatialReference().IsSame(EPSG3035)  # "Transform srs
    assert np.isclose(sum([t.Area() for t in t2]),
                      83747886418.48529)  # "Transform Area


def test_extractVerticies():
    # Test polygon
    pts1 = geom.extractVerticies(GEOM)
    assert np.isclose(pts1[5, 1], 35.1)
    assert pts1.shape == (10, 2)

    # Test multipolygon
    pts2 = geom.extractVerticies(geom.flatten(SUB_GEOMS))
    assert pts2.shape == (12, 2)

    # Test linestring
    pts3 = geom.extractVerticies(GEOM.Boundary())
    assert np.isclose(pts3[5, 1], 35.1)
    assert pts3.shape == (10, 2)

    # Test multilinestring
    assert np.isclose(pts3[5, 1], 35.1)
    assert pts3.shape == (10, 2)

    # Test Point
    pts5 = geom.extractVerticies(geom.point(5, 20))
    assert np.isclose(pts5[0, 0], 5)
    assert pts5.shape == (1, 2)


def test_drawGeoms():
    # Draw single polygon
    r = geom.drawGeoms(SUB_GEOM)
    plt.savefig(result("drawGeoms-1.png"), dpi=100)

    # Draw single linestring
    r = geom.drawGeoms(SUB_GEOM.Boundary())
    plt.savefig(result("drawGeoms-2.png"), dpi=100)

    # Draw a multipolygon
    r = geom.drawGeoms(geom.flatten(SUB_GEOMS))
    plt.savefig(result("drawGeoms-3.png"), dpi=100)

    # Draw a list of polygons and set an MPL argument
    r = geom.drawGeoms(SUB_GEOMS, fc='b')
    plt.savefig(result("drawGeoms-4.png"), dpi=100)

    # Change projection systems
    r = geom.drawGeoms(SUB_GEOMS, fc='r', srs=3035)
    plt.savefig(result("drawGeoms-5.png"), dpi=100)

    # Draw from a dataframe
    df = pd.DataFrame(dict(geom=SUB_GEOMS, hats=[1, 2, 3]))

    r = geom.drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-6.png"), dpi=100)

    # Set individual mpl args
    df["MPL:hatch"] = ["//", "+", None]
    r = geom.drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-7.png"), dpi=100)

    # Test colorby
    r = geom.drawGeoms(df, srs=3035, colorBy="hats")
    plt.savefig(result("drawGeoms-8.png"), dpi=100)

    assert True
