import os
from test.helpers import *

from geokit import Location, LocationSet, geom

xy = (9, 5)


def test_Location___init__():
    l = Location(*xy)
    assert True


def test_Location___hash__():
    l = Location(*xy)
    assert isinstance(hash(l), int)


def test_Location___eq__():
    # TEst against locations
    l1 = Location(*xy)
    l2 = Location(*xy)

    assert l1 == l2

    l3 = Location(xy[0], xy[1] + 0.001)
    assert (l1 == l3) == False

    # test against tuple
    assert l1 == xy

    # test against geometry
    pt = geom.point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)

    assert l1 == pt


def test_Location___ne__():
    l1 = Location(*xy)
    l3 = Location(xy[0], xy[1] + 0.001)
    assert l1 != l3


def test_Location___str__():
    l1 = Location(*xy)
    assert str(l1) == "(9.00000,5.00000)"


def test_Location_fromString():
    l1 = Location(*xy)

    okay = [
        "( 9.00000,5.00000)",
        " (9.00000,5.00000)",
        "(9.00000,5.00000) ",
        " ( 9.00000,5.00000) ",
        " qweqdada( 9.00000,5.00000)adfafdq ",
        " ( 9.00,  5) ",
        " ( 9.00000,  5.00000) ",
    ]

    for p in okay:
        assert l1 == Location.fromString(p)


def test_Location_fromPointGeom():
    l1 = Location(*xy)

    pt = geom.point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)
    l2 = Location.fromPointGeom(pt)
    assert l1 == l2


def test_Location_fromXY():
    l1 = Location(*xy)

    pt = geom.point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)

    l2 = Location.fromXY(pt.GetX(), pt.GetY(), srs=EPSG3035)
    assert l1 == l2


def test_Location_latlon():
    pt = geom.point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)

    l1 = Location.fromPointGeom(pt)
    assert np.isclose(l1.latlon, (xy[1], xy[0])).all()


def test_Location_asGeom():
    l1 = Location(*xy)

    g = l1.asGeom()
    assert g.GetSpatialReference().IsSame(EPSG4326)
    assert np.isclose(g.GetX(), xy[0])
    assert np.isclose(g.GetY(), xy[1])

    pt = geom.point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)
    g = l1.asGeom(srs=EPSG3035)
    assert g.GetSpatialReference().IsSame(EPSG3035)
    assert np.isclose(g.GetX(), pt.GetX())
    assert np.isclose(g.GetY(), pt.GetY())


def test_Location_asXY():
    l1 = Location(*xy)
    assert np.isclose(l1.asXY(srs=EPSG4326), xy).all()

    pt = geom.point(*xy)
    pt.TransformTo(EPSG3035)
    assert np.isclose(l1.asXY(srs=EPSG3035), (pt.GetX(), pt.GetY())).all()


def test_Location_geom():
    l1 = Location(*xy)

    g = l1.geom
    assert g.GetSpatialReference().IsSame(EPSG4326)
    assert np.isclose(g.GetX(), xy[0])
    assert np.isclose(g.GetY(), xy[1])

    l2 = Location.fromXY(*xy, srs=EPSG3035)
    assert not g.GetSpatialReference().IsSame(EPSG3035)


def test_Location_makePickleable():
    l1 = Location(*xy)
    assert l1._geom is None

    g = l1.geom
    assert not l1._geom is None

    l1.makePickleable()
    assert l1._geom is None


def test_Location_load():
    l1 = Location(*xy)

    assert l1 == Location.load(l1)

    # From pt
    pt = geom.point(*xy, srs=4326)
    pt.TransformTo(EPSG3035)
    assert l1 == Location.load(pt)

    # From xy
    assert l1 == Location.load(xy)

    # From str
    assert l1 == Location.load(
        " ( 9.00000,5.00000) ",
    )

    # From xy with srs
    xy_3035 = pt.GetX(), pt.GetY()
    assert l1 == Location.load(xy_3035, srs=3035)
    assert l1 == Location.load(list(xy_3035), srs=3035)
    assert l1 == Location.load(np.array(xy_3035), srs=3035)


def test_LocationSet___init__():
    # From xy list
    ls = LocationSet(pointsInAachen4326)

    # From numpy array
    ls2 = LocationSet(np.array(pointsInAachen4326))
    assert ls[1] == ls2[1]

    # From numpy array with srs change
    ls2 = LocationSet(np.array(pointsInAachen3035), srs=3035)
    assert ls[1] == ls2[1]

    # From single pt
    ls3 = LocationSet(xy)
    assert ls3.count == 1

    # From single geom
    pt = geom.point(*xy, srs=4326)
    ls4 = LocationSet(pt)
    assert ls4.count == 1
    assert ls3[0] == ls4[0]

    # From many geoms
    pts = [geom.point(x, y, srs=4326) for x, y in np.random.random(size=(10, 2))]
    ls5 = LocationSet(pts)
    assert ls5.count == 10


def test_LocationSet___getitem__():
    ls = LocationSet(
        [
            [1, 1],
            [1, 2],
            [2, 2.5],
            [2, 3],
        ]
    )

    assert ls[2] == (2, 2.5)


def test_LocationSet_getBounds():
    ls = LocationSet(pointsInAachen4326)

    bounds = ls.getBounds(3035)

    assert np.isclose(bounds[0], 4039553.1900635841)
    assert np.isclose(bounds[1], 3052769.5385426758)
    assert np.isclose(bounds[2], 4065568.4155270099)
    assert np.isclose(bounds[3], 3087947.74365965)


def test_LocationSet_asString():
    pts = [(2, 3), (4, 2), (5, 5)]
    ls = LocationSet(pts)

    s = ls.asString()
    assert s[1] == "(4.00000,2.00000)"


def test_LocationSet_asGeom():
    pts = [(2, 3), (4, 2), (5, 7)]
    ls = LocationSet(pts)
    geoms = ls.asGeom()

    assert np.isclose(geoms[1].GetX(), 4)
    assert np.isclose(geoms[1].GetY(), 2)
    assert np.isclose(geoms[2].GetX(), 5)
    assert np.isclose(geoms[2].GetY(), 7)


def test_LocationSet_makePickleable():
    pts = [(2, 3), (4, 2), (5, 5)]
    ls = LocationSet(pts)
    geoms = ls.asGeom()

    assert not ls[1]._geom is None

    ls.makePickleable()

    assert ls[1]._geom is None


def test_LocationSet_asXY():
    pts = [(2, 3), (4, 2), (5, 7)]
    ls = LocationSet(pts)
    xyvals = ls.asXY(srs=EPSG4326)

    assert np.isclose(xyvals[1], (4, 2)).all()
    assert np.isclose(xyvals[2], (5, 7)).all()


def test_LocationSet_asHash():
    pts = [(2, 3), (4, 2), (5, 7)]
    ls = LocationSet(pts)
    h = ls.asHash()

    assert isinstance(h[0], int)
    assert isinstance(h[1], int)
    assert isinstance(h[2], int)

    assert h[0] != h[1]
    assert h[0] != h[2]
    assert h[1] != h[2]


def test_LocationSet_splitKMeans():
    os.environ["OMP_NUM_THREADS"] = "1"
    pts = [(-1, -1), (-1, -1.5), (2, 1), (2, 1.5), (2, -1), (2, -1.5), (2, -1.25)]
    locs = LocationSet(pts)

    sublocsGen = locs.splitKMeans(groups=3, random_state=0)

    sublocs = list(sublocsGen)

    assert sublocs[0].count == 2
    assert sublocs[0][0] == (2, 1)
    assert sublocs[0][1] == (2, 1.5)

    assert sublocs[1].count == 3
    assert sublocs[1][0] == (2, -1)
    assert sublocs[1][1] == (2, -1.5)
    assert sublocs[1][2] == (2, -1.25)

    assert sublocs[2].count == 2
    assert sublocs[2][0] == (-1, -1)
    assert sublocs[2][1] == (-1, -1.5)


def test_LocationSet_bisect():
    pts = [(-1, -1), (-1, -1.5), (2, 1), (2, 1.5), (2, -1), (2, -1.5), (2, -1.25)]
    locs = LocationSet(pts)

    # Lon Only
    sublocsGen = locs.bisect(lon=True, lat=False)

    sublocs = list(sublocsGen)

    assert sublocs[0].count == 2
    assert sublocs[0][0] == (-1, -1)
    assert sublocs[0][1] == (-1, -1.5)

    assert sublocs[1].count == 5
    assert sublocs[1][0] == (2, 1)
    assert sublocs[1][1] == (2, 1.5)
    assert sublocs[1][2] == (2, -1)
    assert sublocs[1][3] == (2, -1.5)
    assert sublocs[1][4] == (2, -1.25)

    # Lat Only
    sublocsGen = locs.bisect(lon=False, lat=True)

    sublocs = list(sublocsGen)

    assert sublocs[0].count == 3
    assert sublocs[0][0] == (-1, -1.5)
    assert sublocs[0][1] == (2, -1.5)
    assert sublocs[0][2] == (2, -1.25)

    assert sublocs[1].count == 4
    assert sublocs[1][0] == (-1, -1)
    assert sublocs[1][1] == (2, 1)
    assert sublocs[1][2] == (2, 1.5)
    assert sublocs[1][3] == (2, -1)
