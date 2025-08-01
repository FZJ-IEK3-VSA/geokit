from test.helpers import (
    EPSG3035,
    EPSG4326,
    FJI_SHAPE_PATH,
    GEOM,
    MASK_DATA,
    POLY,
    SUB_GEOM,
    SUB_GEOMS,
    np,
    pointInAachen3035,
    pointsInAachen4326,
    result,
)

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from osgeo import ogr

from geokit import geom, vector


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
    tile = geom.tileAt(x=6, y=50, zoom=7, srs=EPSG4326)

    a = np.array(tile.Boundary().GetPoints())
    assert np.isclose(
        a,
        np.array(
            [
                [626172.13571216, 6261721.35712164, 0.0],
                [939258.20356825, 6261721.35712164, 0.0],
                [939258.20356825, 6574807.42497772, 0.0],
                [626172.13571216, 6574807.42497772, 0.0],
                [626172.13571216, 6261721.35712164, 0.0],
            ]
        ),
    ).all()

    tile = geom.tileAt(x=4101103, y=2978620, zoom=7, srs=EPSG3035)

    a = np.array(tile.Boundary().GetPoints())
    assert np.isclose(
        a,
        np.array(
            [
                [626172.13571216, 6261721.35712164, 0.0],
                [939258.20356825, 6261721.35712164, 0.0],
                [939258.20356825, 6574807.42497772, 0.0],
                [626172.13571216, 6574807.42497772, 0.0],
                [626172.13571216, 6261721.35712164, 0.0],
            ]
        ),
    ).all()


def test_subTiles():
    tiles = list(geom.subTiles(GEOM, zoom=5, checkIntersect=False, asGeom=False))
    assert len(tiles) == 4

    assert tiles[0] == (16, 12, 5)
    assert tiles[1] == (16, 13, 5)
    assert tiles[2] == (17, 12, 5)
    assert tiles[3] == (17, 13, 5)

    tiles = list(geom.subTiles(GEOM, zoom=7, checkIntersect=True, asGeom=False))

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


@pytest.mark.parametrize(
    "points_input, srs, output_length, output_bounds",
    [
        (
            # test input as list of tuples
            pointsInAachen4326,
            4326,
            0.52498095,
            (6.02141, 6.371634, 50.51939, 50.846025),
        ),
        (
            # test input as nx2 np.array
            np.array([[tup[0], tup[1]] for tup in pointsInAachen4326]),
            4326,
            0.52498095,
            (6.02141, 6.371634, 50.51939, 50.846025),
        ),
        (
            # test input as list of osgeo.ogr.Geometry point objects
            [geom.point(tup, srs=EPSG4326) for tup in pointsInAachen4326],
            4326,
            0.52498095,
            (6.02141, 6.371634, 50.51939, 50.846025),
        ),
    ],
)
def test_line(points_input, srs, output_length, output_bounds):
    # test input as list of tuples
    l = geom.line(points_input, srs=srs)

    assert l.GetSpatialReference().IsSame(EPSG4326)
    assert np.isclose(l.Length(), output_length)
    assert np.isclose(l.GetEnvelope(), output_bounds).all()


def test_polygon():
    # generate a list of point geoms from x/y
    pointsInAachen4326_geoms = [geom.point(_p) for _p in pointsInAachen4326]

    # create polygon from x/y coordinates with default setting
    poly1 = geom.polygon(pointsInAachen4326, srs="default")
    # test this against polygon created directly from points, as well default srs
    poly2 = geom.polygon(pointsInAachen4326_geoms, srs="default")

    assert poly1.GetSpatialReference().IsSame(EPSG4326)
    assert poly1.Equals(poly2)
    assert poly1.GetEnvelope() == (6.02141, 6.371634, 50.51939, 50.846025)

    # now test with other srs options, e.g. EPSG:3035
    poly3 = geom.polygon(pointsInAachen4326, srs=3035)
    poly4 = geom.polygon(pointsInAachen4326, srs=3035)

    assert poly3.GetSpatialReference().IsSame(EPSG3035)
    assert poly3.Equals(poly4)
    assert poly3.GetEnvelope() == (6.02141, 6.371634, 50.51939, 50.846025)

    # test without SRS
    poly5 = geom.polygon(pointsInAachen4326_geoms, srs=None)

    assert poly5.GetSpatialReference() is None
    assert poly5.GetEnvelope() == (6.02141, 6.371634, 50.51939, 50.846025)


@pytest.mark.skip("No test implemented for: geom.empty")
def test_empty():
    assert False


def test_convertWKT():
    g1 = geom.convertWKT(POLY, srs=EPSG4326)
    assert np.isclose(g1.Area(), 7.8149999999999995)
    assert g1.GetSpatialReference().IsSame(EPSG4326)


def test_polygonizeMatrix():
    # test a simple box
    boxmatrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

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
    complexmatrix = np.array(
        [
            [0, 2, 0, 0, 0],
            [2, 2, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [3, 1, 0, 0, 0],
        ],
        dtype=int,
    )

    g2 = geom.polygonizeMatrix(complexmatrix, shrink=None)
    assert np.isclose(g2.shape[0], 4)  # polygonizeMatrix: geometry count
    assert np.isclose(sum([g.Area() for g in g2.geom]), 11.0)  # polygonizeMatrix: area"
    assert np.isclose(g2.value[0], 2)  # polygonizeMatrix: Value retention

    # flatten the complex area
    g3 = geom.polygonizeMatrix(complexmatrix, flat=True, shrink=None)
    assert np.isclose(g3.shape[0], 3)  # polygonizeMatrix: geometry count
    # polygonizeMatrix: flattened area
    assert np.isclose(g3.geom[0].Area(), 7.0)

    # set a boundary and srs context
    g4 = geom.polygonizeMatrix(
        complexmatrix, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None
    )
    # polygonizeMatrix: contexted area
    assert np.isclose(g4.geom[0].Area(), 175.0)
    assert (
        g4.geom[0].GetSpatialReference().IsSame(EPSG3035)
    )  # polygonizeMatrix: contexted srs


def test_polygonizeMask():
    # test a simple box
    boxmask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    g1 = geom.polygonizeMask(boxmask, shrink=None)
    assert np.isclose(g1.Area(), 8.0)  # polygonizeMask: simple area
    assert g1.GetSpatialReference() is None  # polygonizeMask: empty srs

    # test shrink
    g1b = geom.polygonizeMask(boxmask, shrink=0.0001)
    assert np.isclose(g1b.Area(), 7.99984000)  # polygonizeMask: shrunk area

    # test a more complex area
    complexmask = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=bool,
    )

    g2 = geom.polygonizeMask(complexmask, shrink=None, flat=False)
    assert np.isclose(len(g2), 3)  # polygonizeMask: geometry count
    assert np.isclose(sum([g.Area() for g in g2]), 10.0)  # polygonizeMask: area

    # flatten the complex area
    g3 = geom.polygonizeMask(complexmask, flat=True, shrink=None)
    assert np.isclose(g3.Area(), 10.0)  # polygonizeMask: flattened area

    # set a boundary and srs context
    g4 = geom.polygonizeMask(
        complexmask, bounds=(-3, 10, 22, 35), srs=EPSG3035, flat=True, shrink=None
    )
    assert np.isclose(g4.Area(), 250.0)  # polygonizeMask: contexted area
    assert g4.GetSpatialReference().IsSame(
        EPSG3035
    )  # error("polygonizeMask: contexted srs


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
    complexmask = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=bool,
    )

    polygons = geom.polygonizeMask(
        complexmask, bounds=(6, 45, 11, 50), flat=False, srs=EPSG4326, shrink=None
    )

    t2 = geom.transform(polygons, toSRS="europe_m", segment=0.1)
    assert len(t2) == 3  # "Transform Count
    assert t2[0].GetSpatialReference().IsSame(EPSG3035)  # "Transform srs
    assert np.isclose(sum([t.Area() for t in t2]), 83747886418.48529)  # "Transform Area


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
    assert SUB_GEOM.GetSpatialReference().IsSame(EPSG4326)

    # Draw single linestring
    r = geom.drawGeoms(SUB_GEOM.Boundary())
    plt.savefig(result("drawGeoms-2.png"), dpi=100)

    # Draw a multipolygon
    r = geom.drawGeoms(geom.flatten(SUB_GEOMS))
    plt.savefig(result("drawGeoms-3.png"), dpi=100)

    # Draw a list of polygons and set an MPL argument
    r = geom.drawGeoms(SUB_GEOMS, fc="b")
    plt.savefig(result("drawGeoms-4.png"), dpi=100)

    # Change projection systems
    r = geom.drawGeoms(SUB_GEOMS, fc="r", srs=3035)
    plt.savefig(result("drawGeoms-5.png"), dpi=100)
    assert SUB_GEOMS[0].GetSpatialReference().IsSame(EPSG4326)

    # Draw from a dataframe, once without and once with SRS adaptation
    df = pd.DataFrame(dict(geom=SUB_GEOMS, hats=[1, 2, 3]))

    r = geom.drawGeoms(df)
    plt.savefig(result("drawGeoms-6.png"), dpi=100)
    r = geom.drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-6b.png"), dpi=100)
    assert df.geom[0].GetSpatialReference().IsSame(EPSG4326)

    # Set individual mpl args
    df["MPL:hatch"] = ["//", "+", None]
    r = geom.drawGeoms(df, srs=3035)
    plt.savefig(result("drawGeoms-7.png"), dpi=100)

    # Test colorby
    r = geom.drawGeoms(df, srs=3035, colorBy="hats")
    plt.savefig(result("drawGeoms-8.png"), dpi=100)

    assert True


def test_shift():
    # test point, no srs
    assert geom.shift(geom=geom.point((0, 1)), lonShift=5).Equals(geom.point((5, 1)))

    # test line, epsg 3035
    l1 = geom.line([(0, 0), (1, 1)], srs=3035)
    l1_check = geom.line([(0, -10), (1, -9)], srs=3035)
    assert geom.shift(l1, latShift=-10).Equals(l1_check)

    # test polygon, srs 4326
    b1 = geom.box(-170, 60, -160, 70, srs=4326)
    b1_check = geom.box(10, -30, 20, -20, srs=4326)
    assert geom.shift(geom=b1, lonShift=180, latShift=-90).Equals(b1_check)

    # test multipolygon
    b2 = geom.box(-120, 10, -100, 30, srs=4326)
    b2_check = geom.box(60, -80, 80, -60, srs=4326)
    b_multi = b1.Union(b2)
    b_multi_check = b1_check.Union(b2_check)
    assert geom.shift(geom=b_multi, lonShift=180, latShift=-90).Equals(b_multi_check)


def test_divideMultipolygonIntoEasternAndWesternPart():
    # load FJI.shp from test data
    FJI_geom = vector.extractFeatures(FJI_SHAPE_PATH).geom.iloc[0]
    # divide the geometry at the antimeridian, keep main
    main_geom = geom.divideMultipolygonIntoEasternAndWesternPart(FJI_geom, side="main")

    assert main_geom.GetGeometryName() == "MULTIPOLYGON"
    assert np.isclose(main_geom.GetGeometryCount(), 274)
    assert np.isclose(
        main_geom.GetEnvelope(),
        (
            176.89971924,
            180.0,
            -19.19361115,
            -12.46172428,
        ),
        atol=0,
    ).all()
    assert np.isclose(main_geom.Area(), 1.5381107036696313, atol=0)

    # make sure extraction of both geoms works, too
    both_geoms = geom.divideMultipolygonIntoEasternAndWesternPart(FJI_geom, side="both")

    assert len(both_geoms) == 2
    assert np.isclose(
        [g.Area() for g in both_geoms],
        [1.5381107036696313, 0.07496812355738873],
        atol=0,
    ).all()

    # make sure extraction of right geom is always the same
    right_geom = geom.divideMultipolygonIntoEasternAndWesternPart(
        FJI_geom, side="right"
    )

    assert np.isclose(
        right_geom.GetEnvelope(),
        (
            -180.0,
            -178.22860718,
            -21.04249954,
            -15.70972347,
        ),
        atol=0,
    ).all()
    assert np.isclose(right_geom.Area(), 0.07496812355738873, atol=0)


def test_fixOutOfBoundsGeoms():
    # generate circles near and crossing the antimeridian, merge to multipolygon
    testcircle = geom.point(0, 0).Buffer(1)
    testcircleeast = geom.shift(
        testcircle, lonShift=-179.5
    )  # extends over entimeridian in the East
    testcirclewest = geom.shift(
        testcircle, lonShift=+180.8
    )  # extends over entimeridian in the West
    multi = testcirclewest.Union(testcircleeast)

    # clip off the parts extending over antimeridian
    multi_clipped = geom.fixOutOfBoundsGeoms(multi, how="clip")
    assert isinstance(multi_clipped, ogr.Geometry)
    assert np.isclose(
        multi_clipped.GetEnvelope(),
        (
            -180.0,
            180.0,
            -1.0,
            +1.0,
        ),
        atol=0,
    ).all()
    # do again just for the western testcircle
    testcirclewest_clipped = geom.fixOutOfBoundsGeoms(testcirclewest, how="clip")
    assert isinstance(testcirclewest_clipped, ogr.Geometry)
    assert np.isclose(
        testcirclewest_clipped.GetEnvelope(),
        (
            179.8,
            180.0,
            -0.5995364281486639,
            +0.5995364281486636,
        ),
        atol=0,
    ).all()

    # split multi along the antimeridian
    testcirclewest_shifted = geom.fixOutOfBoundsGeoms(testcirclewest, how="shift")

    assert isinstance(testcirclewest_shifted, ogr.Geometry)
    assert np.isclose(
        testcirclewest_shifted.GetEnvelope(),
        (
            -180.0,
            +180.0,
            -1.0,
            +1.0,
        ),
        atol=0,
    ).all()


def test_applyBuffer():
    # generate point at lat = 0 to apply a test buffer
    testpoint_equator = geom.point(-179.9, 0, srs=4326)
    # first test latlon buffer
    buf_none = geom.applyBuffer(
        geom=testpoint_equator, buffer=1.0, applyBufferInSRS=False, split="none"
    )
    assert np.isclose(buf_none.GetEnvelope(), (-180.9, -178.9, -1.0, 1.0), atol=0).all()
    assert np.isclose(buf_none.Area(), np.pi, rtol=0.001)
    buf_shift = geom.applyBuffer(
        geom=testpoint_equator, buffer=1.0, applyBufferInSRS=False, split="shift"
    )
    assert np.isclose(
        buf_shift.GetEnvelope(), (-180.0, +180.0, -1.0, 1.0), atol=0
    ).all()
    assert np.isclose(buf_shift.Area(), buf_none.Area(), rtol=0.001)
    buf_clip = geom.applyBuffer(
        geom=testpoint_equator, buffer=1.0, applyBufferInSRS=False, split="clip"
    )
    assert np.isclose(buf_clip.GetEnvelope(), (-180.0, -178.9, -1.0, 1.0), atol=0).all()
    # then do metric buffer with 50 kms
    buf_clip_6933 = geom.applyBuffer(
        geom=testpoint_equator, buffer=50000, applyBufferInSRS=6933, split="clip"
    )
    assert np.isclose(
        buf_clip_6933.GetEnvelope(),
        (
            -180.0,
            -179.38179160943938,
            -0.391934549810819,
            0.391934549810819,
        ),
        atol=0,
    ).all()

    # now try again near 90° lat
    testpoint_north = geom.point(0, 89.9, srs=4326)
    # first latlon buffer
    buf_north_clip = geom.applyBuffer(
        geom=testpoint_north, buffer=1, applyBufferInSRS=False, split="clip"
    )
    assert np.isclose(buf_north_clip.GetEnvelope(), (-1, +1, 88.9, 90), atol=0).all()
    # try again with metric system and 50kms buffer
    buf_north_clip_6933 = geom.applyBuffer(
        geom=testpoint_north, buffer=50000, applyBufferInSRS=6933, split="clip"
    )
    assert np.isclose(buf_north_clip_6933.GetEnvelope()[0], -0.5182083905606406)
    assert np.isclose(buf_north_clip_6933.GetEnvelope()[1], 0.5182083905606406)
    assert np.isclose(buf_north_clip_6933.GetEnvelope()[2], 83.33841323028614)
    assert np.isclose(buf_north_clip_6933.GetEnvelope()[3], 89.99999879797518)
    # assert buf_north_clip_6933.GetEnvelope() == (
    #     -0.5182083905606406,
    #     0.5182083905606406,
    #     83.33841323028614,
    #     89.99999879797518,
    # )
    assert np.isclose(
        geom.transform(buf_north_clip_6933, toSRS=6933).Area(),
        3926325058.480929,
        atol=0,
    )


if __name__ == "__main__":
    test_applyBuffer()
