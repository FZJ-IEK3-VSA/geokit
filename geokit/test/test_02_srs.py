import pytest

from geokit import srs, vector
from geokit.test.helpers import (
    AACHEN_SHAPE_PATH,
    MASK_DATA,
    np,
    osr,
    pointInAachen3035,
    pointInAccraEcowasM,
    pointInWindhoekSadcM,
    pointsInAachen4326,
)


@pytest.mark.parametrize(
    "points, fromSRS, toSRS, expected",
    [
        (
            pointsInAachen4326,
            "latlon",
            "europe_laea",
            [
                (4042131.1581, 3052769.5385, 0.0),
                (4039553.1900, 3063551.9478, 0.0),
                (4065568.415, 3087947.743, 0.0),
            ],
        ),
        (pointInAccraEcowasM, "ecowas_laea", "latlon", (5.562, -0.1389, 0.0)),
        (pointInWindhoekSadcM, "sadc_laea", "latlon", (-22.389, 17.398, 0.0)),
    ],
)
def test_xyTransform(points, fromSRS, toSRS, expected):
    if isinstance(points, tuple):
        # test single point
        p = srs.xyTransform(points, fromSRS=fromSRS, toSRS=toSRS, outputFormat="raw")[0]
        assert np.isclose(p[0], expected[0], 1e-3)
        assert np.isclose(p[1], expected[1], 1e-3)

    else:
        # test multiple points
        p = srs.xyTransform(points, fromSRS=fromSRS, toSRS=toSRS, outputFormat="raw")

        assert np.isclose(p[0][0], expected[0][0], 1e-6)
        assert np.isclose(p[0][1], expected[0][1], 1e-6)
        assert np.isclose(p[1][0], expected[1][0], 1e-6)
        assert np.isclose(p[1][1], expected[1][1], 1e-6)
        assert np.isclose(p[2][0], expected[2][0], 1e-6)
        assert np.isclose(p[2][1], expected[2][1], 1e-6)


def test_loadSRS():
    # Test creating an osr SRS object
    s1 = srs.loadSRS(srs.EPSG4326)
    # Test an EPSG identifier
    s2 = srs.loadSRS(4326)
    # Test an EPSG code
    s3 = srs.loadSRS("epsg:4326")
    # Are they the same?
    assert s1.IsSame(s2)
    assert s1.IsSame(s3)

    # test an invalid srs, must raise error
    with pytest.raises(AssertionError):
        s3 = srs.loadSRS(1000)


def test_centeredLAEA():
    # first test procedure using lat and lon coordinates
    s1 = srs.centeredLAEA(6.8, 50.0775)
    assert isinstance(s1, osr.SpatialReference)

    # then test procedure using a geom
    Aachen_geom = vector.extractFeatures(AACHEN_SHAPE_PATH).geom[0]
    s2 = srs.centeredLAEA(geom=Aachen_geom)
    assert isinstance(s2, osr.SpatialReference)


def test_tileIndexAt():
    tile = srs.tileIndexAt(6.083, 50.775, zoom=8, srs=srs.EPSG4326)
    assert tile.xi == 132
    assert tile.yi == 85
    assert tile.zoom == 8

    tile = srs.tileIndexAt(x=4101103, y=2978620, zoom=12, srs=srs.EPSG3035)
    assert tile.xi == 2126
    assert tile.yi == 1391
    assert tile.zoom == 12

    tiles = srs.tileIndexAt(
        x=[4101103, 4101000, 4103000],
        y=[2978620, 2978620, 2978620],
        zoom=12,
        srs=srs.EPSG3035,
    )
    assert np.isclose(tiles.xi, [2126, 2126, 2127]).all()
    assert np.isclose(tiles.yi, [1391, 1391, 1391]).all()
    assert np.isclose(tiles.zoom, 12).all()
