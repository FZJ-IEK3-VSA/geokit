from .helpers import MASK_DATA, np, pointInAachen3035, pointsInAachen4326, osr
from geokit import srs


def test_xyTransform():

    # test single point
    p1 = srs.xyTransform(
        pointInAachen3035, fromSRS='europe_m', toSRS='latlon')[0]
    real = (6.313298792067333, 50.905105969570265)
    assert np.isclose(p1[0], real[0], 1e-6)
    assert np.isclose(p1[1], real[1], 1e-6)

    # test multiple points
    p2 = srs.xyTransform(pointsInAachen4326,
                         fromSRS='latlon', toSRS='europe_m')
    real = [(4042131.1581, 3052769.5385), (4039553.1900,
                                           3063551.9478), (4065568.415, 3087947.743)]

    assert np.isclose(p2[0][0], real[0][0], 1e-6)
    assert np.isclose(p2[0][1], real[0][1], 1e-6)
    assert np.isclose(p2[1][0], real[1][0], 1e-6)
    assert np.isclose(p2[1][1], real[1][1], 1e-6)
    assert np.isclose(p2[2][0], real[2][0], 1e-6)
    assert np.isclose(p2[2][1], real[2][1], 1e-6)


def test_loadSRS():
    # Test creating an osr SRS object
    s1 = srs.loadSRS(srs.EPSG4326)

    # Test an EPSG identifier
    s2 = srs.loadSRS(4326)

    # Are they the same?
    assert s1.IsSame(s2)


def test_centeredLAEA():
    s1 = srs.centeredLAEA(6.8, 50.0775)
    assert isinstance(s1, osr.SpatialReference)

def test_tileIndexAt():
    tile = srs.tileIndexAt(6.083, 50.775, zoom=8, srs=srs.EPSG4326)
    assert tile.xi == 132
    assert tile.yi == 85
    assert tile.zoom == 8

    tile = srs.tileIndexAt(x=4101103, y=2978620, zoom=12, srs=srs.EPSG3035)
    assert tile.xi == 2126
    assert tile.yi == 1391
    assert tile.zoom == 12

    tiles = srs.tileIndexAt(x=[4101103, 4101000, 4103000], 
                            y=[2978620, 2978620, 2978620], 
                            zoom=12, 
                            srs=srs.EPSG3035)
    assert np.isclose(tiles.xi, [2126, 2126, 2127]).all()
    assert np.isclose(tiles.yi, [1391, 1391, 1391]).all()
    assert np.isclose(tiles.zoom, 12).all()
