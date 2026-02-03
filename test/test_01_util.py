import geokit.core.util
from geokit import util
from geokit.core.srs import loadSRS
from geokit.core.raster import rasterInfo, extractMatrix
from test.helpers import AACHEN_SHAPE_PATH, CLC_RASTER_PATH, MASK_DATA, np


from geokit.error import GeoKitError


def test_scaleMatrix():
    # setup
    sumCheck = MASK_DATA.sum()

    # Equal down scale
    scaledMatrix1 = util.scaleMatrix(MASK_DATA, -2)
    assert np.isclose(scaledMatrix1.sum() * 2 * 2, sumCheck)

    # Unequal down scale
    scaledMatrix2 = util.scaleMatrix(MASK_DATA, (-2, -4))
    assert np.isclose(scaledMatrix2.sum() * 2 * 4, sumCheck)

    # Unequal up scale
    scaledMatrix3 = util.scaleMatrix(MASK_DATA, (2, 4))
    assert np.isclose(scaledMatrix3.sum() / 2 / 4, sumCheck)

    # Strict downscale fail
    try:
        util.scaleMatrix(MASK_DATA, -3)
        assert False
    except GeoKitError:
        assert True
    else:
        assert False

    # non-strict downscale
    scaledMatrix5 = util.scaleMatrix(MASK_DATA, -3, strict=False)
    assert scaledMatrix5.sum() / 2 / 4 != sumCheck


def test_isRaster():
    s1 = util.isRaster(CLC_RASTER_PATH)
    assert s1 == True

    s2 = util.isRaster(AACHEN_SHAPE_PATH)
    assert s2 == False


def test_isVector():
    s1 = util.isVector(CLC_RASTER_PATH)
    assert s1 == False

    s2 = util.isVector(AACHEN_SHAPE_PATH)
    assert s2 == True


def test_fitBoundsTo():
    inBounds = (50.7, 6.8, 52.3, 7.8)
    # cast to full degree resolution
    dx = 1.0  # test float
    dy = 1  # test int
    # simple rounding
    outBounds1 = util.fitBoundsTo(
        bounds=inBounds,
        dx=dx,
        dy=dy,
        startAtZero=False,  # default
        expand=False,  # default
    )
    assert outBounds1 == (
        51.0,
        6.8,
        52.0,
        7.8,
    )  # bounds height is a multiple of dy, but does not start at zero
    # start bounds at zero, i.e. every bounds value must e a multiple of dx/dy
    outBounds2 = util.fitBoundsTo(
        bounds=inBounds,
        dx=dx,
        dy=dy,
        startAtZero=True,
        expand=False,  # default
    )
    assert outBounds2 == (
        51.0,
        7.0,
        52.0,
        8.0,
    )  # bounds are now all multiples of dx/dy, but inBounds are not fully included
    # require expansion - ensures that inBounds are fully included in outBounds
    outBounds3 = util.fitBoundsTo(
        bounds=inBounds,
        dx=dx,
        dy=dy,
        startAtZero=False,  # default
        expand=True,
    )
    assert (
        outBounds3
        == (
            50.0,
            6.8,
            53.0,
            7.8,
        )
    )  # outBounds now fully include inBounds and height/width are multiples of dx/dy, but bounds entries are not necessarily multiples
    # require expansion AND that each bounds entry must be a multiple of dx/dy
    outBounds4 = util.fitBoundsTo(
        bounds=inBounds,
        dx=dx,
        dy=dy,
        startAtZero=True,
        expand=True,
    )
    assert outBounds4 == (
        50.0,
        6.0,
        53.0,
        8.0,
    )  # outBounds now fully include inBounds and each entry is a multiple of dx/dy


def test_quickRaster():
    load_srs = loadSRS(source=4326)
    new_raster = geokit.core.util.quickRaster(bounds=(0, 0, 4, 4), srs=load_srs, dx=1, dy=1, noData=-9999)
    extracted_raster = extractMatrix(source=new_raster)
    raster_for_comparison = np.array(
        [
            [-9999, -9999, -9999, -9999],
            [-9999, -9999, -9999, -9999],
            [-9999, -9999, -9999, -9999],
            [-9999, -9999, -9999, -9999],
        ],
        dtype="int16",
    )
    np.testing.assert_allclose(actual=extracted_raster, desired=raster_for_comparison)
