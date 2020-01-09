from .helpers import MASK_DATA, np
from geokit.util import scaleMatrix, GeoKitError

# Scale Matrix
def test_scaleMatrix():
    # setup
    sumCheck = MASK_DATA.sum()

    # Equal down scale
    scaledMatrix1 = scaleMatrix(MASK_DATA, -2)
    assert np.isclose(scaledMatrix1.sum()*2*2, sumCheck)

    # Unequal down scale
    scaledMatrix2 = scaleMatrix(MASK_DATA, (-2, -4))
    assert np.isclose(scaledMatrix2.sum()*2*4, sumCheck)

    # Unequal up scale
    scaledMatrix3 = scaleMatrix(MASK_DATA, (2, 4))
    assert np.isclose(scaledMatrix3.sum()/2/4, sumCheck)

    # Strict downscale fail
    try:
        scaleMatrix(MASK_DATA, -3)
        assert(False)
    except GeoKitError:
        assert(True)
    else:
        assert(False)

    # non-strict downscale
    scaledMatrix5 = scaleMatrix(MASK_DATA, -3, strict=False)
    assert(scaledMatrix5.sum()/2/4 != sumCheck)

def test_quickVector(): assert(False)
def test_quickRaster(): assert(False)
def test_drawImage(): assert(False)
