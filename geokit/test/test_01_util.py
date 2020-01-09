from .helpers import MASK_DATA, np
from geokit import util
import pytest 

# Scale Matrix
def test_scaleMatrix():
    # setup
    sumCheck = MASK_DATA.sum()

    # Equal down scale
    scaledMatrix1 = util.scaleMatrix(MASK_DATA, -2)
    assert np.isclose(scaledMatrix1.sum()*2*2, sumCheck)

    # Unequal down scale
    scaledMatrix2 = util.scaleMatrix(MASK_DATA, (-2, -4))
    assert np.isclose(scaledMatrix2.sum()*2*4, sumCheck)

    # Unequal up scale
    scaledMatrix3 = util.scaleMatrix(MASK_DATA, (2, 4))
    assert np.isclose(scaledMatrix3.sum()/2/4, sumCheck)

    # Strict downscale fail
    try:
        util.scaleMatrix(MASK_DATA, -3)
        assert(False)
    except util.GeoKitError:
        assert(True)
    else:
        assert(False)

    # non-strict downscale
    scaledMatrix5 = util.scaleMatrix(MASK_DATA, -3, strict=False)
    assert(scaledMatrix5.sum()/2/4 != sumCheck)

@pytest.mark.skip("No test implemented for: util.quickVector")
def test_quickVector(): assert False

@pytest.mark.skip("No test implemented for: util.quickRaster")
def test_quickRaster(): assert False

@pytest.mark.skip("No test implemented for: util.drawImage")
def test_drawImage(): assert False
