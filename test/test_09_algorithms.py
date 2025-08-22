from geokit._algorithms.combineSimilarRasters import (
    checkSimilarRasters,
    combineSimilarRasters,
)
from geokit import raster, util
import numpy as np
import pytest
from test.helpers import (
    DIVIDED_RASTER_1_PATH,
    DIVIDED_RASTER_2_PATH,
    DIVIDED_RASTER_3_PATH,
)


def test_checkSimilarRasters():

    # PREPARE DATA

    # get these test raster paths with exactly matching contexts from disk
    test_rasters = [DIVIDED_RASTER_1_PATH, DIVIDED_RASTER_2_PATH, DIVIDED_RASTER_3_PATH]

    # get the raster info for the first (#0) raster to adapt
    rInfo0 = raster.rasterInfo(test_rasters[0])
    # create an alternative first raster with bounds slightly adapted
    rstr_boundschanged = raster.warp(
        source=test_rasters[0],
        bounds=tuple(
            np.array(rInfo0.bounds) + 0.001
        ),  # only shift the bounds slightly upwards to the right
        resampleAlg="near",
    )
    # create an alternative first raster with x-resolution slightly adapted
    rstr_dxchanged = raster.warp(
        source=test_rasters[0],
        pixelWidth=0.99999999999 * rInfo0.pixelWidth,  # make it slightly smaller
        bounds=(
            rInfo0.bounds[0],
            rInfo0.bounds[1],
            rInfo0.bounds[2] * 0.99999999999,
            rInfo0.bounds[3],
        ),  # shrink bounds width by the same factor
        resampleAlg="near",
    )

    # FIRST TEST EXACT MATCH

    # must work for the original rasters
    checkSimilarRasters(
        datasets=test_rasters,
        rtol=0,
    )
    # must fail for the shifted bounds...
    with pytest.raises(util.GeoKitError):
        checkSimilarRasters(
            datasets=[rstr_boundschanged] + test_rasters[1:],
            rtol=0,  # no tolerance = exact match
        )
    # ... and the shifted x res
    with pytest.raises(util.GeoKitError):
        checkSimilarRasters(
            datasets=[rstr_dxchanged] + test_rasters[1:],
            rtol=0,  # no tolerance = exact match
        )

    # NOW CHECK SIMILAR MATCH

    # should pass when sufficient relative tolerance is allowed...
    checkSimilarRasters(
        datasets=[rstr_boundschanged] + test_rasters[1:],
        rtol=0.00001,  # normal tolerance
    )
    # but not when rtol is too tight
    with pytest.raises(util.GeoKitError):
        checkSimilarRasters(
            datasets=[rstr_boundschanged] + test_rasters[1:],
            rtol=0.00000000001,  # super narrow tolerance
        )
    # the same for the x-resolution mismatch
    checkSimilarRasters(
        datasets=[rstr_dxchanged] + test_rasters[1:],
        rtol=0.00001,  # normal tolerance
    )
    # and again fail for an excessively tight tolerance
    with pytest.raises(util.GeoKitError):
        checkSimilarRasters(
            datasets=[rstr_dxchanged] + test_rasters[1:],
            rtol=0.00000000001,  # super narrow tolerance
        )


def test_combineSimilarRasters():
    # first test the straightforward way with aligned rasters
    metadata = {"test": "data"}
    new_rstr = combineSimilarRasters(
        datasets=[DIVIDED_RASTER_1_PATH, DIVIDED_RASTER_2_PATH, DIVIDED_RASTER_3_PATH],
        output=None,
        combiningFunc=None,
        verbose=True,
        updateMeta=False,
        allowPreWarp=True,
        meta=metadata,
    )
    # make also sure that the kwarg was passed correctly
    assert raster.rasterInfo(new_rstr).meta == metadata

    # now test the prewarp route, therefore slightly alter one of the rasters
    rInfo1 = raster.rasterInfo(DIVIDED_RASTER_1_PATH)
    DIVIDED_RASTER_1_SHIFTED = raster.warp(
        source=DIVIDED_RASTER_1_PATH,
        bounds=tuple(
            np.array(rInfo1.bounds) + 0.001
        ),  # only shift the bounds slightly upwards to the right
        resampleAlg="near",
    )
    new_rstr2 = combineSimilarRasters(
        datasets=[
            DIVIDED_RASTER_1_SHIFTED,
            DIVIDED_RASTER_2_PATH,
            DIVIDED_RASTER_3_PATH,
        ],  # with shifted raster
        output=None,
        combiningFunc=None,
        verbose=True,
        updateMeta=False,
        allowPreWarp=True,
    )
    # make sure the slightly corrected output remains the same as with the unaltered rasters
    mx = raster.extractMatrix(new_rstr)
    mx2 = raster.extractMatrix(new_rstr2)
    assert np.array_equal(mx, mx2)
    # make sure that the output raster/matrix remains the same - no changes in the process
    assert mx2.shape == (1535, 1736)
    assert mx2.sum() == 38610800
    np.median(
        mx2, axis=0
    ).mean() == 16.028225806451612  # this would change if the matrix was transposed

    # test again with pixel res deviation
    DIVIDED_RASTER_1_SHRUNK = raster.warp(
        source=DIVIDED_RASTER_1_PATH,
        pixelWidth=0.99999999999 * rInfo1.pixelWidth,  # make it slightly smaller
        bounds=(
            rInfo1.bounds[0],
            rInfo1.bounds[1],
            rInfo1.bounds[2] * 0.99999999999,
            rInfo1.bounds[3],
        ),  # shrink bounds width by the same factor
        resampleAlg="near",
    )
    new_rstr3 = combineSimilarRasters(
        datasets=[
            DIVIDED_RASTER_1_SHRUNK,
            DIVIDED_RASTER_2_PATH,
            DIVIDED_RASTER_3_PATH,
        ],  # with shifted raster
        output=None,
        combiningFunc=None,
        verbose=True,
        updateMeta=False,
        allowPreWarp=True,
    )

    assert np.isclose(
        raster.extractMatrix(new_rstr),
        raster.extractMatrix(new_rstr3),
    ).all()

    # now try the same again but WITHOUT prewarp - must fail
    with pytest.raises(util.GeoKitError):
        combineSimilarRasters(
            datasets=[
                DIVIDED_RASTER_1_SHRUNK,
                DIVIDED_RASTER_2_PATH,
                DIVIDED_RASTER_3_PATH,
            ],  # with shifted raster
            output=None,
            combiningFunc=None,
            verbose=True,
            updateMeta=False,
            allowPreWarp=False,  # not allowed to prewarp rasters to same context
        )
