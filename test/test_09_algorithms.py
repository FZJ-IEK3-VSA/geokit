from geokit._algorithms.combineSimilarRasters import combineSimilarRasters
from geokit import raster, util
import numpy as np
import pytest
from test.helpers import (
    DIVIDED_RASTER_1_PATH,
    DIVIDED_RASTER_2_PATH,
    DIVIDED_RASTER_3_PATH,
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
    # make sure that the output raster/matrix remains the same - no changes in the process
    mx2 = raster.extractMatrix(new_rstr2)
    assert mx2.shape == (1535, 1736)
    assert mx2.sum() == 38610800
    np.median(mx2, axis=0).mean() == 16.028225806451612 # this would change if the matrix was transposed

    assert np.isclose(
        raster.extractMatrix(new_rstr),
        raster.extractMatrix(new_rstr2),
    ).all()

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
