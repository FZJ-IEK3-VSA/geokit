import geokit.core.util
from geokit import util
from geokit.core.srs import loadSRS
from geokit.core.raster import rasterInfo, extractMatrix
from test.helpers import AACHEN_SHAPE_PATH, CLC_RASTER_PATH, MASK_DATA, np
from geokit.data_types import (
    _gdal_c_raster_data_types_list,
)

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


# @pytest.mark.skip("No test implemented for: util.quickVector")
# def test_quickVector():
#     assert False


def test_quickRaster():
    load_srs = loadSRS(source=4326)
    new_raster = geokit.core.util.quickRaster(bounds=(0, 0, 4, 4), srs=load_srs, dx=1, dy=1, noData=-9999, fill=-9999)
    # new_raster = geokit.core.util.quickRaster(
    #     bounds=(0, 0, 4, 4), srs=load_srs, dx=1, dy=1, noData=255, fill=None, dtype="Unknown"
    # )
    # print("Hello")
    # new_raster = geokit.core.util.quickRaster(
    #     bounds=(0, 0, 4, 4), srs=load_srs, dx=1, dy=1, noData=None, fill=None, dtype="Unknown"
    # )
    # print("Hello2")
    # new_band = new_raster.GetRasterBand(1)
    # new_band.ComputeStatistics(0)
    # get_maximum = new_band.GetMaximum()
    # get_minimum = new_band.GetMinimum()
    # print(get_maximum)
    # print(get_minimum)
    # raster_info = rasterInfo(sourceDS=new_raster)
    # raster_info.data_type_name_str
    extracted_raster = extractMatrix(source=new_raster)
    print(extracted_raster)
    # np.array()
    pass


def test_raster_datatypes():
    load_srs = loadSRS(source=4326)
    for current_datatype in _gdal_c_raster_data_types_list:
        new_raster = geokit.core.util.quickRaster(bounds=(0, 0, 4, 4), srs=load_srs, dx=1, dy=1, dtype=current_datatype)
        raster_info = rasterInfo(sourceDS=new_raster)
        output_datatype = raster_info.data_type_name_str
        print("Input datatype: ", current_datatype)
        print("Output datatype: ", output_datatype, "\n")
        # assert output_datatype == current_datatype
    # extracted_raster = extractMatrix(source=new_raster)
    # print(extracted_raster)
    # np.array()
    pass


# @pytest.mark.skip("No test implemented for: util.drawImage")
# def test_drawImage():
#     assert False


# @pytest.mark.skip("No test implemented for: util.KernelProcessor")
# def test_KernelProcessor():
#     assert False


# def test_get_common_dtype():
#     dtypes = [7, 2, 3, 5, 3]  # must yield 7 as the most versatile

#     out = GdalDataTypeHandler.get_common_dtype(dtypes=dtypes, fallback=11)
#     assert out == 7

#     dtypes = [7, 2, 3, 5, 3, 10]  # 7 cannot be represented by 10, so use 11
#     out = GdalDataTypeHandler.get_common_dtype(dtypes=dtypes, fallback=11)
#     assert out == 11

#     dtypes = [7, 2, 3, 5, 3, 10, 15]  # 15 is not a known datatype!
#     # one option is to use fallback
#     fallback = 11
#     out = GdalDataTypeHandler.get_common_dtype(dtypes=dtypes, fallback=fallback)
#     assert out == fallback
#     # another to raise an error
#     with pytest.raises(TypeError):
#         out = GdalDataTypeHandler.get_common_dtype(dtypes=dtypes, fallback=None)


if __name__ == "__main__":
    test_quickRaster()
    # test_raster_datatypes()
    # promote_dtype()
    # test_numpy_min_scalar()
    # test_min_max_values_np_data_types()
    # pass
