"""Regression tests for the data-type defects of issue #396.

One test per defect from ``Geokit_data_type_plan.md`` (D1-D28), its review (M1-M11) and the
behaviour-changes list (rows A14, A18). Every test asserts *values*, not only types.

Each test carries ``xfail(strict=True)`` with the defect ID. The PR that fixes a defect removes
the mark; because the marks are strict, a test that starts passing unexpectedly fails the run,
so the marks cannot go stale.

The rasters used here are built with plain GDAL so that GeoKit's type logic is not involved in
the inputs.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, gdal_array, ogr, osr

import geokit as gk
from geokit.error import GeoKitCDataError

# ----------------------------------------------------------------------------------------------
# helpers


def xfail(defect: str):
    return pytest.mark.xfail(strict=True, reason=defect)


def _srs(epsg: int) -> osr.SpatialReference:
    s = osr.SpatialReference()
    s.ImportFromEPSG(epsg)
    if gdal.__version__ >= "3.0.0":
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def gdal_raster(a, gdt, noData=None, scale=None, epsg=3035, px=100, x0=0.0, y0=None, driver="MEM", path=""):
    """A raster built with plain GDAL. ``y0`` is the top edge (defaults to rows * px)."""
    rows, cols = a.shape
    if y0 is None:
        y0 = rows * px
    ds = gdal.GetDriverByName(driver).Create(path, cols, rows, 1, gdt)
    ds.SetGeoTransform((x0, px, 0, y0, 0, -px))
    ds.SetProjection(_srs(epsg).ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(a)
    if noData is not None:
        band.SetNoDataValue(noData)
    if scale is not None:
        band.SetScale(scale)
    band.FlushCache()
    return ds


def gtiff(path, a, gdt, **kw):
    ds = gdal_raster(a, gdt, driver="GTiff", path=str(path), **kw)
    ds = None
    return str(path)


def band_type(x) -> str:
    ds = gk.raster.loadRaster(x)
    return gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)


def band_dtype(x) -> np.dtype:
    ds = gk.raster.loadRaster(x)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType))


def truth(ds, alg, px, **kw):
    """Reference: the same warp done by GDAL into Float64."""
    out = gdal.Warp("", ds, format="MEM", xRes=px, yRes=px, resampleAlg=alg, outputType=gdal.GDT_Float64, **kw)
    return out.ReadAsArray()


M = gk.raster.extractMatrix

PTS = [gk.geom.point(x, y, srs=4326) for x, y in [(6.1, 50.1), (7.5, 52.0), (6.8, 51.4)]]
GRID = dict(pixelWidth=0.5, pixelHeight=0.5, srs=4326, bounds=(5, 50, 8, 53))


def square(x0: float, size: float = 1000.0) -> ogr.Geometry:
    return gk.geom.polygon([(x0, 0), (x0 + size, 0), (x0 + size, size), (x0, size), (x0, 0)], srs=3035)


# ----------------------------------------------------------------------------------------------
# A. Type representation


@xfail("D1: float attribute rasterized as Int16, decimals truncated (#396)")
def test_D1_rasterize_float_attribute_exact():
    """A Real field is rasterized into Float64 with its decimals intact."""
    vec = gk.vector.createVector(pd.DataFrame({"geom": PTS, "v": [7.0, 9.3, 1.392]}))
    r = gk.vector.rasterize(vec, value="v", **GRID)
    assert band_type(r) == "Float64"
    assert set(np.unique(M(r))) == {0.0, 1.392, 7.0, 9.3}


@xfail("D1: integer attribute raises GeoKitCDataError('Unknown')")
def test_D1_rasterize_int16_attribute():
    """An int16 column rasterizes without error into an integer band that holds the values exactly.

    Only "integer" is asserted: the width depends on how the column is stored as an OGR field.
    """
    vec = gk.vector.createVector(pd.DataFrame({"geom": PTS, "v": np.array([7, 9, 1], np.int16)}))
    r = gk.vector.rasterize(vec, value="v", **GRID)
    assert np.issubdtype(band_dtype(r), np.integer)
    assert set(np.unique(M(r))) == {0, 1, 7, 9}


@xfail("D2: rasterize(value=200) burns 127 into an Int8 raster")
def test_D2_rasterize_constant_200():
    """A constant burn value of 200 gives a Byte band holding 200, not an Int8 band holding 127."""
    r = gk.vector.rasterize(gk.vector.createVector([square(0)]), pixelWidth=100, pixelHeight=100, value=200)
    assert band_type(r) == "Byte"
    assert M(r).max() == 200


@pytest.mark.parametrize(
    "value, expected_type",
    [(40000, "UInt16"), (2**31, "UInt32")],
    ids=["40000", "2**31"],
)
@xfail("M2: every unsigned-only constant is clipped to the signed maximum")
def test_M2_rasterize_constant_unsigned_ranges(value, expected_type):
    """A constant above the signed maximum gets the unsigned type that holds it."""
    r = gk.vector.rasterize(gk.vector.createVector([square(0)]), pixelWidth=100, pixelHeight=100, value=value)
    assert band_type(r) == expected_type
    assert M(r).max() == value


@xfail("M3: rasterize(value=1, dtype='Byte') returns Int8")
def test_M3_rasterize_explicit_byte():
    """The explicit dtype='Byte' is honoured by rasterize instead of becoming Int8."""
    r = gk.vector.rasterize(gk.vector.createVector([square(0)]), pixelWidth=100, pixelHeight=100, value=1, dtype="Byte")
    assert band_type(r) == "Byte"
    assert M(r).max() == 1


@pytest.mark.parametrize("dtype", ["Byte", "UInt16", "UInt32"])
@xfail("D3: explicit unsigned types are turned into signed ones")
def test_D3_createRaster_explicit_unsigned(dtype):
    """Explicit unsigned dtypes are honoured by createRaster and quickRaster."""
    r = gk.raster.createRaster(bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326, dtype=dtype)
    assert band_type(r) == dtype
    q = gk.util.quickRaster(bounds=(0, 0, 10, 10), srs=_srs(4326), dx=1, dy=1, dtype=dtype)
    assert band_type(q) == dtype


@xfail("D4: saveRasterAsTif writes a Byte raster as Int8 when every value fits Int8")
def test_D4_saveRasterAsTif_keeps_type(tmp_path):
    """A Byte source stays Byte: saveRasterAsTif writes an exact copy."""
    src = gdal_raster(np.array([[0, 50], [100, 127]], np.uint8), gdal.GDT_Byte)
    out = gk.raster.saveRasterAsTif(src, str(tmp_path / "saved.tif"))
    assert band_type(out) == "Byte"
    assert M(out).max() == 127


@xfail("D5: rasterMosaic clips 200 to 127")
def test_D5_rasterMosaic_keeps_byte():
    """The Byte type of the source survives rasterMosaic, so 200 is not clipped to 127."""
    src = gdal_raster(np.full((4, 4), 200, np.uint8), gdal.GDT_Byte)
    m = gk.Extent(0, 0, 400, 400, srs=3035).rasterMosaic([src], _skipFiltering=True)
    assert band_type(m) == "Byte"
    assert M(m).max() == 200


@xfail("D6: createRasterLike does not copy the dtype")
def test_D6_createRasterLike_inherits_dtype():
    """Without data, createRasterLike copies the source dtype."""
    src = gdal_raster(np.array([[0.5, 2.5]], np.float32), gdal.GDT_Float32)
    like = gk.raster.createRasterLike(src)
    assert band_type(like) == "Float32"


@pytest.mark.parametrize(
    "dtype, expected", [(np.float32, "Float32"), (float, "Float64"), (np.dtype("uint16"), "UInt16")]
)
@xfail("D7: non-string dtype is ignored")
def test_D7_createRaster_numpy_dtype(dtype, expected):
    """NumPy dtypes and Python float are accepted as dtype by createRaster."""
    r = gk.raster.createRaster(bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326, dtype=dtype)
    assert band_type(r) == expected


@xfail("D-13: a bare integer dtype must be rejected (GDAL and OGR constants overlap)")
def test_D13_bare_int_dtype_rejected():
    """A bare integer dtype is rejected: GDAL and OGR constants overlap, which caused #396."""
    with pytest.raises(GeoKitCDataError):
        gk.raster.createRaster(bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326, dtype=gdal.GDT_Float32)


# ----------------------------------------------------------------------------------------------
# B. Promotion rules


@xfail("D8: integer IDs with noData=nan are stored as Float32 and lose precision")
def test_D8_rasterize_int_ids_nan_nodata_exact():
    """Integer IDs rasterized with noData=NaN go to Float64 and stay exact."""
    ids = gk.vector.createVector(pd.DataFrame({"geom": PTS, "id": np.array([123456789, 987654321, 5], np.int64)}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Int64 + NaN may warn (exact up to 2**53)
        r = gk.vector.rasterize(ids, value="id", noData=np.nan, **GRID)
    assert band_type(r) == "Float64"
    vals = M(r)
    assert {123456789.0, 987654321.0, 5.0} <= set(np.unique(vals[~np.isnan(vals)]))


@xfail("D8: warp of Int32 with noData=nan gives Float32")
def test_D8_warp_int32_nan_nodata_exact():
    """Warping Int32 with noData=NaN gives Float64, not a Float32 that rounds the values."""
    src = gdal_raster(np.array([[123456789, 987654321]], np.int32), gdal.GDT_Int32)
    w = gk.raster.warp(src, noData=np.nan, resampleAlg="near")
    assert band_type(w) == "Float64"
    assert set(np.unique(M(w))) == {123456789.0, 987654321.0}


@xfail("D10: an explicit dtype that cannot hold noData is silently widened")
def test_D10_explicit_dtype_cannot_hold_nodata_raises():
    """An explicit dtype that cannot hold noData raises instead of being widened silently."""
    with pytest.raises(GeoKitCDataError):
        gk.raster.createRaster(bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326, dtype="UInt16", noData=-1)


@xfail("D11: warp(dtype='Float32') on a Float64 source returns Float64 with a warning")
def test_D11_warp_explicit_narrower_float():
    """warp(dtype='Float32') on a Float64 source returns Float32."""
    src = gdal_raster(np.array([[0.5, 0.25]], np.float64), gdal.GDT_Float64)
    w = gk.raster.warp(src, dtype="Float32", resampleAlg="near")
    assert band_type(w) == "Float32"
    assert set(np.unique(M(w))) == {0.25, 0.5}


# ----------------------------------------------------------------------------------------------
# C. Operations that widen the value range


@xfail("D12: averaging a 0/1 mask stores 0.75 as 1")
def test_D12_warp_average_fractional():
    """Averaging a 0/1 mask keeps the fractions; the reference is gdal.Warp into Float64."""
    mask = np.zeros((20, 20), np.uint8)
    mask[:, :7] = 1
    src = gdal_raster(mask, gdal.GDT_Byte)
    w = gk.raster.warp(src, resampleAlg="average", pixelWidth=400, pixelHeight=400)
    np.testing.assert_allclose(M(w), truth(src, "average", 400), atol=1e-6)


@xfail("D13: cubic overshoot is clipped to the Byte range")
def test_D13_warp_cubic_overshoot_kept():
    """Cubic overshoot outside the Byte range is kept; the reference is gdal.Warp into Float64."""
    step = np.zeros((20, 20), np.uint8)
    step[:, 10:] = 255
    src = gdal_raster(step, gdal.GDT_Byte)
    w = gk.raster.warp(src, resampleAlg="cubic", pixelWidth=30, pixelHeight=30)
    ref = truth(src, "cubic", 30)
    assert ref.min() < 0 < 255 < ref.max()  # the reference really overshoots
    np.testing.assert_allclose(M(w), ref, atol=1e-3)


@xfail("D14: warp(sum) into Byte clips 3200 to 255")
def test_D14_warp_sum_not_clipped():
    """warp(sum) of 16 Byte pixels of 200 gives 3200 instead of clipping at 255."""
    src = gdal_raster(np.full((20, 20), 200, np.uint8), gdal.GDT_Byte)
    w = gk.raster.warp(src, resampleAlg="sum", pixelWidth=400, pixelHeight=400)
    np.testing.assert_allclose(M(w), truth(src, "sum", 400), rtol=1e-9)
    assert M(w).max() == 3200


@xfail("D15: reprojection creates unflagged zero pixels without any warning")
def test_D15_warp_reprojection_warns_about_created_pixels():
    """A reprojection without noData warns that the created pixels hold 0."""
    src = gdal_raster(np.arange(1, 401, dtype=np.int16).reshape(20, 20), gdal.GDT_Int16, x0=4_000_000, y0=3_000_000)
    with pytest.warns(UserWarning, match="not flagged"):
        w = gk.raster.warp(src, srs=4326, resampleAlg="near")
    assert (M(w) == 0).any()  # GDAL behaviour is kept; the user is told about it


@xfail("D16: rasterize(add=True) overflows Int8")
def test_D16_rasterize_add_overflow():
    """rasterize(add=True) sums overlapping features without overflowing the band type."""
    vec = gk.vector.createVector([square(0), square(500)])
    r = gk.vector.rasterize(vec, pixelWidth=100, pixelHeight=100, value=100, add=True)
    assert M(r).max() == 200


@xfail("D17: gradient of a UInt16 DEM wraps around")
def test_D17_gradient_unsigned():
    """The gradient is computed in float, so a UInt16 DEM does not wrap around."""
    dem = np.array([[100, 101, 102, 103]] * 4, np.uint16)
    g = gk.raster.gradient(gdal_raster(dem, gdal.GDT_UInt16), mode="east-west", asMatrix=True)[:, 1:-1]
    np.testing.assert_allclose(g, -0.01)


@xfail("incidental: gradient(mode='ew') raises UnboundLocalError")
def test_gradient_ew_mode():
    """gradient(mode='ew') is the same as mode='east-west'."""
    dem = np.array([[100.0, 101.0, 102.0, 103.0]] * 4)
    ew = gk.raster.gradient(gdal_raster(dem, gdal.GDT_Float64), mode="ew", asMatrix=True)
    full = gk.raster.gradient(gdal_raster(dem, gdal.GDT_Float64), mode="east-west", asMatrix=True)
    np.testing.assert_array_equal(ew, full)


@xfail("D18: KernelProcessor pads with an integer array and truncates floats")
def test_D18_kernel_processor_float_padding():
    """KernelProcessor pads in the matrix dtype, so an integer edgeValue does not truncate floats."""
    mat = np.array([[0.5, 1.5], [2.5, 3.5]])
    out = gk.util.KernelProcessor(1, edgeValue=0)(lambda m: m[1, 1])(mat)
    np.testing.assert_array_equal(out, mat)


@xfail("D19: combineSimilarRasters turns Byte into Int8")
def test_D19_combineSimilarRasters_keeps_byte(tmp_path):
    """Byte and the values of both inputs survive combineSimilarRasters."""
    from geokit._algorithms.combineSimilarRasters import combineSimilarRasters

    a = gtiff(tmp_path / "a.tif", np.full((2, 2), 100, np.uint8), gdal.GDT_Byte)
    b = gtiff(tmp_path / "b.tif", np.full((2, 2), 200, np.uint8), gdal.GDT_Byte, x0=200)
    out = str(tmp_path / "combined.tif")
    combineSimilarRasters([a, b], output=out, verbose=False)
    assert band_type(out) == "Byte"
    assert set(np.unique(M(out))) == {100, 200}


@xfail("M1: warp(near) of a Byte raster returns Int8")
def test_M1_warp_near_keeps_byte():
    """warp(near) keeps the Byte type of the source."""
    src = gdal_raster(np.array([[0, 1], [2, 3]], np.uint8), gdal.GDT_Byte)
    w = gk.raster.warp(src, resampleAlg="near")
    assert band_type(w) == "Byte"


@pytest.mark.parametrize(
    "dtype, value, expected_type",
    [(np.uint8, 200, "Byte"), (np.uint16, 40000, "UInt16")],
    ids=["uint8", "uint16"],
)
@xfail("M11: mutateRaster clips unsigned processor output")
def test_M11_mutateRaster_unsigned_output(dtype, value, expected_type):
    """An unsigned processor output is kept by mutateRaster instead of being clipped to the signed range."""
    src = gdal_raster(np.ones((4, 4), np.uint8), gdal.GDT_Byte)
    m = gk.raster.mutateRaster(src, processor=lambda a: a.astype(dtype) * dtype(value))
    assert band_type(m) == expected_type
    assert M(m).max() == value


@xfail("D28: dtype='bool' maps to Int8; Byte is the compatible choice")
def test_D28_mutateRaster_bool_gives_byte():
    """mutateRaster(dtype='bool') gives a Byte band holding 0 and 1."""
    src = gdal_raster(np.array([[0, 5], [10, 15]], np.uint8), gdal.GDT_Byte)
    m = gk.raster.mutateRaster(src, processor=lambda a: a > 5, dtype="bool")
    assert band_type(m) == "Byte"
    assert set(np.unique(M(m))) == {0, 1}


# ----------------------------------------------------------------------------------------------
# D. Crossing between raster and vector


@xfail("D20: uint32 above 2**31 is written as OFTInteger and clipped")
def test_D20_createVector_uint32_large():
    """A uint32 value above 2**31 survives createVector."""
    ds = gk.vector.createVector(pd.DataFrame({"geom": [PTS[0]], "v": np.array([3_000_000_000], np.uint32)}))
    assert ds.GetLayer().GetNextFeature().GetField("v") == 3_000_000_000


@pytest.mark.parametrize(
    "column, expected",
    [
        (np.array([5], np.uint64), 5),
        (np.array([1.5], np.float16), 1.5),
        (pd.array([True], dtype="boolean"), 1),
    ],
    ids=["uint64", "float16", "pandas-boolean"],
)
@xfail("D20: uint64, float16 and pandas boolean columns fall back to OFTString")
def test_D20_createVector_column_types(column, expected):
    """uint64, float16 and pandas boolean columns become numeric fields, not strings."""
    ds = gk.vector.createVector(pd.DataFrame({"geom": [PTS[0]], "v": column}))
    value = ds.GetLayer().GetNextFeature().GetField("v")
    assert not isinstance(value, str)
    assert value == expected


@xfail("D21: polygonizeRaster of UInt32 values above 2**31 returns 2147483647")
def test_D21_polygonizeRaster_uint32():
    """UInt32 values above 2**31 survive polygonizeRaster."""
    src = gdal_raster(np.full((4, 4), 3_000_000_000, np.uint32), gdal.GDT_UInt32)
    df = gk.raster.polygonizeRaster(src)
    assert list(df["value"]) == [3_000_000_000]


@xfail("D21: polygonizeRaster silently truncates float rasters and merges polygons")
def test_D21_polygonizeRaster_float_raises():
    """A float raster is rejected by polygonizeRaster instead of being truncated and merged into one polygon."""
    fl = np.full((4, 4), 1.7, np.float32)
    fl[:, 2:] = 2.4
    with pytest.raises(GeoKitCDataError):
        gk.raster.polygonizeRaster(gdal_raster(fl, gdal.GDT_Float32))


@xfail("D22: polygonizeMatrix always uses an Int32 raster")
def test_D22_polygonizeMatrix_uint32():
    """Values above 2**31 in a uint32 matrix survive polygonizeMatrix."""
    df = gk.geom.polygonizeMatrix(np.full((2, 2), 3_000_000_000, np.uint32))
    assert list(df["value"]) == [3_000_000_000]


# ----------------------------------------------------------------------------------------------
# E. Values handled in NumPy


def _scaled_raster():
    return gdal_raster(np.array([[-9999, 100, 200]], np.int16), gdal.GDT_Int16, noData=-9999, scale=0.1)


@xfail("D23: extractMatrix(autocorrect=True) compares noData after scaling")
def test_D23_extractMatrix_autocorrect_scaled():
    """extractMatrix(autocorrect=True) masks noData on the raw values before applying the scale."""
    out = M(_scaled_raster(), autocorrect=True)
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[0, 1:], [10.0, 20.0])


@xfail("D23: extractValues compares noData after scaling")
def test_D23_extractValues_scaled_nodata():
    """The noData mask of extractValues is built on the raw values, before the scale is applied."""
    src = _scaled_raster()
    got = gk.raster.extractValues(src, [(50, 50), (150, 50)], pointSRS=3035)
    assert np.isnan(got.data[0])
    assert np.isclose(got.data[1], 10.0)


@xfail("D24: rasterStats treats scaled noData as data")
def test_D24_rasterStats_scaled():
    """The statistics of rasterStats leave out the noData pixels of a scaled raster."""
    stats = gk.raster.rasterStats(_scaled_raster())
    assert stats.nobs == 2
    assert np.isclose(stats.mean, 15.0)


@pytest.mark.parametrize("noData", [-1, np.nan], ids=["-1", "nan"])
@xfail("D25: indicateValues writes noData into a bool array and indicates every noData pixel")
def test_D25_indicateValues_nodata_not_indicated(noData):
    """NoData pixels are not indicated by indicateValues, for an integer and for a NaN noData."""
    half = np.full((10, 10), 5.0, np.float32)
    half[:, :5] = np.nan
    src = gk.raster.createRaster(
        bounds=(0, 0, 1000, 1000), pixelWidth=100, pixelHeight=100, srs=3035, data=half, noData=np.nan
    )
    rm = gk.RegionMask.fromGeom(gk.geom.box(0, 0, 1000, 1000, srs=3035), pixelRes=100, srs=3035)
    ind = rm.indicateValues(src, value="[0-10]", noData=noData, applyMask=False, multiProcess=False)
    is_nodata = np.isnan(ind) if np.isnan(noData) else ind == noData
    assert is_nodata.sum() == 50
    assert (ind[~is_nodata] == 1).sum() == 50


@xfail("D26: applyMask wraps a NumPy-integer noData into uint8")
def test_D26_applyMask_numpy_int_nodata():
    """A uint8 matrix is promoted by applyMask so that a NumPy-integer noData of -1 is stored, not wrapped to 255."""
    tri = gk.RegionMask.fromGeom(
        gk.geom.polygon([(0, 0), (1000, 0), (0, 1000), (0, 0)], srs=3035), pixelRes=100, srs=3035
    )
    out = tri.applyMask(np.ones(tri.mask.shape, np.uint8), noData=np.int64(-1))
    assert out.min() == -1
    assert out[tri.mask].min() == 1


# ----------------------------------------------------------------------------------------------
# F. Other


@xfail("A14: createRaster(noData=x) without fill or data fills with 0")
def test_A14_createRaster_nodata_fill():
    """createRaster(noData=x) without data or fill is filled with noData, not with 0."""
    r = gk.raster.createRaster(bounds=(0, 0, 200, 100), pixelWidth=100, pixelHeight=100, srs=3035, noData=5)
    assert set(np.unique(M(r))) == {5}


@xfail("A18: vectorInfo reports GDAL names of OGR constants")
def test_A18_vectorInfo_field_type_names():
    """The field type names reported by vectorInfo are OGR names, not GDAL names of the OGR constants."""
    vec = gk.vector.createVector(
        pd.DataFrame({"geom": PTS, "i": [1, 2, 3], "f": [0.5, 1.5, 2.5], "s": ["a", "b", "c"]})
    )
    info = gk.vector.vectorInfo(vec)
    assert info.attribute_data_types_str == {"i": "Integer64", "f": "Real", "s": "String"}


@xfail("M4: RegionMask.createRaster() creates Int8 and RegionMask.rasterize() returns int8")
def test_M4_regionmask_byte():
    """RegionMask.createRaster gives a Byte band and RegionMask.rasterize returns uint8."""
    rm = gk.RegionMask.fromGeom(square(0), pixelRes=100, srs=3035)
    assert band_type(rm.createRaster()) == "Byte"
    mat = rm.rasterize(gk.vector.createVector([square(0)]), value=1)
    assert mat.dtype == np.uint8
    assert mat.max() == 1
