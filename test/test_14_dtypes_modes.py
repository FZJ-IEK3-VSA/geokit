"""The ``dtype`` modes ``"auto"``, ``"preserve_input"`` and ``"smallest"``.

Encodes the worked examples of ``Geokit_data_type_pr_stack.md`` section 1.6. Every case is run
for ``dtype=None`` (which must behave like ``"auto"``) and the three mode strings.

All tests are ``xfail(strict=True)`` until the ``geokit.dtypes`` module and its call sites land
(PRs 4-8 of the stack). A case whose expectation is a tuple ``(type, "warning")`` must emit a
``GeoKitDataTypeWarning`` (a ``UserWarning``); ``"error"`` must raise ``GeoKitCDataError`` (the
base class of ``GeoKitDataTypeError``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

import geokit as gk
from geokit.error import GeoKitCDataError
from test.test_13_dtypes_regression import M, band_type, gdal_raster, gtiff, square, _srs

MODES = [None, "auto", "preserve_input", "smallest"]

XFAIL = pytest.mark.xfail(strict=True, reason="dtype modes are implemented in PRs 4-8 of the stack")

# (case, mode) combinations whose expectation already holds today; they carry no xfail mark so
# that the marks stay strict for the real defects.
PASSING_TODAY = {
    ("createRaster_int64_small_values", None),
    ("mutateRaster_float_processor", None),
}


# ----------------------------------------------------------------------------------------------
# cases: name -> (callable(tmp_path, dtype) -> raster, {mode: expected})
#   expected is a GDAL type name, (type name, "warning") or "error"


def _points_vector(values, dtype=None):
    pts = [gk.geom.point(x, y, srs=4326) for x, y in [(6.1, 50.1), (7.5, 52.0), (6.8, 51.4)]]
    col = np.array(values, dtype) if dtype is not None else values
    return gk.vector.createVector(pd.DataFrame({"geom": pts, "v": col}))


GRID = dict(pixelWidth=0.5, pixelHeight=0.5, srs=4326, bounds=(5, 50, 8, 53))
BOUNDS = dict(bounds=(0, 0, 10, 10), pixelWidth=1, pixelHeight=1, srs=4326)


def _mask_raster():
    mask = np.zeros((20, 20), np.uint8)
    mask[:, :7] = 1
    return gdal_raster(mask, gdal.GDT_Byte)


def _int32_raster():
    return gdal_raster(np.arange(16, dtype=np.int32).reshape(4, 4), gdal.GDT_Int32)


CASES = {
    "createRaster_empty": (
        lambda _tmp, dt: gk.raster.createRaster(dtype=dt, **BOUNDS),
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "createRaster_int64_small_values": (
        lambda _tmp, dt: gk.raster.createRaster(
            data=np.arange(100, dtype=np.int64).reshape(10, 10) % 6, dtype=dt, **BOUNDS
        ),
        {"auto": "Int64", "preserve_input": "Int64", "smallest": "Byte"},
    ),
    "createRaster_uint8_nodata_-1": (
        lambda _tmp, dt: gk.raster.createRaster(data=np.ones((10, 10), np.uint8), noData=-1, dtype=dt, **BOUNDS),
        {"auto": "Int16", "preserve_input": "error", "smallest": "Int16"},
    ),
    "rasterize_1": (
        lambda _tmp, dt: gk.vector.rasterize(gk.vector.createVector([square(0)]), 100, 100, value=1, dtype=dt),
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "rasterize_200": (
        lambda _tmp, dt: gk.vector.rasterize(gk.vector.createVector([square(0)]), 100, 100, value=200, dtype=dt),
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "rasterize_40000": (
        lambda _tmp, dt: gk.vector.rasterize(gk.vector.createVector([square(0)]), 100, 100, value=40000, dtype=dt),
        {"auto": "UInt16", "preserve_input": "UInt16", "smallest": "UInt16"},
    ),
    "rasterize_-1": (
        lambda _tmp, dt: gk.vector.rasterize(gk.vector.createVector([square(0)]), 100, 100, value=-1, dtype=dt),
        {"auto": "Int16", "preserve_input": "Int16", "smallest": "Int16"},
    ),
    "rasterize_0.1": (
        lambda _tmp, dt: gk.vector.rasterize(gk.vector.createVector([square(0)]), 100, 100, value=0.1, dtype=dt),
        {"auto": "Float64", "preserve_input": "Float64", "smallest": "Float64"},
    ),
    "rasterize_integer_field": (
        lambda _tmp, dt: gk.vector.rasterize(_points_vector([1, 2, 3], np.int32), value="v", dtype=dt, **GRID),
        {"auto": "Int32", "preserve_input": "Int32", "smallest": "Byte"},
    ),
    "rasterize_real_field": (
        lambda _tmp, dt: gk.vector.rasterize(_points_vector([0.5, 1.5, 2.5]), value="v", dtype=dt, **GRID),
        {"auto": "Float64", "preserve_input": "Float64", "smallest": "Float32"},
    ),
    "rasterize_int64_field_nan_nodata": (
        lambda _tmp, dt: gk.vector.rasterize(
            _points_vector([1, 2, 3], np.int64), value="v", noData=np.nan, dtype=dt, **GRID
        ),
        {"auto": ("Float64", "warning"), "preserve_input": "error", "smallest": ("Float64", "warning")},
    ),
    "rasterize_add_100_x2": (
        lambda _tmp, dt: gk.vector.rasterize(
            gk.vector.createVector([square(0), square(500)]), 100, 100, value=100, add=True, dtype=dt
        ),
        {"auto": "UInt16", "preserve_input": ("Byte", "warning"), "smallest": "Byte"},
    ),
    "warp_byte_near": (
        lambda _tmp, dt: gk.raster.warp(_mask_raster(), resampleAlg="near", dtype=dt),
        {"auto": "Byte", "preserve_input": "Byte", "smallest": "Byte"},
    ),
    "warp_byte_average": (
        lambda _tmp, dt: gk.raster.warp(
            _mask_raster(), resampleAlg="average", pixelWidth=400, pixelHeight=400, dtype=dt
        ),
        {"auto": "Float32", "preserve_input": ("Byte", "warning"), "smallest": "Float32"},
    ),
    "warp_byte_bilinear_default": (
        lambda _tmp, dt: gk.raster.warp(_mask_raster(), pixelWidth=400, pixelHeight=400, dtype=dt),
        {"auto": "Float32", "preserve_input": ("Byte", "warning"), "smallest": "Float32"},
    ),
    "warp_int32_average": (
        lambda _tmp, dt: gk.raster.warp(
            _int32_raster(), resampleAlg="average", pixelWidth=200, pixelHeight=200, dtype=dt
        ),
        {"auto": "Float64", "preserve_input": ("Int32", "warning"), "smallest": "Float32"},
    ),
    "warp_byte_sum_16x200": (
        lambda _tmp, dt: gk.raster.warp(
            gdal_raster(np.full((20, 20), 200, np.uint8), gdal.GDT_Byte),
            resampleAlg="sum",
            pixelWidth=400,
            pixelHeight=400,
            dtype=dt,
        ),
        {"auto": "Float64", "preserve_input": ("Byte", "warning"), "smallest": "UInt16"},
    ),
    "mutateRaster_float_processor": (
        lambda _tmp, dt: gk.raster.mutateRaster(
            gdal_raster(np.ones((4, 4), np.uint8), gdal.GDT_Byte), processor=lambda a: a * 0.5, dtype=dt
        ),
        {"auto": "Float64", "preserve_input": ("Byte", "warning"), "smallest": "Float32"},
    ),
    "rasterMosaic_byte_int16": (
        lambda tmp, dt: gk.Extent(0, 0, 800, 400, srs=3035).rasterMosaic(
            [
                gtiff(tmp / "a.tif", np.full((4, 4), 100, np.uint8), gdal.GDT_Byte),
                gtiff(tmp / "b.tif", np.full((4, 4), 50, np.int16), gdal.GDT_Int16, x0=400),
            ],
            _skipFiltering=True,
            dtype=dt,
        ),
        {"auto": "Int16", "preserve_input": "Int16", "smallest": "Byte"},
    ),
    "saveRasterAsTif_float64_integral": (
        lambda tmp, dt: gk.raster.saveRasterAsTif(
            gdal_raster(np.array([[0.0, 1.0], [2.0, 3.0]]), gdal.GDT_Float64), str(tmp / "out.tif"), dtype=dt
        ),
        {"auto": "Float64", "preserve_input": "Float64", "smallest": "Byte"},
    ),
}


def _matrix_params():
    for case in CASES:
        for mode in MODES:
            marks = () if (case, mode) in PASSING_TODAY else (XFAIL,)
            yield pytest.param(case, mode, id=f"{case}-{mode}", marks=marks)


@pytest.mark.parametrize("case, mode", list(_matrix_params()))
def test_mode_matrix(case, mode, tmp_path):
    make, expectations = CASES[case]
    expected = expectations["auto" if mode is None else mode]

    if expected == "error":
        with pytest.raises(GeoKitCDataError):
            make(tmp_path, mode)
        return

    if isinstance(expected, tuple):
        expected_type, _ = expected
        with pytest.warns(UserWarning, match="dtype"):
            out = make(tmp_path, mode)
    else:
        expected_type = expected
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            out = make(tmp_path, mode)

    assert band_type(out) == expected_type


# ----------------------------------------------------------------------------------------------
# explicit dtype next to the modes


@XFAIL
def test_explicit_narrower_float_warns():
    """An explicit dtype narrower than the source is used, with a warning."""
    src = gdal_raster(np.array([[0.1, 0.2]], np.float64), gdal.GDT_Float64)
    with pytest.warns(UserWarning, match="dtype"):
        w = gk.raster.warp(src, dtype="Float32", resampleAlg="near")
    assert band_type(w) == "Float32"


def test_explicit_dtype_with_scalars_that_fit_is_silent():
    """An explicit dtype that holds every scalar is used as is, silently; this holds today and must keep holding."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        r = gk.raster.createRaster(dtype="UInt16", noData=65535, fill=1, **BOUNDS)
    assert band_type(r) == "UInt16"
    assert set(np.unique(M(r))) == {1}


@XFAIL
def test_smallest_keeps_nodata_representable():
    """'smallest' shrinks to a type that still holds noData: values fit Byte, noData=-1 needs Int16, never Int8."""
    r = gk.raster.createRaster(data=np.ones((10, 10), np.int32), noData=-1, dtype="smallest", **BOUNDS)
    assert band_type(r) == "Int16"


@XFAIL
def test_bool_dtype_is_byte():
    """dtype=bool and dtype='bool' give Byte in createRaster and quickRaster."""
    r = gk.raster.createRaster(dtype=bool, **BOUNDS)
    assert band_type(r) == "Byte"
    q = gk.util.quickRaster(bounds=(0, 0, 10, 10), srs=_srs(4326), dx=1, dy=1, dtype="bool")
    assert band_type(q) == "Byte"
