import os
import pathlib
from collections import OrderedDict as _OrderedDict
from os.path import dirname, isdir, isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from geokit.get_test_data import get_test_data

# Runtime vars
RESULT = "results"
DATA = "data"


def result(s):
    return join(dirname(__file__), RESULT, s)


# Committed golden raster results (checked into the repo for cross-version regression testing).
RASTER_RESULT = join(dirname(dirname(__file__)), "geokit", "data", "raster_results")


def raster_result(s):
    """Path to a committed golden raster result: geokit/data/raster_results/<s>."""
    return join(RASTER_RESULT, s)


### make working items
EPSG4326 = osr.SpatialReference()
if gdal.__version__ >= "3.0.0":
    EPSG4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
EPSG4326.ImportFromEPSG(4326)

EPSG3035 = osr.SpatialReference()
if gdal.__version__ >= "3.0.0":
    EPSG3035.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
EPSG3035.ImportFromEPSG(3035)


pointsInAachen4326 = [(6.06590, 50.51939), (6.02141, 50.61491), (6.371634, 50.846025)]
pointsInAachen3035 = [
    (4042131.15813425, 3052769.53854268),
    (4039553.19006358, 3063551.94787756),
    (4065568.41552701, 3087947.74365965),
]
pointInAachen3035 = (4061794.7, 3094718.4)
pointInAccraEcowasM = (535733.2772457143, -1508102.6643880515)
pointInWindhoekSadcM = (-5255934.573657837, 3239007.1380184838)

POLY = "POLYGON ((10.1 32, 10.9 35.1, 12 36, 14.6 38.1, 13.5 35, 12.9 35.1, 11.1 33, 10.6 32.2, 10.5 30.5, 10.1 32))"
SUB_POLY1 = "POLYGON ((7 49.7, 7 49.9, 7.4 49.75, 7 49.7))"
SUB_POLY2 = "POLYGON ((8 49.7, 8 49.9, 8.4 49.75, 8 49.7))"
SUB_POLY3 = "POLYGON ((9 49.7, 9 49.9, 9.4 49.75, 9 49.7))"
POINT_SET = [
    "POINT (7 49.7)",
    "POINT (7 49.9)",
    "POINT (7.4 49.75)",
    "POINT (7 49.7)",
    "POINT (8 49.7)",
    "POINT (8 49.9)",
    "POINT (8.4 49.75)",
    "POINT (8 49.7)",
    "POINT (9 49.7)",
    "POINT (9 49.9)",
    "POINT (9.4 49.75)",
    "POINT (9 49.7)",
]

GEOM = ogr.CreateGeometryFromWkt(POLY)
GEOM.AssignSpatialReference(EPSG4326)

SUB_GEOM = ogr.CreateGeometryFromWkt(SUB_POLY1)
SUB_GEOM.AssignSpatialReference(EPSG4326)

SUB_GEOM2 = ogr.CreateGeometryFromWkt(SUB_POLY2)
SUB_GEOM2.AssignSpatialReference(EPSG4326)

SUB_GEOM3 = ogr.CreateGeometryFromWkt(SUB_POLY3)
SUB_GEOM3.AssignSpatialReference(EPSG4326)

SUB_GEOMS = [SUB_GEOM, SUB_GEOM2, SUB_GEOM3]

GEOM_3035 = SUB_GEOM.Clone()
GEOM_3035.TransformTo(EPSG3035)

MULTI_FTR_SHAPE_PATH = get_test_data(file_name="multiFeature.shp")
BOXES = get_test_data(
    file_name="boxes.shp",
)
LUX_SHAPE_PATH = get_test_data(file_name="LuxShape.shp")
LUX_LINES_PATH = get_test_data(file_name="LuxLines.shp")

AACHEN_SHAPE_PATH = get_test_data(file_name="aachenShapefile.shp")

AACHEN_SHAPE_EXTENT = (
    5.974861621856746,
    50.494369506836165,
    6.419306755066032,
    50.95013427734369,
)
AACHEN_SHAPE_EXTENT_3035 = (4035500.0, 3048700.0, 4069500.0, 3101000.0)
AACHEN_ELIGIBILITY_RASTER = get_test_data(file_name="aachen_eligibility.tif")

AACHEN_ZONES = get_test_data(
    file_name="aachen_zones.shp",
)
AACHEN_POINTS = get_test_data(file_name="aachen_points.shp")

AACHEN_URBAN_LC = get_test_data(file_name="urban_land_cover_aachenClipped.tif")

FJI_SHAPE_PATH = get_test_data(file_name="FJI.shp")


NUMPY_FLOAT_ARRAY = np.arange(10, dtype="float")

ELIGIBILITY_DATA = np.zeros((100, 100))
ELIGIBILITY_DATA[:, 0:40] = 1.0
ELIGIBILITY_DATA[:, 10:30] = 0.25
ELIGIBILITY_DATA[:, 50:70] = 0.5
ELIGIBILITY_DATA[:, 80:] = 0.75

MASK_DATA = np.zeros((100, 100), dtype="bool")
MASK_DATA[range(100), range(100)] = True
MASK_DATA[20:23, :] = True
MASK_DATA[50:53, ::20] = True
for x, y in zip(np.arange(100), 10 * np.sin(np.pi * np.arange(100) / 20)):
    _x = np.round(x).astype("int")
    _y = np.round(y).astype("int")
    MASK_DATA[_y + 75 : _y + 77, _x] = True

EUR_STATS_FILE = get_test_data(
    file_name="Europe_with_H2MobilityData_GermanyClip.shp",
)


CLC_RASTER_PATH = get_test_data(file_name="clc-aachen_clipped.tif")
CLC_FLIPCHECK_PATH = get_test_data(file_name="clc-aachen_clipped-unflipped.tif")

RASTER_GDAL_244 = get_test_data(file_name="raster_gdal_244.tif")


SINGLE_HILL_PATH = get_test_data(file_name="elevation_singleHill.tif")

ELEVATION_PATH = get_test_data(file_name="elevation.tif")
CDDA_PATH = get_test_data(
    file_name="CDDA_aachenClipped.shp",
)

NATURA_PATH = get_test_data(
    file_name="Natura2000_aachenClipped.shp",
)

DIVIDED_RASTER_1_PATH = get_test_data(file_name="divided_raster_1.tif")
DIVIDED_RASTER_2_PATH = get_test_data(file_name="divided_raster_2.tif")
DIVIDED_RASTER_3_PATH = get_test_data(file_name="divided_raster_3.tif")


## Def a visualizer func
def vis(mat, points=None):
    plt.figure(figsize=(10, 10))
    h = plt.imshow(mat)
    plt.colorbar(h)

    if points:
        plt.plot(points[1], points[0], "o")

    plt.show()


def assert_raster_equal(a, b, *, atol=0):
    """Assert two rasters describe the same grid and hold the same values.

    `a` and `b` may each be a file path or a gdal.Dataset. Compares the full grid definition
    (bounds, pixel size, SRS, noData, dtype) and the pixel matrix exactly (atol=0 by default).
    Used to prove the in-memory and on-disk warp paths produce byte-identical output.
    """
    import math

    import numpy as _np

    from geokit import raster as _raster

    ia = _raster.rasterInfo(a)
    ib = _raster.rasterInfo(b)

    assert _np.allclose(ia.bounds, ib.bounds, atol=0), f"bounds differ: {ia.bounds} vs {ib.bounds}"
    assert (ia.dx, ia.dy) == (ib.dx, ib.dy), f"pixel size differs: {(ia.dx, ia.dy)} vs {(ib.dx, ib.dy)}"
    assert ia.srs.IsSame(ib.srs), "SRS differs"
    assert ia.dtype == ib.dtype, f"dtype differs: {ia.dtype} vs {ib.dtype}"

    na, nb = ia.noData, ib.noData
    nan_a = isinstance(na, float) and math.isnan(na)
    nan_b = isinstance(nb, float) and math.isnan(nb)
    assert (na is None and nb is None) or (nan_a and nan_b) or na == nb, f"noData differs: {na} vs {nb}"

    ma = _raster.extractMatrix(a)
    mb = _raster.extractMatrix(b)
    assert ma.shape == mb.shape, f"matrix shape differs: {ma.shape} vs {mb.shape}"
    assert _np.allclose(ma, mb, atol=atol, equal_nan=True), "pixel values differ"


def make_resampling_test_raster(pixel=100, srs=None):
    """Build a deterministic source raster whose features make resampling algorithms differ.

    Contains a smooth diagonal ramp, a constant block, a sharp step edge, two discrete class
    patches and a single bright spike -- enough that near / bilinear / cubic / lanczos / mode /
    min / max / median / quartile / sum each produce a distinguishable result. Returns an in-memory
    gdal.Dataset on a 32x32 grid with bounds (0, 0, 32*pixel, 32*pixel). Shared by the
    golden-regression tests and the test_case_inspector notebook so both exercise the same input.
    """
    import numpy as _np

    from geokit import raster as _raster

    n = 32
    rows, cols = _np.meshgrid(_np.arange(n), _np.arange(n), indexing="ij")
    data = (rows + cols).astype(_np.float32)  # smooth diagonal ramp (0..62)
    data[2:10, 2:10] = 10.0  # constant block
    data[12:20, :16] = 5.0  # step edge: low ...
    data[12:20, 16:] = 90.0  # ... to high at column 16
    data[22:26, 2:10] = 120.0  # discrete class patch A
    data[22:26, 18:26] = 210.0  # discrete class patch B
    data[29, 29] = 300.0  # single bright spike

    return _raster.createRaster(
        bounds=(0, 0, n * pixel, n * pixel),
        data=data,
        pixelWidth=pixel,
        pixelHeight=pixel,
        srs=EPSG3035 if srs is None else srs,
    )
