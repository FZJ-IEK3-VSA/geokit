import os
import pathlib
from os.path import dirname, isdir, isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

from geokit.get_test_data import get_test_data, get_test_shape_file

# Runtime vars
RESULT = "results"
DATA = "data"


def source(s):
    return join(dirname(__file__), DATA, s)


def result(s):
    return join(dirname(__file__), RESULT, s)


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

MULTI_FTR_SHAPE_PATH = get_test_shape_file(
    file_name_without_extension="multiFeature", extension=".shp"
)
BOXES = get_test_shape_file(file_name_without_extension="boxes", extension=".shp")
LUX_SHAPE_PATH = get_test_shape_file(
    file_name_without_extension="LuxShape", extension=".shp"
)
LUX_LINES_PATH = get_test_shape_file(
    file_name_without_extension="LuxLines", extension=".shp"
)

AACHEN_SHAPE_PATH = get_test_shape_file(
    file_name_without_extension="aachenShapefile", extension=".shp"
)

AACHEN_SHAPE_EXTENT = (
    5.974861621856746,
    50.494369506836165,
    6.419306755066032,
    50.95013427734369,
)
AACHEN_SHAPE_EXTENT_3035 = (4035500.0, 3048700.0, 4069500.0, 3101000.0)
AACHEN_ELIGIBILITY_RASTER = get_test_data(file_name="aachen_eligibility.tif")

AACHEN_ZONES = get_test_shape_file(
    file_name_without_extension="aachen_zones", extension=".shp"
)
AACHEN_POINTS = get_test_shape_file(
    file_name_without_extension="aachen_points", extension=".shp"
)

AACHEN_URBAN_LC = get_test_data(file_name="urban_land_cover_aachenClipped.tif")

FJI_SHAPE_PATH = get_test_shape_file(
    file_name_without_extension="FJI", extension=".shp"
)


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

EUR_STATS_FILE = get_test_shape_file(
    file_name_without_extension="Europe_with_H2MobilityData_GermanyClip",
    extension=".shp",
)


CLC_RASTER_PATH = get_test_data(file_name="clc-aachen_clipped.tif")
CLC_FLIPCHECK_PATH = get_test_data(file_name="clc-aachen_clipped-unflipped.tif")

RASTER_GDAL_244 = get_test_data(file_name="raster_gdal_244.tif")


SINGLE_HILL_PATH = get_test_data(file_name="elevation_singleHill.tif")

ELEVATION_PATH = get_test_data(file_name="elevation.tif")
CDDA_PATH = get_test_shape_file(
    file_name_without_extension="CDDA_aachenClipped",
    extension=".shp",
)

NATURA_PATH = get_test_shape_file(
    file_name_without_extension="Natura2000_aachenClipped",
    extension=".shp",
)

DIVIDED_RASTER_1_PATH = get_test_data("divided_raster_1.tif")
DIVIDED_RASTER_2_PATH = get_test_data("divided_raster_2.tif")
DIVIDED_RASTER_3_PATH = get_test_data("divided_raster_3.tif")


## Def a visualizer func
def vis(mat, points=None):
    plt.figure(figsize=(10, 10))
    h = plt.imshow(mat)
    plt.colorbar(h)

    if points:
        plt.plot(points[1], points[0], "o")

    plt.show()
