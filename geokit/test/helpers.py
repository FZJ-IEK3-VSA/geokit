import numpy as np
from os.path import join, dirname, isfile, isdir
from osgeo import ogr, gdal, osr
import matplotlib.pyplot as plt
import os
import pandas as pd


# Runtime vars
RESULT = "results"
DATA  = "data"
def source(s): return join(dirname(__file__), DATA, s)
def result(s): return join(dirname(__file__), RESULT, s)

### make working items
EPSG4326 = osr.SpatialReference()
EPSG4326.ImportFromEPSG(4326)

EPSG3035 = osr.SpatialReference()
EPSG3035.ImportFromEPSG(3035)


pointsInAachen4326 = [(6.06590,50.51939), (6.02141,50.61491), (6.371634,50.846025)]
pointsInAachen3035 = [(4042131.15813425, 3052769.53854268),(4039553.19006358, 3063551.94787756),(4065568.41552701, 3087947.74365965),]
pointInAachen3035 = (4061794.7,3094718.4)


POLY = "POLYGON ((10.1 32, 10.9 35.1, 12 36, 14.6 38.1, 13.5 35, 12.9 35.1, 11.1 33, 10.6 32.2, 10.5 30.5, 10.1 32))"
SUB_POLY1 = "POLYGON ((7 49.7, 7 49.9, 7.4 49.75, 7 49.7))"
SUB_POLY2 = "POLYGON ((8 49.7, 8 49.9, 8.4 49.75, 8 49.7))"
SUB_POLY3 = "POLYGON ((9 49.7, 9 49.9, 9.4 49.75, 9 49.7))"
POINT_SET = [ "POINT (7 49.7)", "POINT (7 49.9)", "POINT (7.4 49.75)", "POINT (7 49.7)","POINT (8 49.7)", 
              "POINT (8 49.9)", "POINT (8.4 49.75)", "POINT (8 49.7)","POINT (9 49.7)", "POINT (9 49.9)", 
              "POINT (9.4 49.75)", "POINT (9 49.7)"]

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

MULTI_FTR_SHAPE_PATH = source("multiFeature.shp")
BOXES = source("boxes.shp")
LUX_SHAPE_PATH = source("LuxShape.shp")
LUX_LINES_PATH = source("LuxLines.shp")

AACHEN_SHAPE_PATH = source("aachenShapefile.shp")
AACHEN_SHAPE_EXTENT = (5.974861621856746, 50.494369506836165, 6.419306755066032, 50.95013427734369)
AACHEN_SHAPE_EXTENT_3035 = (4035500.0, 3048700.0, 4069500.0, 3101000.0)
AACHEN_ELIGIBILITY_RASTER = source("aachen_eligibility.tif")
AACHEN_ZONES = source("aachen_zones.shp")
AACHEN_POINTS = source("aachen_points.shp")
AACHEN_URBAN_LC = source("urban_land_cover_aachenClipped.tif")

NUMPY_FLOAT_ARRAY = np.arange(10, dtype="float")

ELIGIBILITY_DATA = np.zeros((100,100))
ELIGIBILITY_DATA[:, 0:40] = 1.0
ELIGIBILITY_DATA[:,10:30] = 0.25
ELIGIBILITY_DATA[:,50:70] = 0.5
ELIGIBILITY_DATA[:,80:  ] = 0.75

MASK_DATA = np.zeros((100,100), dtype="bool")
MASK_DATA[range(100),range(100)] = True
MASK_DATA[20:23,:] = True
MASK_DATA[50:53,::20] = True
for x,y in zip(np.arange(100), 10*np.sin( np.pi*np.arange(100)/20 )):
    _x = np.round(x).astype("int")
    _y = np.round(y).astype("int")
    MASK_DATA[_y+75:_y+77,_x] = True

EUR_STATS_FILE = source("Europe_with_H2MobilityData_GermanyClip.shp")

CLC_RASTER_PATH = source("clc-aachen_clipped.tif")
CLC_FLIPCHECK_PATH = source("clc-aachen_clipped-unflipped.tif")

SINGLE_HILL_PATH = source("elevation_singleHill.tif")
ELEVATION_PATH = source("elevation.tif")
CDDA_PATH = source("CDDA_aachenClipped.shp")
NATURA_PATH = source("Natura2000_aachenClipped.shp")

## Def a visualizer func
def vis(mat, points=None):
    plt.figure(figsize=(10,10))
    h = plt.imshow(mat)
    plt.colorbar(h)

    if(points):
        plt.plot(points[1], points[0], 'o')

    plt.show()
