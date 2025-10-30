import pathlib
from typing import Union

import numpy as np
from osgeo import gdal
import osgeo.ogr
from typing import NamedTuple


numeric = Union[int, float, np.number]

load_raster_input = Union[str, pathlib.Path, gdal.Dataset]
load_vector_input = Union[str, pathlib.Path, gdal.Dataset, osgeo.ogr.Layer]

srs_input = Union[gdal.osr.SpatialReference, int, str]


class TransformedPointsXY(NamedTuple):
    x: np.ndarray
    y: np.ndarray


class TransformedPointsXYZ(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
