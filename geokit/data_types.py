import pathlib
from typing import Literal, NamedTuple, Union

import matplotlib.colorbar
import matplotlib.image
import matplotlib.lines
import matplotlib.patches
import numpy as np
import osgeo.ogr
import pandas.core.series
from matplotlib.axes._axes import Axes
from osgeo import gdal, osr


class AxHands(NamedTuple):
    ax: Axes
    handles: (
        pandas.core.series.Series
        | matplotlib.image.AxesImage
        | list[matplotlib.patches.PathPatch | matplotlib.lines.Line2D]
        | list[list[matplotlib.lines.Line2D | matplotlib.patches.PathPatch]]
    )
    cbar: matplotlib.colorbar.Colorbar | None


numeric = Union[int, float, np.number]

load_raster_input = Union[str, pathlib.Path, gdal.Dataset]
load_vector_input = Union[str, pathlib.Path, gdal.Dataset, osgeo.ogr.Layer]

srs_input = Union[gdal.osr.SpatialReference, int, str]

# supported gdal raster data types can be found here:
# https://gdal.org/en/stable/user/raster_data_model.html
# The gdt data types are not listed in the documentation
# but still work.
gdal_raster_data_types = Union[
    Literal[
        "GDT_Byte",
        "GDT_Int32",
        "GDT_Int64",
        "GDT_Float32",
        "GDT_Float64",
        "Byte",
        "Int8",
        "UInt16",
        "Int16",
        "UInt32",
        "Int32",
        "UInt64",
        "Int64",
        "Float16",
        "Float32",
        "Float64",
        "CInt16",
        "CInt32",
        "CFloat16",
        "CFloat32",
        "CFloat64",
        "NoneType",
        # Additional ambigous types that gets translated to the above types
        "float",
        "int",
    ],
    # The above data types can also be represented as strings
    # The string representation can be obtained from the integer
    # representation using the gdal.GetDataTypeName() method.
    int,
]


class TransformedPointsXY(NamedTuple):
    x: np.ndarray
    y: np.ndarray


class TransformedPointsXYZ(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


class RasterInfo(NamedTuple):
    srs: osr.SpatialReference | None
    dtype: int  # returns an integer representing the datatype
    flipY: bool
    yAtTop: bool
    bounds: tuple[numeric, numeric, numeric, numeric]
    xMin: numeric
    yMin: numeric
    xMax: numeric
    yMax: numeric
    dx: numeric
    dy: numeric
    pixelWidth: numeric
    pixelHeight: numeric
    noData: numeric | None
    xWinSize: numeric
    yWinSize: numeric
    meta: dict
    source: str
    scale: numeric | None
    offset: numeric | None
    # data_type_name_str returns a string representation of the
    # pixels of the raster. It is derived from the integer
    # data type representation stored as dtype
    data_type_name_str: str


class ptValue(NamedTuple):
    data: numeric | np.ndarray
    xOffset: numeric
    yOffset: numeric
    inBounds: bool | np.bool_
