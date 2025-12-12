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
# gdal_raster_data_types = Union[
#     Literal[
#         "GDT_Byte",
#         "GDT_Int32",
#         "GDT_Int64",
#         "GDT_Float32",
#         "GDT_Float64",
#         "Byte",
#         "Int8",
#         "UInt16",
#         "Int16",
#         "UInt32",
#         "Int32",
#         "UInt64",
#         "Int64",
#         "Float16",
#         "Float32",
#         "Float64",
#         "CInt16",
#         "CInt32",
#         "CFloat16",
#         "CFloat32",
#         "CFloat64",
#         "NoneType",
#         # Additional ambiguous types that gets translated to the above types
#         "float",
#         "int",
#     ],
#     # The above data types can also be represented as strings
#     # The string representation can be obtained from the integer
#     # representation using the gdal.GetDataTypeName() method.
#     int,
# ]


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
    # Minimum and maximum value of the raster data
    minimum_value: numeric | None
    maximum_value: numeric | None


# vecInfo = namedtuple("vecInfo", "srs bounds xMin yMin xMax yMax count attributes source")
class vecInfo(NamedTuple):
    srs: osr.SpatialReference | None
    bounds: tuple[numeric, numeric, numeric, numeric]
    xMin: numeric
    yMin: numeric
    xMax: numeric
    yMax: numeric
    count: int
    attributes: list[str]
    source: str
    attribute_data_types_constant: dict[str, int]
    attribute_data_types_str: dict[str, str]


_integer_data_type_list = [
    "GDT_Int8",
    "GDT_Byte",  # Unsigned 8 bit integer
    "GDT_UInt16",
    "GDT_Int16",
    "GDT_UInt32",
    "GDT_Int32",
    "GDT_UInt64",
    "GDT_Int64",
]
integer_data_types_literal = Literal[*_integer_data_type_list]

_float_data_types_list = [
    "GDT_Float32",
    "GDT_Float64",
]
float_data_types_literal = Literal[*_float_data_types_list]


# # Drop use support for complex data types due to rare usage
# complex_integer_data_types_list = [
#     "GDT_CInt16",
#     "GDT_CInt32",
# ]
# complex_integer_data_types = Literal[*complex_integer_data_types_list]

# complex_float_data_types_list = [
#     "GDT_CFloat32",
#     "GDT_CFloat64",
# ]
# complex_float_data_types = Literal[*complex_float_data_types_list]


_gdal_c_raster_data_types_list = [
    *_integer_data_type_list,
    *_float_data_types_list,
    # *complex_integer_data_types,
    # *complex_float_data_types,
]
gdal_c_raster_data_types_literal = Literal[*_gdal_c_raster_data_types_list]

numpy_data_types_list_literal = Literal[
    "bool", "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float16", "float32", "float64"
]

_gdal_c_raster_data_types_abbreviations_list = []
gdal_abbreviation_mapper_dict = {}
for current_type in _gdal_c_raster_data_types_list:
    abbreviation = current_type.replace("GDT_", "")
    abbreviation_lower = abbreviation.lower()
    _gdal_c_raster_data_types_abbreviations_list.append(abbreviation)
    _gdal_c_raster_data_types_abbreviations_list.append(abbreviation_lower)
    gdal_abbreviation_mapper_dict[abbreviation] = current_type
    gdal_abbreviation_mapper_dict[abbreviation_lower] = current_type

gdal_c_raster_data_types_with_abbreviations_literal = Literal[
    *_gdal_c_raster_data_types_list, *_gdal_c_raster_data_types_abbreviations_list
]

geokit_c_data_types_literal = Union[gdal_c_raster_data_types_with_abbreviations_literal, numpy_data_types_list_literal]


class ptValue(NamedTuple):
    data: numeric | np.ndarray
    xOffset: numeric
    yOffset: numeric
    inBounds: bool | np.bool
