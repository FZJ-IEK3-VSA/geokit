import pathlib
from typing import Literal, NamedTuple, Union, get_args

import matplotlib.colorbar
import matplotlib.image
import matplotlib.lines
import matplotlib.patches
import numpy as np
import osgeo.ogr
import pandas.core.series
from matplotlib.axes._axes import Axes
from osgeo import gdal, osr
from packaging.version import Version


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


class ptValue(NamedTuple):
    data: numeric | np.ndarray
    xOffset: numeric
    yOffset: numeric
    inBounds: bool | np.bool_


if Version(gdal.__version__) >= Version("3.5"):
    integer_data_types_literal = Literal[
        "GDT_Int8",
        "GDT_Byte",  # Unsigned 8 bit integer
        "GDT_UInt16",
        "GDT_Int16",
        "GDT_UInt32",
        "GDT_Int32",
        "GDT_UInt64",
        "GDT_Int64",
    ]
    _integer_data_types_no_prefix_literal = Literal[
        "Int8",
        "Byte",  # Unsigned 8 bit integer
        "UInt16",
        "Int16",
        "UInt32",
        "Int32",
        "UInt64",
        "Int64",
    ]
    _integer_data_types_no_prefix_lower_literal = Literal[
        "int8",
        "byte",  # Unsigned 8 bit integer
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
    ]

    float_data_types_literal = Literal[
        "GDT_Float32",
        "GDT_Float64",
    ]
    _float_data_types_no_prefix_literal = Literal[
        "Float32",
        "Float64",
    ]

    _float_data_types_no_prefix_lower_literal = Literal[
        "float32",
        "float64",
    ]

    # # Drop use support for complex data types due to rare usage

    numpy_data_types_list_literal = Literal[
        "bool",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float16",
        "float32",
        "float64",
    ]

else:
    integer_data_types_literal = Literal[
        "GDT_Byte",  # Unsigned 8 bit integer
        "GDT_UInt16",
        "GDT_Int16",
        "GDT_UInt32",
        "GDT_Int32",
    ]
    _integer_data_types_no_prefix_literal = Literal[
        "Byte",  # Unsigned 8 bit integer
        "UInt16",
        "Int16",
        "UInt32",
        "Int32",
    ]
    _integer_data_types_no_prefix_lower_literal = Literal[
        "byte",  # Unsigned 8 bit integer
        "uint16",
        "int16",
        "uint32",
        "int32",
    ]

    float_data_types_literal = Literal[
        "GDT_Float32",
        "GDT_Float64",
    ]
    _float_data_types_no_prefix_literal = Literal[
        "Float32",
        "Float64",
    ]

    _float_data_types_no_prefix_lower_literal = Literal[
        "float32",
        "float64",
    ]

    # # Drop use support for complex data types due to rare usage

    numpy_data_types_list_literal = Literal[
        "bool",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float16",
        "float32",
        "float64",
    ]

gdal_c_raster_data_types_literal = Union[
    integer_data_types_literal,
    float_data_types_literal,
]

float_data_types_with_abbreviations_literal = Union[
    float_data_types_literal, _float_data_types_no_prefix_literal, _float_data_types_no_prefix_lower_literal
]

integer_data_types_with_abbreviations_literal = Union[
    integer_data_types_literal, _integer_data_types_no_prefix_literal, _integer_data_types_no_prefix_lower_literal
]
gdal_c_raster_data_types_with_abbreviations_literal = Union[
    gdal_c_raster_data_types_literal,
    integer_data_types_with_abbreviations_literal,
    float_data_types_with_abbreviations_literal,
]

geokit_c_data_types_literal = Union[gdal_c_raster_data_types_with_abbreviations_literal, numpy_data_types_list_literal]


_integer_data_type_list = list(get_args(integer_data_types_literal))
_integer_data_types_no_prefix_list = list(get_args(_integer_data_types_no_prefix_literal))
_integer_data_types_no_prefix_lower_list = list(get_args(_integer_data_types_no_prefix_lower_literal))

_float_data_types_list = list(get_args(float_data_types_literal))
_float_data_types_no_prefix_list = list(get_args(_float_data_types_no_prefix_literal))
_float_data_types_no_prefix_lower_list = list(get_args(_float_data_types_no_prefix_lower_literal))


_gdal_c_raster_data_types_list = [*_integer_data_type_list, *_float_data_types_list]
_gdal_c_raster_data_types_with_abbreviations_list = [
    *_integer_data_type_list,
    *_integer_data_types_no_prefix_list,
    *_integer_data_types_no_prefix_lower_list,
    *_float_data_types_list,
    *_float_data_types_no_prefix_list,
    *_float_data_types_no_prefix_lower_list,
]
gdal_abbreviation_mapper_dict = {}
for rigorous_gdal_type, no_prefix_type, no_prefix_lower_type in zip(
    _integer_data_type_list, _integer_data_types_no_prefix_list, _integer_data_types_no_prefix_lower_list
):
    gdal_abbreviation_mapper_dict[no_prefix_type] = rigorous_gdal_type
    gdal_abbreviation_mapper_dict[no_prefix_lower_type] = rigorous_gdal_type

for rigorous_gdal_type, no_prefix_type, no_prefix_lower_type in zip(
    _float_data_types_list, _float_data_types_no_prefix_list, _float_data_types_no_prefix_lower_list
):
    gdal_abbreviation_mapper_dict[no_prefix_type] = rigorous_gdal_type
    gdal_abbreviation_mapper_dict[no_prefix_lower_type] = rigorous_gdal_type
