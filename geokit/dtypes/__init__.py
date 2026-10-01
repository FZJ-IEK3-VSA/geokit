"""Data types of rasters and vector fields.

Inside GeoKit a type is always a :class:`numpy.dtype`. This package converts between that and the types of
GDAL (pixel types), OGR (field types) and pandas at the edges (``conversions``), chooses the output type of an
operation (``resolve``: the ``dtype`` modes ``"auto"``, ``"preserve_input"`` and ``"smallest"`` and explicit
types), shrinks a finished output to the smallest lossless type (``smallest``), and holds the option that turns
the data-type warnings off (``options``). Everything a caller needs is importable from ``geokit.dtypes``.

The decisions behind this package are the architecture decision records in the documentation, section
*Explanation / Data Types* (``docs/explanation/data_types``).
"""

from geokit.dtypes.conversions import (
    DTYPE_MODES,
    can_hold,
    dtype_for_value,
    from_band,
    from_ogr_field,
    gdal_type_name,
    promote_dtypes,
    to_dtype,
    to_gdal,
    to_ogr_field,
)
from geokit.dtypes.options import Options, get_options, issue_warning, options, set_options
from geokit.dtypes.resolve import DTYPE_PARAMETER_DOCSTRING, DtypeRule, ResolvedDtype, dtype_mode, resolve_dtype
from geokit.dtypes.smallest import shrink_dataset, smallest_dtype_for_array, smallest_dtype_for_dataset
from geokit.error import GeoKitDataTypeError, GeoKitDataTypeWarning

__all__ = [
    "DTYPE_MODES",
    "DTYPE_PARAMETER_DOCSTRING",
    "DtypeRule",
    "GeoKitDataTypeError",
    "GeoKitDataTypeWarning",
    "Options",
    "ResolvedDtype",
    "can_hold",
    "dtype_for_value",
    "dtype_mode",
    "from_band",
    "from_ogr_field",
    "gdal_type_name",
    "get_options",
    "issue_warning",
    "options",
    "promote_dtypes",
    "resolve_dtype",
    "set_options",
    "shrink_dataset",
    "smallest_dtype_for_array",
    "smallest_dtype_for_dataset",
    "to_dtype",
    "to_gdal",
    "to_ogr_field",
]
