"""Conversions at the edges of GeoKit: GDAL pixel types, OGR field types and pandas types to ``numpy.dtype``.

Inside GeoKit a type is always a ``numpy.dtype`` (ADR 7). This module holds the only mappings to and from the
other vocabularies, and the scalar helpers that decide which type stores a value exactly (ADR 4, ADR 5).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from osgeo import gdal, ogr

from geokit.error import GeoKitDataTypeError

__all__ = [
    "DTYPES_EXACT_IN_FLOAT32",
    "MODES",
    "can_hold",
    "describe_range",
    "dtype_for_integer_range",
    "dtype_for_value",
    "from_band",
    "from_ogr_field",
    "gdal_type_name",
    "is_number",
    "is_whole_number",
    "promote_dtypes",
    "to_dtype",
    "to_gdal",
    "to_ogr_field",
]

MODES = ("auto", "preserve_input", "smallest")
"""The strings that select a mode of the ``dtype`` parameter. They are modes, not types, so ``to_dtype`` rejects them."""


# ----------------------------------------------------------------------------------------------
# Pixel types: GDAL <-> NumPy

_GDAL_TYPE_NAME_BY_NUMPY_NAME = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
}
_GDAL_VERSION_THAT_INTRODUCED = {"Int8": "3.7", "UInt64": "3.5", "Int64": "3.5"}
_COMPLEX_GDAL_TYPE_NAMES = {"cint16", "cint32", "cfloat32", "cfloat64"}


def _gdal_constants_of_the_installed_gdal() -> dict[np.dtype, int]:
    constants_by_dtype = {}
    for numpy_name, gdal_name in _GDAL_TYPE_NAME_BY_NUMPY_NAME.items():
        constant = getattr(gdal, f"GDT_{gdal_name}", None)
        if constant is not None:
            constants_by_dtype[np.dtype(numpy_name)] = constant
    return constants_by_dtype


_GDAL_CONSTANT_BY_DTYPE = _gdal_constants_of_the_installed_gdal()
_DTYPE_BY_GDAL_CONSTANT = {constant: dtype for dtype, constant in _GDAL_CONSTANT_BY_DTYPE.items()}
_NUMPY_NAME_BY_LOWER_GDAL_NAME = {
    gdal_name.lower(): numpy_name for numpy_name, gdal_name in _GDAL_TYPE_NAME_BY_NUMPY_NAME.items()
}


def to_dtype(dtype) -> np.dtype | None:
    """Convert any accepted spelling of a type to a ``numpy.dtype``.

    Accepted are GDAL names with or without the ``GDT_`` prefix in any case (``"Byte"``, ``"GDT_Byte"``,
    ``"byte"``), NumPy names and types (``"uint8"``, ``np.uint8``, ``np.dtype("uint8")``) and the Python types
    ``bool``, ``int`` and ``float``. ``bool`` becomes ``uint8`` and ``float16`` becomes ``float32``, because
    GDAL has no pixel type for either. ``None`` stays ``None``.

    Raises
    ------
    GeoKitDataTypeError
        For a bare integer such as ``gdal.GDT_Float32`` (GDAL and OGR constants share the same integers), a
        mode string, a complex, string or object type, or a type that the installed GDAL does not support.
    """
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return _dtype_from_name(dtype)
    if isinstance(dtype, np.dtype):
        return _supported_dtype(dtype, spelling=repr(dtype))
    if isinstance(dtype, (bool, int, float, np.generic)):
        raise _bare_value_error(dtype)
    if dtype is bool:
        return np.dtype(np.uint8)
    if isinstance(dtype, type):
        try:
            numpy_dtype = np.dtype(dtype)
        except TypeError:
            raise GeoKitDataTypeError(f"dtype={dtype!r} is not a type that a raster band can store.") from None
        return _supported_dtype(numpy_dtype, spelling=dtype.__name__)
    raise GeoKitDataTypeError(
        f'dtype={dtype!r} is not a type. Pass a GDAL name such as "Byte", a NumPy dtype, or one of the modes '
        f"{', '.join(repr(mode) for mode in MODES)}."
    )


def _dtype_from_name(name: str) -> np.dtype:
    lowered = name.strip().lower()
    if lowered in MODES:
        raise GeoKitDataTypeError(
            f'dtype="{name}" is a mode, not a type. It is valid only for functions that choose the type '
            f"themselves, not for a helper that takes a final type."
        )
    if lowered.startswith("gdt_"):
        lowered = lowered[len("gdt_") :]
    if lowered in _COMPLEX_GDAL_TYPE_NAMES or lowered.startswith("complex"):
        raise GeoKitDataTypeError(f'dtype="{name}": complex types are not supported.')
    if lowered == "boolean":
        lowered = "bool"
    # GDAL names come first: "byte" is GDAL's unsigned 8-bit type, while NumPy's "byte" means int8
    numpy_name = _NUMPY_NAME_BY_LOWER_GDAL_NAME.get(lowered, lowered)
    try:
        numpy_dtype = np.dtype(numpy_name)
    except TypeError:
        raise GeoKitDataTypeError(
            f'dtype="{name}" is not a known type. Use a GDAL name such as "Byte", "Int16" or "Float32", '
            f'or a NumPy name such as "uint8".'
        ) from None
    return _supported_dtype(numpy_dtype, spelling=f'"{name}"')


def _supported_dtype(numpy_dtype: np.dtype, spelling: str) -> np.dtype:
    """Normalise a NumPy dtype to one that a GDAL band of the installed GDAL can store."""
    if numpy_dtype.kind == "b":  # "b" = boolean
        return np.dtype(np.uint8)
    if numpy_dtype == np.float16:
        return np.dtype(np.float32)
    if numpy_dtype.kind == "c":  # "c" = complex floating point
        raise GeoKitDataTypeError(f"dtype={spelling}: complex types are not supported.")
    if numpy_dtype.kind not in "iuf":  # "i"/"u"/"f" = signed/unsigned integer, floating point
        raise GeoKitDataTypeError(
            f"dtype={spelling} cannot be stored in a raster band. GeoKit supports integer and float types."
        )
    native_dtype = np.dtype(numpy_dtype.name)  # drops a non-native byte order
    if native_dtype in _GDAL_CONSTANT_BY_DTYPE:
        return native_dtype
    gdal_name = _GDAL_TYPE_NAME_BY_NUMPY_NAME.get(native_dtype.name)
    if gdal_name is None:
        raise GeoKitDataTypeError(f"dtype={spelling} ({native_dtype}) has no GDAL pixel type.")
    first_version = _GDAL_VERSION_THAT_INTRODUCED.get(gdal_name, "?")
    raise GeoKitDataTypeError(
        f"dtype={spelling} needs GDAL {first_version} or later (GDT_{gdal_name}); installed is GDAL {gdal.__version__}."
    )


def _bare_value_error(value) -> GeoKitDataTypeError:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return GeoKitDataTypeError(f"dtype={value!r} is a value, not a type.")
    message = (
        f"dtype={value!r} is a bare integer. GDAL and OGR constants share the same integers, so GeoKit does "
        f"not accept them."
    )
    if 0 < int(value) < gdal.GDT_TypeCount:
        gdal_name = gdal.GetDataTypeName(int(value))
        message += f' If you meant gdal.GDT_{gdal_name}, pass dtype="{gdal_name}".'
    return GeoKitDataTypeError(message)


def to_gdal(dtype) -> int:
    """Return the GDAL pixel type constant (``gdal.GDT_*``) of a type in any accepted spelling."""
    numpy_dtype = to_dtype(dtype)
    if numpy_dtype is None:
        raise GeoKitDataTypeError("to_gdal needs a type, not None.")
    return _GDAL_CONSTANT_BY_DTYPE[numpy_dtype]


def gdal_type_name(dtype) -> str:
    """Return the GDAL name of a type, such as ``"UInt16"``, for messages."""
    return gdal.GetDataTypeName(to_gdal(dtype))


def from_band(band: gdal.Band | gdal.Dataset) -> np.dtype:
    """Return the pixel type of a GDAL band (or of the first band of a dataset) as a ``numpy.dtype``."""
    if isinstance(band, gdal.Dataset):
        band = band.GetRasterBand(1)
    constant = band.DataType
    numpy_dtype = _DTYPE_BY_GDAL_CONSTANT.get(constant)
    if numpy_dtype is None:
        raise GeoKitDataTypeError(
            f"The band has the pixel type {gdal.GetDataTypeName(constant)}, which GeoKit does not support."
        )
    return numpy_dtype


# ----------------------------------------------------------------------------------------------
# Field types: OGR <-> NumPy


def from_ogr_field(field_definition: ogr.FieldDefn) -> np.dtype | None:
    """Return the ``numpy.dtype`` of an OGR field definition, or ``None`` for a non-numeric field.

    ``Integer`` fields give ``int32`` (``int16`` with the ``Int16`` subtype, ``uint8`` with the ``Boolean``
    subtype), ``Integer64`` fields ``int64``, and ``Real`` fields ``float64`` (``float32`` with the ``Float32``
    subtype). The function takes the field definition object, not an integer constant, so that OGR and GDAL
    constants cannot be mixed up.
    """
    if not isinstance(field_definition, ogr.FieldDefn):
        raise GeoKitDataTypeError(
            f"from_ogr_field expects an ogr.FieldDefn, not {field_definition!r}. An integer could be an OGR or "
            f"a GDAL constant."
        )
    field_type = field_definition.GetType()
    subtype = field_definition.GetSubType()
    if field_type == ogr.OFTInteger:
        if subtype == ogr.OFSTBoolean:
            return np.dtype(np.uint8)
        if subtype == ogr.OFSTInt16:
            return np.dtype(np.int16)
        return np.dtype(np.int32)
    if field_type == ogr.OFTInteger64:
        return np.dtype(np.int64)
    if field_type == ogr.OFTReal:
        if subtype == ogr.OFSTFloat32:
            return np.dtype(np.float32)
        return np.dtype(np.float64)
    return None


def to_ogr_field(dtype) -> int:
    """Return the OGR field type constant (``ogr.OFT*``) for a NumPy, pandas or Python type.

    Integers up to 32 bits and ``bool`` give ``Integer``; ``uint32``, ``int64`` and ``uint64`` give
    ``Integer64`` (values above 2**63 - 1 cannot be stored and are checked when they are written); floats give
    ``Real``; strings, objects and dates give ``String``.
    """
    if dtype is str:
        return ogr.OFTString
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        # pandas' nullable Int64/Float64/boolean dtypes know their NumPy counterpart; string dtypes do not
        numpy_dtype = getattr(dtype, "numpy_dtype", None)
        if numpy_dtype is None:
            return ogr.OFTString
    else:
        try:
            numpy_dtype = np.dtype(dtype)
        except TypeError:
            raise GeoKitDataTypeError(f"{dtype!r} is not a type that an OGR field can store.") from None
    if numpy_dtype.kind == "b":  # "b" = boolean
        return ogr.OFTInteger
    if numpy_dtype.kind in "iu":  # "i"/"u" = signed/unsigned integer
        is_wider_than_int32 = numpy_dtype.itemsize > 4 or numpy_dtype == np.uint32
        if is_wider_than_int32:
            return ogr.OFTInteger64
        return ogr.OFTInteger
    if numpy_dtype.kind == "f":  # "f" = floating point
        return ogr.OFTReal
    if numpy_dtype.kind == "c":  # "c" = complex floating point
        raise GeoKitDataTypeError("Complex values cannot be stored in an OGR field.")
    return ogr.OFTString


# ----------------------------------------------------------------------------------------------
# Scalars and promotion

_INTEGER_DTYPES_IN_ORDER_OF_CHOICE = [
    np.dtype(np.uint8),
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.uint16),
    np.dtype(np.int32),
    np.dtype(np.uint32),
    np.dtype(np.int64),
    np.dtype(np.uint64),
]
"""Where GeoKit chooses an integer width: ``Byte`` before ``Int8``, then the signed type before the unsigned one."""

_LARGEST_EXACT_WHOLE_NUMBER_BY_FLOAT_DTYPE = {
    np.dtype(np.float32): 2**24,
    np.dtype(np.float64): 2**53,
}

DTYPES_EXACT_IN_FLOAT32 = frozenset(
    {
        np.dtype(np.uint8),
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
        np.dtype(np.float32),
    }
)
"""The types whose every value is exact in ``float32``."""


def is_number(value) -> bool:
    """Return whether ``value`` is a Python or NumPy number (``bool`` counts as a number)."""
    return isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))


def is_whole_number(value) -> bool:
    """Return whether ``value`` is an integer, or a float without a fractional part such as ``3.0``."""
    if isinstance(value, (bool, int, np.bool_, np.integer)):
        return True
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return False
    return as_float.is_integer()


def can_hold(dtype, value) -> bool:
    """Return whether ``value`` can be stored exactly in ``dtype``.

    Whole numbers (also whole floats such as ``3.0``) fit an integer type within its range and a float type up
    to the largest whole number it stores exactly (2**24 for ``float32``, 2**53 for ``float64``). NaN and
    infinity fit every float type. A fractional value fits ``float32`` only if the round trip through
    ``float32`` is exact.
    """
    numpy_dtype = to_dtype(dtype)
    if numpy_dtype is None or not is_number(value):
        return False
    if is_whole_number(value):
        whole_number = int(value)
        if numpy_dtype.kind in "iu":  # "i" signed / "u" unsigned integer kinds
            limits = np.iinfo(numpy_dtype)
            return limits.min <= whole_number <= limits.max
        return abs(whole_number) <= _LARGEST_EXACT_WHOLE_NUMBER_BY_FLOAT_DTYPE[numpy_dtype]
    as_float = float(value)
    if numpy_dtype.kind != "f":  # "f" = floating point
        return False
    if math.isnan(as_float) or math.isinf(as_float):
        return True
    if numpy_dtype == np.float32:
        round_trip = float(np.float32(as_float))
        return round_trip == as_float
    return True


def dtype_for_value(value) -> np.dtype:
    """Return the smallest type that stores ``value`` exactly.

    Whole numbers take the first integer type in the order ``Byte``, ``Int8``, ``Int16``, ``UInt16``, ``Int32``,
    ``UInt32``, ``Int64``, ``UInt64`` that holds them. NaN and infinity give ``float32``, every other float
    ``float64``.
    """
    if not is_number(value):
        raise GeoKitDataTypeError(f"{value!r} is not a number.")
    if is_whole_number(value):
        whole_number = int(value)
        integer_dtype = dtype_for_integer_range(whole_number, whole_number)
        if integer_dtype is not None:
            return integer_dtype
        return np.dtype(np.float64)
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def dtype_for_integer_range(low: int, high: int) -> np.dtype | None:
    """Return the first integer type of the order of choice that holds every value from ``low`` to ``high``.

    ``None`` means that no integer type holds the range.
    """
    for integer_dtype in _INTEGER_DTYPES_IN_ORDER_OF_CHOICE:
        limits = np.iinfo(integer_dtype)
        if limits.min <= low and high <= limits.max:
            return integer_dtype
    return None


def promote_dtypes(dtypes: Iterable) -> np.dtype | None:
    """Promote types as NumPy promotes arrays, and return ``None`` for no types.

    Only dtypes are promoted, never scalars, so the result is the same on NumPy 1.26 and NumPy 2.
    """
    numpy_dtypes = [to_dtype(dtype) for dtype in dtypes if dtype is not None]
    if not numpy_dtypes:
        return None
    promoted = np.result_type(*numpy_dtypes)
    return _supported_dtype(promoted, spelling=repr(promoted))


def describe_range(dtype) -> str:
    """Describe what a type stores exactly, for messages: the integer range, or the largest exact whole number."""
    numpy_dtype = to_dtype(dtype)
    if numpy_dtype.kind in "iu":  # "i"/"u" = signed/unsigned integer
        limits = np.iinfo(numpy_dtype)
        return f"range {limits.min} to {limits.max}"
    largest_whole_number = _LARGEST_EXACT_WHOLE_NUMBER_BY_FLOAT_DTYPE[numpy_dtype]
    return f"whole numbers exact up to {largest_whole_number}"
