"""Unit tests of ``geokit.dtypes.conversions``: pixel types, field types and scalars (ADR 2, 4, 5, 7, 9).

The rasters used here are built with plain GDAL; no GeoKit function is called apart from ``rasterInfo``.
"""

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, ogr
from typeguard import TypeCheckError

import geokit
from geokit import dtypes
from geokit.dtypes import conversions
from geokit.error import GeoKitDataTypeError, GeoKitDataTypeWarning, GeoKitError
from test.gdal_builders import create_gdal_raster

ALL_RASTER_DTYPES = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float32", "float64"]


# ----------------------------------------------------------------------------------------------
# to_dtype, to_gdal, from_band


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("Byte", "uint8"),
        ("GDT_Byte", "uint8"),
        ("byte", "uint8"),
        ("BYTE", "uint8"),
        ("uint8", "uint8"),
        ("u1", "uint8"),
        ("Int8", "int8"),
        ("UInt16", "uint16"),
        ("Int16", "int16"),
        ("UInt32", "uint32"),
        ("Int32", "int32"),
        ("UInt64", "uint64"),
        ("Int64", "int64"),
        ("Float32", "float32"),
        ("gdt_float64", "float64"),
        ("float64", "float64"),
        ("bool", "uint8"),
        ("float16", "float32"),
        (np.uint8, "uint8"),
        (np.dtype("int16"), "int16"),
        (np.float32, "float32"),
        (np.bool_, "uint8"),
        (np.float16, "float32"),
        (np.dtype(">f4"), "float32"),
        (bool, "uint8"),
        (int, str(np.dtype(int))),
        (float, "float64"),
    ],
)
def test_to_dtype_accepts_every_documented_spelling(spelling, expected):
    """Every spelling of ADR 2 gives the same numpy.dtype; bool and float16 become types GDAL has."""
    assert dtypes.to_dtype(spelling) == np.dtype(expected)


def test_to_dtype_keeps_none():
    """None has no type to convert and stays None."""
    assert dtypes.to_dtype(None) is None


@pytest.mark.parametrize(
    "rejected",
    [
        gdal.GDT_Float32,
        2,
        np.int64(3),
        3.5,
        True,
        "auto",
        "smallest",
        "preserve_input",
        "CFloat32",
        "complex64",
        np.complex64,
        "U10",
        "str",
        "object",
        np.dtype("O"),
        "no_such_type",
        [1, 2],
    ],
    ids=repr,
)
def test_to_dtype_rejects_what_a_band_cannot_store(rejected):
    """Bare integers, values, mode strings, complex, string and object types raise GeoKitDataTypeError."""
    with pytest.raises(GeoKitDataTypeError):
        dtypes.to_dtype(rejected)


def test_bare_integer_error_names_the_string_spelling():
    """The error for gdal.GDT_Float32 tells the caller to write dtype="Float32" instead."""
    with pytest.raises(GeoKitDataTypeError, match='dtype="Float32"'):
        dtypes.to_dtype(gdal.GDT_Float32)


@pytest.mark.parametrize(
    "spelling, constant",
    [
        ("Byte", gdal.GDT_Byte),
        ("uint16", gdal.GDT_UInt16),
        (np.float64, gdal.GDT_Float64),
        (bool, gdal.GDT_Byte),
        ("Int8", gdal.GDT_Int8),
        (np.dtype("int64"), gdal.GDT_Int64),
    ],
)
def test_to_gdal_returns_the_pixel_type_constant(spelling, constant):
    """to_gdal accepts every spelling and returns the matching gdal.GDT_* constant."""
    assert dtypes.to_gdal(spelling) == constant


def test_to_gdal_rejects_none():
    """A GDAL constant cannot be made from None."""
    with pytest.raises(GeoKitDataTypeError):
        dtypes.to_gdal(None)


def test_gdal_type_name_is_the_gdal_spelling():
    """gdal_type_name gives the GDAL name used in messages."""
    assert dtypes.gdal_type_name(np.uint16) == "UInt16"
    assert dtypes.gdal_type_name("bool") == "Byte"


@pytest.mark.parametrize("numpy_name", ALL_RASTER_DTYPES)
def test_from_band_round_trip(numpy_name):
    """A band created with to_gdal reads back as the same numpy.dtype, from the band or from the dataset."""
    raster = create_gdal_raster(np.zeros((2, 2), numpy_name), dtypes.to_gdal(numpy_name))

    assert dtypes.from_band(raster.GetRasterBand(1)) == np.dtype(numpy_name)
    assert dtypes.from_band(raster) == np.dtype(numpy_name)


def test_from_band_rejects_complex_bands():
    """A complex band has no GeoKit type."""
    complex_raster = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_CFloat32)

    with pytest.raises(GeoKitDataTypeError, match="CFloat32"):
        dtypes.from_band(complex_raster)


def test_raster_info_carries_the_numpy_dtype():
    """RasterInfo.numpy_dtype is the band type as a numpy.dtype."""
    raster = create_gdal_raster(np.zeros((2, 2), np.uint16), gdal.GDT_UInt16)

    assert geokit.raster.rasterInfo(raster).numpy_dtype == np.dtype("uint16")


# ----------------------------------------------------------------------------------------------
# OGR fields


@pytest.mark.parametrize(
    "field_type, subtype, expected",
    [
        (ogr.OFTInteger, ogr.OFSTNone, "int32"),
        (ogr.OFTInteger, ogr.OFSTInt16, "int16"),
        (ogr.OFTInteger, ogr.OFSTBoolean, "uint8"),
        (ogr.OFTInteger64, ogr.OFSTNone, "int64"),
        (ogr.OFTReal, ogr.OFSTNone, "float64"),
        (ogr.OFTReal, ogr.OFSTFloat32, "float32"),
    ],
)
def test_from_ogr_field_uses_type_and_subtype(field_type, subtype, expected):
    """Numeric OGR fields map to the dtype of ADR 9, with the subtypes narrowing them."""
    field_definition = ogr.FieldDefn("value", field_type)
    field_definition.SetSubType(subtype)

    assert dtypes.from_ogr_field(field_definition) == np.dtype(expected)


@pytest.mark.parametrize("field_type", [ogr.OFTString, ogr.OFTDate, ogr.OFTDateTime, ogr.OFTBinary])
def test_from_ogr_field_is_none_for_non_numeric_fields(field_type):
    """A field that holds no numbers has no dtype."""
    assert dtypes.from_ogr_field(ogr.FieldDefn("value", field_type)) is None


def test_from_ogr_field_rejects_integer_constants():
    """An integer could be an OGR or a GDAL constant, so only the FieldDefn object is accepted."""
    # The test suite runs typeguard, which rejects the integer at the annotation before the function does
    with pytest.raises((GeoKitDataTypeError, TypeCheckError)):
        dtypes.from_ogr_field(ogr.OFTReal)


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (bool, ogr.OFTInteger),
        ("int8", ogr.OFTInteger),
        ("uint8", ogr.OFTInteger),
        ("int16", ogr.OFTInteger),
        ("uint16", ogr.OFTInteger),
        ("int32", ogr.OFTInteger),
        ("uint32", ogr.OFTInteger64),
        ("int64", ogr.OFTInteger64),
        ("uint64", ogr.OFTInteger64),
        ("float16", ogr.OFTReal),
        ("float32", ogr.OFTReal),
        ("float64", ogr.OFTReal),
        (str, ogr.OFTString),
        ("object", ogr.OFTString),
        (np.dtype("U5"), ogr.OFTString),
        ("datetime64[ns]", ogr.OFTString),
        (pd.BooleanDtype(), ogr.OFTInteger),
        (pd.Int32Dtype(), ogr.OFTInteger),
        (pd.Int64Dtype(), ogr.OFTInteger64),
        (pd.Float64Dtype(), ogr.OFTReal),
        (pd.StringDtype(), ogr.OFTString),
    ],
    ids=str,
)
def test_to_ogr_field_follows_the_table_of_adr_9(dtype, expected):
    """NumPy, pandas and Python types map to the OGR field type of ADR 9; uint32 and 64-bit integers to Integer64."""
    assert dtypes.to_ogr_field(dtype) == expected


def test_to_ogr_field_rejects_complex():
    """Complex values cannot be stored in an OGR field."""
    with pytest.raises(GeoKitDataTypeError):
        dtypes.to_ogr_field("complex64")


# ----------------------------------------------------------------------------------------------
# Scalars and promotion


@pytest.mark.parametrize(
    "value, expected",
    [(3, True), (3.0, True), (True, True), (np.int16(-2), True), (0.5, False), (np.nan, False), (np.inf, False)],
    ids=str,
)
def test_is_whole_number(value, expected):
    """Integers and whole floats are whole numbers; fractions, NaN and infinity are not."""
    assert conversions.is_whole_number(value) is expected


@pytest.mark.parametrize(
    "dtype, value, expected",
    [
        ("Byte", 0, True),
        ("Byte", 255, True),
        ("Byte", 256, False),
        ("Byte", -1, False),
        ("Byte", 3.0, True),
        ("Byte", 0.5, False),
        ("Byte", np.nan, False),
        ("Byte", True, True),
        ("Byte", np.int64(200), True),
        ("Byte", np.float32(0.5), False),
        ("Int8", -128, True),
        ("Int8", 128, False),
        ("UInt16", -1, False),
        ("UInt16", 65535, True),
        ("UInt16", 65536, False),
        ("Int16", -9999, True),
        ("Int64", 2**63 - 1, True),
        ("Int64", 2**63, False),
        ("UInt64", 2**64 - 1, True),
        ("Float32", 0.5, True),
        ("Float32", 0.1, False),
        ("Float32", np.nan, True),
        ("Float32", np.inf, True),
        ("Float32", 2**24, True),
        ("Float32", 2**24 + 1, False),
        ("Float32", -9999, True),
        ("Float64", 0.1, True),
        ("Float64", 2**53, True),
        ("Float64", 2**53 + 1, False),
        ("Float64", np.nan, True),
        ("Byte", "7", False),
        ("Byte", None, False),
    ],
    ids=str,
)
def test_can_hold_means_stored_exactly(dtype, value, expected):
    """A value fits a type only if it can be stored exactly: range for integers, precision for floats."""
    assert dtypes.can_hold(dtype, value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "uint8"),
        (255, "uint8"),
        (256, "int16"),
        (-1, "int8"),
        (-128, "int8"),
        (-129, "int16"),
        (300, "int16"),
        (32767, "int16"),
        (32768, "uint16"),
        (40000, "uint16"),
        (65536, "int32"),
        (2**31 - 1, "int32"),
        (2**31, "uint32"),
        (2**32, "int64"),
        (2**63, "uint64"),
        (2**64, "float64"),
        (np.nan, "float32"),
        (np.inf, "float32"),
        (0.1, "float64"),
        (0.5, "float64"),
        (3.0, "uint8"),
        (True, "uint8"),
        (np.int64(-5), "int8"),
        (np.float64(2.5), "float64"),
    ],
    ids=str,
)
def test_dtype_for_value_follows_the_order_of_adr_5(value, expected):
    """Byte comes before Int8; from 16 bits on the signed type comes first; NaN gives float32, other floats float64."""
    assert dtypes.dtype_for_value(value) == np.dtype(expected)


def test_dtype_for_value_rejects_non_numbers():
    """Only numbers have a smallest type."""
    with pytest.raises(GeoKitDataTypeError):
        dtypes.dtype_for_value("7")


@pytest.mark.parametrize(
    "low, high, expected",
    [
        (0, 255, "uint8"),
        (-1, 1, "int8"),
        (0, 300, "int16"),
        (-5, 40000, "int32"),
        (0, 40000, "uint16"),
        (0, 2**64, None),
    ],
    ids=str,
)
def test_dtype_for_integer_range(low, high, expected):
    """The first integer type of the order of choice that holds the whole range is chosen, or None."""
    chosen = conversions.dtype_for_integer_range(low, high)

    if expected is None:
        assert chosen is None
    else:
        assert chosen == np.dtype(expected)


@pytest.mark.parametrize(
    "given, expected",
    [
        ([], None),
        (["Byte"], "uint8"),
        (["Byte", "Int8"], "int16"),
        (["UInt16", "Int16"], "int32"),
        (["UInt32", "Int32"], "int64"),
        (["UInt64", "Int8"], "float64"),
        (["Int64", "Float32"], "float64"),
        (["Int16", "Float32"], "float32"),
        ([np.bool_], "uint8"),
        (["Byte", np.float16], "float32"),
        (["Byte", None], "uint8"),
    ],
    ids=str,
)
def test_promote_dtypes_promotes_like_numpy_arrays(given, expected):
    """Promotion follows NumPy's array promotion and returns None for no types."""
    promoted = dtypes.promote_dtypes(given)

    if expected is None:
        assert promoted is None
    else:
        assert promoted == np.dtype(expected)


def test_describe_range_for_messages():
    """Integer types are described by their range, float types by the largest exact whole number."""
    assert conversions.describe_range("UInt16") == "range 0 to 65535"
    assert conversions.describe_range("Float32") == "whole numbers exact up to 16777216"


# ----------------------------------------------------------------------------------------------
# error classes


def test_error_and_warning_classes():
    """GeoKitDataTypeError is a GeoKitError, GeoKitDataTypeWarning a UserWarning, and both are re-exported."""
    assert issubclass(GeoKitDataTypeError, GeoKitError)
    assert issubclass(GeoKitDataTypeWarning, UserWarning)
    assert dtypes.GeoKitDataTypeError is GeoKitDataTypeError
    assert dtypes.GeoKitDataTypeWarning is GeoKitDataTypeWarning
