"""Unit tests of ``geokit.dtypes.resolve`` and ``geokit.dtypes.options``: the modes, the rules and the checks.

The tables are those of ADR 1 (modes and worked examples), ADR 3 (rules), ADR 4 (scalar widening) and
ADR 10 (options). No GeoKit function is called; the call sites follow in later pull requests.
"""

import threading
import warnings

import numpy as np
import pytest
from osgeo import gdal

from geokit import dtypes
from geokit.dtypes import DtypeRule
from geokit.error import GeoKitDataTypeError, GeoKitDataTypeWarning


@pytest.fixture(autouse=True)
def restore_options():
    """Every test starts with the default options and leaves them as they were."""
    options_before = dtypes.get_options()
    dtypes.set_options(checks=True)
    yield
    dtypes.set_options(checks=options_before.checks)


def gdal_name(numpy_dtype) -> str:
    return dtypes.gdal_type_name(numpy_dtype)


# ----------------------------------------------------------------------------------------------
# dtype_mode and the shared docstring


@pytest.mark.parametrize(
    "dtype, expected",
    [(None, "auto"), ("auto", "auto"), ("preserve_input", "preserve_input"), ("smallest", "smallest")]
    + [("Auto", "auto"), ("Byte", "explicit"), (np.uint8, "explicit"), (gdal.GDT_Byte, "explicit")],
    ids=str,
)
def test_dtype_mode(dtype, expected):
    """None means auto, the three mode strings are recognised, and everything else is an explicit type."""
    assert dtypes.dtype_mode(dtype) == expected


def test_shared_dtype_docstring_names_the_modes():
    """The docstring block that every raster-writing function will reuse documents the three modes."""
    assert dtypes.DTYPE_PARAMETER_DOCSTRING.startswith("dtype : ")
    for mode in dtypes.DTYPE_MODES:
        assert f'"{mode}"' in dtypes.DTYPE_PARAMETER_DOCSTRING


# ----------------------------------------------------------------------------------------------
# auto


@pytest.mark.parametrize(
    "input_dtype, scalar, expected",
    [
        ("Byte", 255, "Byte"),
        ("Byte", -1, "Int16"),
        ("UInt16", -1, "Int32"),
        ("Byte", np.nan, "Float32"),
        ("Int16", np.nan, "Float32"),
        ("Int32", np.nan, "Float64"),
        ("Float32", -9999, "Float32"),
        (None, 0.1, "Float64"),
        (None, -1, "Int16"),
    ],
    ids=str,
)
def test_auto_widens_until_every_scalar_fits(input_dtype, scalar, expected):
    """Under auto the type widens for a noData value that does not fit, as in the table of ADR 4."""
    input_dtypes = [] if input_dtype is None else [input_dtype]

    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoKitDataTypeWarning)
        resolved = dtypes.resolve_dtype(input_dtypes, DtypeRule.IDENTITY, scalars={"noData": scalar})

    assert gdal_name(resolved.dtype) == expected
    assert resolved.mode == "auto"
    assert resolved.shrink_output is False


@pytest.mark.parametrize("input_dtype, scalar", [("Int64", np.nan), ("UInt64", -1)], ids=str)
def test_auto_warns_when_a_64_bit_integer_type_must_become_float(input_dtype, scalar):
    """Int64 with a NaN noData, or UInt64 with a negative one, gives Float64 and a warning that names dtype."""
    with pytest.warns(GeoKitDataTypeWarning, match="dtype"):
        resolved = dtypes.resolve_dtype([input_dtype], DtypeRule.IDENTITY, scalars={"noData": scalar}, context="rasterize")

    assert gdal_name(resolved.dtype) == "Float64"


def test_a_raster_made_from_nothing_is_byte():
    """createRaster() without data, dtype, fill or noData gives Byte in every automatic mode."""
    for mode in [None, "auto", "preserve_input", "smallest"]:
        assert gdal_name(dtypes.resolve_dtype(dtype=mode).dtype) == "Byte"


def test_none_scalars_are_skipped_and_non_numbers_rejected():
    """A scalar of None is not a value to store; a string is not a scalar at all."""
    resolved = dtypes.resolve_dtype(["Byte"], scalars={"noData": None, "fill": None})
    assert gdal_name(resolved.dtype) == "Byte"

    with pytest.raises(GeoKitDataTypeError, match="noData"):
        dtypes.resolve_dtype(["Byte"], scalars={"noData": "x"})


@pytest.mark.parametrize(
    "rule, input_dtypes, expected",
    [
        (DtypeRule.SUBSET, ["Byte"], "Byte"),
        (DtypeRule.SUBSET, ["UInt16"], "UInt16"),
        (DtypeRule.IDENTITY, ["Int64"], "Int64"),
        (DtypeRule.UNION, ["Byte", "Int16"], "Int16"),
        (DtypeRule.UNION, ["Byte", "Float32"], "Float32"),
        (DtypeRule.FRACTIONAL, ["Byte"], "Float32"),
        (DtypeRule.FRACTIONAL, ["Int8"], "Float32"),
        (DtypeRule.FRACTIONAL, ["Int16"], "Float32"),
        (DtypeRule.FRACTIONAL, ["UInt16"], "Float32"),
        (DtypeRule.FRACTIONAL, ["Float32"], "Float32"),
        (DtypeRule.FRACTIONAL, ["Int32"], "Float64"),
        (DtypeRule.FRACTIONAL, ["UInt32"], "Float64"),
        (DtypeRule.FRACTIONAL, ["Int64"], "Float64"),
        (DtypeRule.FRACTIONAL, ["Float64"], "Float64"),
        (DtypeRule.SUM, ["Byte"], "Float64"),
        (DtypeRule.SUM, ["Float32"], "Float64"),
        (DtypeRule.DERIVATIVE, ["UInt16"], "Float64"),
        (DtypeRule.USER_FUNCTION, ["Float32"], "Float32"),
        (DtypeRule.USER_FUNCTION, ["UInt16"], "UInt16"),
    ],
    ids=str,
)
def test_auto_applies_the_rule_table_of_adr_3(rule, input_dtypes, expected):
    """Each rule gives the type of ADR 3: subset, identity, union and user function keep it, the others widen."""
    resolved = dtypes.resolve_dtype(input_dtypes, rule)

    assert gdal_name(resolved.dtype) == expected


@pytest.mark.parametrize(
    "input_dtypes, input_values, sum_count, expected",
    [
        ([], [100], 3, "Int16"),
        ([], [1], 3, "Byte"),
        ([], [100], 1, "Byte"),
        ([], [200], 2, "Int16"),
        ([], [0.5], 3, "Float64"),
        ([], [-1], 3, "Int8"),
        (["Byte"], [], 3, "Int16"),
        (["Int32"], [], 3, "Int64"),
        (["Float64"], [], 3, "Float64"),
    ],
    ids=str,
)
def test_sum_of_burns_holds_count_times_the_burn(input_dtypes, input_values, sum_count, expected):
    """rasterize(add=True) gets the narrowest integer type for count times the burn value or the field range."""
    resolved = dtypes.resolve_dtype(input_dtypes, DtypeRule.SUM_OF_BURNS, input_values=input_values, sum_count=sum_count)

    assert gdal_name(resolved.dtype) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(1, "Byte"), (200, "Byte"), (40000, "UInt16"), (-1, "Int8"), (0.1, "Float64"), (300, "Int16")],
    ids=str,
)
def test_a_constant_burn_value_is_an_input_of_its_smallest_type(value, expected):
    """The worked examples of ADR 1 for rasterize(value=constant) hold in every automatic mode."""
    for mode in [None, "auto", "preserve_input", "smallest"]:
        resolved = dtypes.resolve_dtype(input_values=[value], dtype=mode, scalars={"burn value": value})
        assert gdal_name(resolved.dtype) == expected


# ----------------------------------------------------------------------------------------------
# preserve_input and smallest


@pytest.mark.parametrize(
    "rule, input_dtypes, expected",
    [
        (DtypeRule.FRACTIONAL, ["Byte"], "Byte"),
        (DtypeRule.SUM, ["Byte"], "Byte"),
        (DtypeRule.UNION, ["Byte", "Int16"], "Int16"),
        (DtypeRule.IDENTITY, ["Float64"], "Float64"),
    ],
    ids=str,
)
def test_preserve_input_keeps_the_promoted_input_type_without_a_warning(rule, input_dtypes, expected):
    """Under preserve_input no input is narrowed, the rule does not widen, and nothing is checked or warned."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoKitDataTypeWarning)
        resolved = dtypes.resolve_dtype(input_dtypes, rule, dtype="preserve_input")

    assert gdal_name(resolved.dtype) == expected
    assert resolved.mode == "preserve_input"
    assert resolved.shrink_output is False


@pytest.mark.parametrize("input_dtype, scalar", [("Byte", -1), ("Int64", np.nan), ("UInt16", np.nan)], ids=str)
def test_preserve_input_raises_for_a_scalar_the_input_type_cannot_store(input_dtype, scalar):
    """A noData value the input type cannot store at all is an error under preserve_input."""
    with pytest.raises(GeoKitDataTypeError, match="preserve_input"):
        dtypes.resolve_dtype([input_dtype], scalars={"noData": scalar}, dtype="preserve_input")


def test_smallest_resolves_like_auto_and_asks_for_a_shrink():
    """The smallest mode chooses the auto type for the operation and tells the caller to shrink the output."""
    resolved = dtypes.resolve_dtype(["Byte"], DtypeRule.FRACTIONAL, dtype="smallest")

    assert gdal_name(resolved.dtype) == "Float32"
    assert resolved.mode == "smallest"
    assert resolved.shrink_output is True
    assert resolved.gdal_constant == gdal.GDT_Float32


# ----------------------------------------------------------------------------------------------
# explicit types


@pytest.mark.parametrize(
    "explicit, input_dtypes, rule, expected",
    [
        ("Float32", ["Float64"], DtypeRule.SUBSET, "Float32"),
        ("Byte", ["Byte"], DtypeRule.SUM, "Byte"),
        ("Int16", ["Byte"], DtypeRule.FRACTIONAL, "Int16"),
        (np.uint16, [], DtypeRule.IDENTITY, "UInt16"),
        ("GDT_Byte", ["Float64"], DtypeRule.IDENTITY, "Byte"),
        (bool, [], DtypeRule.IDENTITY, "Byte"),
    ],
    ids=str,
)
def test_an_explicit_type_is_used_as_given_without_a_warning(explicit, input_dtypes, rule, expected):
    """An explicit dtype replaces the mode and is not checked against the inputs or the operation (ADR 2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoKitDataTypeWarning)
        resolved = dtypes.resolve_dtype(input_dtypes, rule, dtype=explicit, scalars={"noData": 0})

    assert gdal_name(resolved.dtype) == expected
    assert resolved.mode == "explicit"
    assert resolved.shrink_output is False


def test_an_explicit_type_that_cannot_store_a_scalar_raises_with_the_message_of_adr_2():
    """The error names the function, the type and its range, the value, and how to fix the call."""
    with pytest.raises(GeoKitDataTypeError) as raised:
        dtypes.resolve_dtype(dtype="UInt16", scalars={"noData": -1}, context="createRaster")

    message = str(raised.value)
    assert message.startswith(
        "createRaster: noData=-1 cannot be stored in the requested dtype UInt16 (range 0 to 65535)."
    )
    assert 'dtype="Int32"' in message
    assert 'dtype="auto"' in message


@pytest.mark.parametrize("explicit", [gdal.GDT_Float32, "auto_please", "CInt16"], ids=str)
def test_an_explicit_type_that_is_not_a_type_raises(explicit):
    """Bare integers and unknown or complex names are rejected before anything is created."""
    with pytest.raises(GeoKitDataTypeError):
        dtypes.resolve_dtype(["Byte"], dtype=explicit)


def test_explicit_integer_type_with_nan_nodata_raises():
    """NaN cannot be stored in an integer type, so an explicit Int32 with noData=nan is an error."""
    with pytest.raises(GeoKitDataTypeError, match="Int32"):
        dtypes.resolve_dtype(dtype="Int32", scalars={"noData": np.nan})


# ----------------------------------------------------------------------------------------------
# options


def test_checks_are_on_by_default():
    """The default options issue the warnings."""
    assert dtypes.get_options().checks is True


def test_set_options_turns_the_warnings_off_but_keeps_the_type_and_the_errors():
    """With checks off, Int64 + NaN still gives Float64 silently, and impossible requests still raise."""
    dtypes.set_options(checks=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoKitDataTypeWarning)
        resolved = dtypes.resolve_dtype(["Int64"], scalars={"noData": np.nan})
    assert gdal_name(resolved.dtype) == "Float64"

    with pytest.raises(GeoKitDataTypeError):
        dtypes.resolve_dtype(dtype="Byte", scalars={"noData": -1})


def test_options_context_restores_the_previous_setting():
    """The options block changes the setting inside only and restores it afterwards, also when nested."""
    with dtypes.options(checks=False):
        assert dtypes.get_options().checks is False
        with dtypes.options(checks=True):
            assert dtypes.get_options().checks is True
        assert dtypes.get_options().checks is False
    assert dtypes.get_options().checks is True


def test_options_context_does_not_leak_into_other_threads():
    """A block in one thread leaves the other threads with the process-wide setting."""
    seen_in_thread = []

    def read_checks():
        seen_in_thread.append(dtypes.get_options().checks)

    with dtypes.options(checks=False):
        other_thread = threading.Thread(target=read_checks)
        other_thread.start()
        other_thread.join()

    assert seen_in_thread == [True]


def test_issue_warning_respects_the_checks_option():
    """issue_warning warns with GeoKitDataTypeWarning, and is silent when the checks are off."""
    with pytest.warns(GeoKitDataTypeWarning, match="dtype"):
        dtypes.issue_warning("the dtype may lose values")

    with dtypes.options(checks=False), warnings.catch_warnings():
        warnings.simplefilter("error", GeoKitDataTypeWarning)
        dtypes.issue_warning("the dtype may lose values")
