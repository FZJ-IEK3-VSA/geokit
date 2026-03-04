"""Tests for MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string.

Each test exercises a different aspect of automatic GDAL data-type selection:

* test_minimum_c_data_handler_min_max_values
    Pure boundary-value coverage for signed/unsigned integer types.
    No minimum-type constraints are supplied, so the handler must pick the
    smallest type whose range contains every value in the input list.

* test_minimum_c_data_handler_float_numbers_minimum_data_type
    Verifies that floating-point values (or mixed int/float lists) are mapped
    to the smallest compatible GDAL float type (Float32 vs Float64).

* test_minimum_c_data_handler_only_minimum_data_type_string
    Verifies that when only a minimum data type string is given, it is returned as-is.

* test_other_data_type
    Edge-case inputs: Python/NumPy booleans and the IEEE-754 specials
    ``np.inf`` and ``np.nan``.

* test_minimum_c_data_handler_external_minimum_data_type_exact_match
    The caller supplies both a ``minimum_gdal_type_list`` (a lower bound on
    bit-width) and a ``user_defined_minimum_gdal_type`` (preferred signed/unsigned
    variant). When both constraints are compatible with the data the result must
    match ``user_defined_minimum_gdal_type`` without raising a warning.

* test_minimum_c_data_handler_different_external_minimum_data_type
    The caller supplies a ``user_defined_minimum_gdal_type`` that is *not*
    consistent with the computed minimum type (e.g. signed vs unsigned mismatch,
    or an integer type for float data). The handler must emit a ``UserWarning``
    and still return the *correct* (computed) type rather than the requested one.

Tuple layout
------------
``test_minimum_c_data_handler_min_max_values`` and
``test_minimum_c_data_handler_float_numbers_minimum_data_type`` and
``test_other_data_type``::

    (values, minimum_gdal_type_list, expected_gdal_type)

``test_minimum_c_data_handler_external_minimum_data_type_exact_match`` and
``test_minimum_c_data_handler_different_external_minimum_data_type``::

    (values, minimum_gdal_type_list, user_defined_minimum_gdal_type, expected_gdal_type)
"""

import numpy as np
import pytest

from geokit.c_data_type_handler import MinimumCDataTypeHandler


def _assert_gdal_type_and_numpy_roundtrip(
    min_data_type: str,
    list_of_numbers: list,
    expected: str,
    context: object,
) -> None:
    """Assert that *min_data_type* equals *expected* and that the selected
    NumPy dtype can losslessly represent the first and last value of
    *list_of_numbers* (basic overflow check).
    """
    assert min_data_type == expected, (
        f"min_data_type: {min_data_type} != expected: {expected} for test case:\n{context}"
    )
    numpy_dtype_str = MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]
    dtype = np.dtype(numpy_dtype_str)
    # Verify that the boundary values can be stored in the chosen dtype without
    # silent overflow (constructing the typed scalar raises on out-of-range values
    # for integer dtypes).
    dtype.type(list_of_numbers[0])
    dtype.type(list_of_numbers[1])


def test_minimum_c_data_handler_min_max_values():
    """Boundary-value tests for integer type selection without external constraints.

    Each entry is ``(values, minimum_gdal_type_list, expected_gdal_type)``.
    The handler should pick the *smallest* type whose [min, max] range contains
    every value in *values*.  When both a signed and an unsigned type are equally
    minimal (e.g. Int8 vs Byte for [0, 127]), the signed variant is preferred.
    """
    # fmt: off
    test_data_list: list[tuple[list, None, str]] = [
        # --- Boolean inputs (treated as 0 / 1 integers) ---
        ([True, False],           None, "GDT_Int8"),   # Python bool → Int8 (signed preference)
        ([np.True_, np.False_],   None, "GDT_Int8"),   # NumPy bool_ → same result

        # --- 8-bit signed (Int8): range [-128, 127] ---
        ([-128, 127],             None, "GDT_Int8"),   # exact range boundaries
        ([0, 127],                None, "GDT_Int8"),   # positive-only, still fits in signed Int8

        # --- 8-bit unsigned (Byte / UInt8): range [0, 255] ---
        ([0, 255],                None, "GDT_Byte"),   # 128–255 exceeds Int8 → promotes to Byte

        # --- 16-bit signed (Int16): range [-32768, 32767] ---
        ([-32768, 32767],         None, "GDT_Int16"),  # exact range boundaries
        ([0, 32767],              None, "GDT_Int16"),  # positive-only, fits in Int16 (signed pref.)

        # --- 16-bit unsigned (UInt16): range [0, 65535] ---
        ([0, 65535],              None, "GDT_UInt16"), # 32768–65535 exceeds Int16 → UInt16

        # --- 32-bit signed (Int32): range [-2 147 483 648, 2 147 483 647] ---
        ([-2147483648, 2147483647],  None, "GDT_Int32"),
        ([0, 2147483647],            None, "GDT_Int32"),

        # --- 32-bit unsigned (UInt32): range [0, 4 294 967 295] ---
        ([0, 4294967295],            None, "GDT_UInt32"),

        # --- 64-bit signed (Int64): range [-2^63, 2^63 - 1] ---
        ([-9223372036854775808, 9223372036854775807], None, "GDT_Int64"),
        ([0, 9223372036854775807],                   None, "GDT_Int64"),

        # --- 64-bit unsigned (UInt64): range [0, 2^64 - 1] ---
        ([0, 18446744073709551615],                  None, "GDT_UInt64"),
    ]
    # fmt: on

    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers, minimum_gdal_type, expected = current_test_data
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers, minimum_gdal_type_list=minimum_gdal_type
        )
        _assert_gdal_type_and_numpy_roundtrip(min_data_type, list_of_numbers, expected, current_test_data)


def test_minimum_c_data_handler_float_numbers_minimum_data_type():
    """Float-type selection without external constraints.

    Each entry is ``(values, minimum_gdal_type_list, expected_gdal_type)``.

    NumPy's ``min_scalar_type`` is used internally to map each Python float to
    a NumPy float dtype; the handler then escalates to the larger of Float32 /
    Float64 as needed.  The boundary between Float32 and Float64 is roughly
    3.4 × 10^38 (max representable finite Float32 value).
    """
    # fmt: off
    test_data_list = (
        ([0.333, 1.777],     None, "GDT_Float32"),  # ordinary floats → Float32
        ([0.333, 1],         None, "GDT_Float32"),  # mixed float + int → Float32
        ([1 / 3, 1],         None, "GDT_Float32"),  # irrational fraction → Float32
        ([3 * 10**37, 1],    None, "GDT_Float32"),  # large but still in Float32 range
        ([3 * 10**40, 1],    None, "GDT_Float64"),  # exceeds Float32 max → Float64
    )
    # fmt: on

    for current_test_data in test_data_list:
        list_of_numbers, minimum_gdal_type, expected = current_test_data
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers, minimum_gdal_type_list=minimum_gdal_type
        )
        _assert_gdal_type_and_numpy_roundtrip(min_data_type, list_of_numbers, expected, current_test_data)


def test_minimum_c_data_handler_only_minimum_data_type_string():
    """When only a minimum data type string is given, it is returned as-is.

    Each entry is ``(values, minimum_gdal_type_list, expected_gdal_type)``.
    The handler should return the provided minimum data type string without
    modification, and without raising an error or warning, even if the input
    values are not compatible with that type (e.g. floats with an integer type).
    """
    test_data_list = (
        ([], ["GDT_Float32"], "GDT_Float32"),
        ([], ["GDT_Float64"], "GDT_Float64"),
        ([], ["GDT_Int8"], "GDT_Int8"),
        ([], ["GDT_Int16"], "GDT_Int16"),
        ([], ["GDT_Int32"], "GDT_Int32"),
        ([], ["GDT_Int64"], "GDT_Int64"),
    )

    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers, minimum_gdal_type, expected = current_test_data
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers,
            minimum_gdal_type_list=minimum_gdal_type,
        )
        assert min_data_type == expected, (
            f"min_data_type: {min_data_type} != expected: {expected} for test case:\n{current_test_data}"
        )


def test_other_data_type():
    """Edge-case inputs: single booleans and IEEE-754 specials.

    Each entry is ``(values, minimum_gdal_type_list, expected_gdal_type)``.

    * A single ``True`` / ``False`` must be treated as an integer (not a float).
    * ``np.inf`` and ``np.nan`` are floating-point values and must map to at
      least Float32 (no integer type can represent them).
    """
    # fmt: off
    test_data_list = (
        ([True],    None, "GDT_Int8"),    # single bool True  → integer path → Int8
        ([False],   None, "GDT_Int8"),    # single bool False → integer path → Int8
        ([np.inf],  None, "GDT_Float32"), # infinity → float path → smallest float = Float32
        ([np.nan],  None, "GDT_Float32"), # NaN      → float path → smallest float = Float32
    )
    # fmt: on

    for current_test_data in test_data_list:
        list_of_numbers, minimum_gdal_type, expected = current_test_data
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers, minimum_gdal_type_list=minimum_gdal_type
        )
        assert min_data_type == expected, (
            f"min_data_type: {min_data_type} != expected: {expected} for test case:\n{current_test_data}"
        )
        # Verify the returned GDAL type maps to a valid NumPy dtype string.
        MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]


def test_minimum_c_data_handler_minimum_gdal_type_list_only():
    """``minimum_gdal_type_list`` as the sole constraint — no ``user_defined_minimum_gdal_type``.

    Each entry is ``(values, minimum_gdal_type_list, expected_gdal_type)``.

    ``minimum_gdal_type_list`` sets a **lower bound on bit-width**.  The handler
    returns the larger of:

    * the natural minimum type that fits the data, and
    * the largest type listed in ``minimum_gdal_type_list``.

    When integer bit-widths are up-cast due to the list, the signed variant of
    that bit-width is returned (unsigned only if the data itself requires it).
    When any float type appears in the list the float code-path is activated and
    an integer-only data list is escalated to a float type.
    """
    # fmt: off
    test_data_list: list[tuple[list, list[str], str]] = [
        # --- list matches the natural minimum exactly (no up-cast) ---
        ([-128, 127],       ["GDT_Int8"],               "GDT_Int8"),   # data fits Int8; list confirms Int8
        ([0, 255],          ["GDT_Byte"],               "GDT_Byte"),   # data needs Byte; list confirms Byte
        ([-32768, 32767],   ["GDT_Int16"],              "GDT_Int16"),  # data fits Int16; list confirms Int16
        ([0, 65535],        ["GDT_UInt16"],             "GDT_UInt16"), # data needs UInt16; list confirms UInt16
        ([-2147483648, 2147483647], ["GDT_Int32"],      "GDT_Int32"),
        ([0, 4294967295],   ["GDT_UInt32"],             "GDT_UInt32"),
        ([-9223372036854775808, 9223372036854775807], ["GDT_Int64"], "GDT_Int64"),
        ([0, 18446744073709551615], ["GDT_UInt64"],     "GDT_UInt64"),

        # --- list forces an up-cast (list type is larger than natural minimum) ---
        # data fits Int8, but list demands at least Int16 → signed 16-bit
        ([-128, 127],       ["GDT_Int16"],              "GDT_Int16"),
        # data fits Int8, but list demands at least Int32 → signed 32-bit
        ([-128, 127],       ["GDT_Int32"],              "GDT_Int32"),
        # data fits Int8, but list demands at least Int64 → signed 64-bit
        ([-128, 127],       ["GDT_Int64"],              "GDT_Int64"),
        # data fits Byte (pos-only), list demands Int16 → signed 16-bit
        ([0, 200],          ["GDT_Int16"],              "GDT_Int16"),
        # data fits Byte, list demands UInt16 → the handler up-casts to the
        # *signed* variant of the same bit-width (Int16), because [0, 200]
        # already fits in Int16 and signed is preferred over unsigned.
        ([0, 200],          ["GDT_UInt16"],             "GDT_Int16"),
        # data fits Int16, list demands Int32 → signed 32-bit
        ([-32768, 32767],   ["GDT_Int32"],              "GDT_Int32"),
        # data fits Int16, list demands Int64 → signed 64-bit
        ([-32768, 32767],   ["GDT_Int64"],              "GDT_Int64"),

        # --- multiple types in list; largest bit-width wins ---
        # both Int8 and Int16; Int16 is wider → signed 16-bit
        ([-128, 127],       ["GDT_Int8", "GDT_Int16"],  "GDT_Int16"),
        # Int8 and UInt32; UInt32 is wider → signed 32-bit (up-cast uses signed variant)
        ([-128, 127],       ["GDT_Int8", "GDT_UInt32"], "GDT_Int32"),

        # --- data already larger than the list; list is ignored ---
        # data spans full Int32 range, list only asks for Int8 → Int32 from data
        ([-2147483648, 2147483647], ["GDT_Int8"],       "GDT_Int32"),
        # data needs UInt16, list only asks for Int8 → Uint16 from data
        ([0, 65535],        ["GDT_Int8"],               "GDT_UInt16"),

        # --- float type in list forces float output even for integer data ---
        # data is small integers, but list contains Float32 → Float32
        ([-128, 127],       ["GDT_Float32"],            "GDT_Float32"),
        # data is small integers, list contains Float64 → Float64
        ([0, 255],          ["GDT_Float64"],            "GDT_Float64"),
        # data is ordinary floats, list confirms Float32 → Float32
        ([0.5, 1.5],        ["GDT_Float32"],            "GDT_Float32"),
        # data needs Float32 only, but list demands Float64 → Float64
        ([0.5, 1.5],        ["GDT_Float64"],            "GDT_Float64"),
        # data needs Float64, list only asks for Float32 → Float64 from data
        ([3 * 10**40, 1.0], ["GDT_Float32"],            "GDT_Float64"),

        # --- list contains mixed int + float; float path wins ---
        # Int32 + Float32 in list: float entry activates the float path → Float32.
        ([-128, 127],       ["GDT_Int32", "GDT_Float32"], "GDT_Float32"),
        # Int64 + Float32 in list: float path, small integer data → Float32.
        ([-128, 127],       ["GDT_Int64", "GDT_Float32"], "GDT_Float32"),
        # Int32 + Float64 in list: float path, Float64 in list → Float64.
        ([-128, 127],       ["GDT_Int32", "GDT_Float64"], "GDT_Float64"),
    ]
    # fmt: on

    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers, minimum_gdal_type, expected = current_test_data
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers,
            minimum_gdal_type_list=minimum_gdal_type,
        )
        assert min_data_type == expected, (
            f"min_data_type: {min_data_type} != expected: {expected} for test case:\n{current_test_data}"
        )
        # Verify the returned GDAL type maps to a valid NumPy dtype.
        MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]


@pytest.mark.filterwarnings("error: The user-defined:UserWarning")
def test_minimum_c_data_handler_external_minimum_data_type_exact_match():
    """External constraints that are *consistent* with the data — no warning expected.

    Each entry is
    ``(values, minimum_gdal_type_list, user_defined_minimum_gdal_type, expected_gdal_type)``.

    ``minimum_gdal_type_list`` sets a lower bound on the required bit-width;
    ``user_defined_minimum_gdal_type`` expresses the caller's preferred
    signed/unsigned variant.  When the requested type is valid for the data the
    handler must return it exactly, and must *not* emit a ``UserWarning``
    (the ``filterwarnings("error")`` marker turns any warning into a failure).
    """
    # fmt: off
    test_data_list = (
        # --- minimum_gdal_type_list matches the natural minimum (no up-cast needed) ---
        ([-128, 127],       ["GDT_Int8"],              None,          "GDT_Int8"),   # list pins Int8; no user pref
        ([-128, 127],       ["GDT_Int16"],             "GDT_Int16",   "GDT_Int16"),  # list forces up-cast to Int16; user agrees

        # --- user wants unsigned variant; data fits → honour the request ---
        ([0, 127],          ["GDT_Byte"],              "GDT_Byte",    "GDT_Byte"),   # pos-only, user requests Byte
        ([0, 127],          ["GDT_Byte"],              "GDT_Byte",    "GDT_Byte"),   # duplicate to confirm idempotency

        # --- empty minimum_gdal_type_list; user drives the signed/unsigned choice ---
        ([0, 127],          [],                        "GDT_Byte",    "GDT_Byte"),   # empty list → user picks Byte
        ([0, 127],          [],                        "GDT_Int8",    "GDT_Int8"),   # empty list → user picks Int8

        # --- float types: list pins 64-bit, user agrees ---
        ([0.333, 1.777],    ["GDT_Float64"],           "GDT_Float64", "GDT_Float64"),
        ([3 * 10**40, 1],   ["GDT_Float64"],           "GDT_Float64", "GDT_Float64"),  # data needs 64-bit anyway

        # --- float: data fits Float32, list and user both request Float32 ---
        ([3 * 10**37, 1],   ["GDT_Float32"],           "GDT_Float32", "GDT_Float32"),

        # --- list contains both signed and unsigned targets of same bit-width ---
        ([-128, 127],       ["GDT_Int8", "GDT_Byte"],  "GDT_Int8",    "GDT_Int8"),   # signed preferred; user confirms

        # --- 16-bit: list forces Int16 / UInt16; user matches ---
        ([0, 32767],        ["GDT_Int16"],             "GDT_Int16",   "GDT_Int16"),
        ([0, 32767],        ["GDT_UInt16"],            "GDT_UInt16",  "GDT_UInt16"),
    )
    # fmt: on

    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers, minimum_gdal_type, user_defined_minimum_gdal_type, expected = current_test_data
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers,
            minimum_gdal_type_list=minimum_gdal_type,
            user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
        )
        _assert_gdal_type_and_numpy_roundtrip(min_data_type, list_of_numbers, expected, current_test_data)


def test_minimum_c_data_handler_different_external_minimum_data_type():
    """External constraints that *conflict* with the data — a UserWarning must be raised.

    Each entry is
    ``(values, minimum_gdal_type_list, user_defined_minimum_gdal_type, expected_gdal_type)``.

    The handler cannot honour ``user_defined_minimum_gdal_type`` without causing
    overflow or type-incompatibility (e.g. the user requests an integer type for
    floating-point data, or a signed type when the data contains values that
    exceed the signed range).  In all such cases the handler must:

    1. Emit a ``UserWarning`` (checked with ``pytest.warns``).
    2. Return the *computed* correct type (``expected_gdal_type``), ignoring the
       user's request.
    """
    # fmt: off
    test_data_list = (
        # --- signed/unsigned mismatch at 8-bit ---
        # [-128, 130]: needs a type that covers both negatives and > 127.
        # GDT_Int8 (min list) + GDT_Byte (user) → neither alone works → Int16.
        ([-128, 130],       ["GDT_Int8"],  "GDT_Byte",    "GDT_Int16"),

        # [-128, 127] fits Int8.  User asks for Byte (unsigned), but negative
        # values can't be stored in Byte → handler warns and returns Int8.
        ([-128, 127],       ["GDT_Byte"],  "GDT_Byte",    "GDT_Int8"),

        # [0, 255] fits Byte.  User asks for Int8 (signed 8-bit), but 128–255
        # overflow Int8 → handler warns and returns Byte.
        ([0, 255],          [],            "GDT_Int8",    "GDT_Byte"),

        # --- integer type requested for float-magnitude data ---
        # 3×10^40 cannot be represented by any integer type.
        # Various integer minimum lists / user preferences all conflict → Float64.
        ([3 * 10**40, 1],   ["GDT_Int8"],  "GDT_Int8",    "GDT_Float64"),
        ([3 * 10**40, 1],   ["GDT_Int16"], "GDT_Float32", "GDT_Float64"),  # Float32 insufficient for 10^40
        ([3 * 10**40, 1],   ["GDT_UInt16"],"GDT_Int32",   "GDT_Float64"),
        ([3 * 10**40, 1],   ["GDT_Int64"], "GDT_Int8",    "GDT_Float64"),

        # 3×10^37 fits in Float32 but not in any integer type.
        # User requests integer types (UInt64) → conflict → Float32.
        ([3 * 10**37, 1],   ["GDT_Int8"],  "GDT_UInt64",  "GDT_Float32"),
        ([3 * 10**37, 1],   ["GDT_Int64"], "GDT_UInt64",  "GDT_Float32"),
    )
    # fmt: on

    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers, minimum_gdal_type, user_defined_minimum_gdal_type, expected = current_test_data
        with pytest.warns(UserWarning):
            min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
                list_of_numbers=list_of_numbers,
                minimum_gdal_type_list=minimum_gdal_type,
                user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
            )
        _assert_gdal_type_and_numpy_roundtrip(min_data_type, list_of_numbers, expected, current_test_data)
