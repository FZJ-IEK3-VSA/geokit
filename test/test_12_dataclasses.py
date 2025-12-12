import numpy as np
import pytest

from geokit.c_data_type_handler import MinimumCDataTypeHandler


def test_minimum_c_data_handler_min_max_values():
    test_data_list: list[tuple[list, None, str]] = [
        ([True, False], None, "GDT_Int8"),
        ([np.True_, np.False_], None, "GDT_Int8"),
        ([-128, 127], None, "GDT_Int8"),
        ([0, 127], None, "GDT_Int8"),
        ([0, 255], None, "GDT_Byte"),
        ([-32768, 32767], None, "GDT_Int16"),
        ([0, 32767], None, "GDT_Int16"),
        ([0, 65535], None, "GDT_UInt16"),
        ([-2147483648, 2147483647], None, "GDT_Int32"),
        ([0, 2147483647], None, "GDT_Int32"),
        ([0, 4294967295], None, "GDT_UInt32"),
        ([-9223372036854775808, 9223372036854775807], None, "GDT_Int64"),
        ([0, 9223372036854775807], None, "GDT_Int64"),
        ([0, 18446744073709551615], None, "GDT_UInt64"),
    ]

    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers = current_test_data[0]
        minimum_gdal_type = current_test_data[1]
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers, minimum_gdal_type_list=minimum_gdal_type
        )
        min_data_type_assertion = current_test_data[2]
        assert min_data_type == min_data_type_assertion, (
            f"min_data_type: {min_data_type} != min_data_type_assertion: {min_data_type_assertion} for test case:\n{current_test_data}"
        )
        numpy_data_type_string = MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]
        np.dtype(numpy_data_type_string)
        min_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[0])
        max_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[1])


def test_minimum_c_data_handler_float_numbers_minimum_data_type():
    test_data_list = (
        ([0.333, 1.777], None, "GDT_Float32"),
        ([0.333, 1], None, "GDT_Float32"),
        ([1 / 3, 1], None, "GDT_Float32"),
        ([3 * 10**37, 1], None, "GDT_Float32"),
        ([3 * 10**40, 1], None, "GDT_Float64"),
    )
    for current_test_data in test_data_list:
        list_of_numbers = current_test_data[0]
        minimum_gdal_type = current_test_data[1]
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers, minimum_gdal_type_list=minimum_gdal_type
        )
        min_data_type_assertion = current_test_data[2]
        assert min_data_type == min_data_type_assertion, (
            f"min_data_type: {min_data_type} != min_data_type_assertion: {min_data_type_assertion} for test case:\n{current_test_data}"
        )
        numpy_data_type_string = MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]
        np.dtype(numpy_data_type_string)
        min_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[0])
        max_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[1])
        pass


def test_other_data_type():
    test_data_list = (
        ([True], None, "GDT_Int8"),
        ([False], None, "GDT_Int8"),
        ([np.inf], None, "GDT_Float32"),
        ([np.nan], None, "GDT_Float32"),
    )
    for current_test_data in test_data_list:
        list_of_numbers = current_test_data[0]
        minimum_gdal_type = current_test_data[1]
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers, minimum_gdal_type_list=minimum_gdal_type
        )
        min_data_type_assertion = current_test_data[2]
        assert min_data_type == min_data_type_assertion, (
            f"min_data_type: {min_data_type} != min_data_type_assertion: {min_data_type_assertion} for test case:\n{current_test_data}"
        )
        numpy_data_type_string = MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]


@pytest.mark.filterwarnings("error: The user-defined:UserWarning")
def test_minimum_c_data_handler_external_minimum_data_type_exact_match():
    test_data_list = (
        ([-128, 127], ["GDT_Int8"], None, "GDT_Int8"),
        ([-128, 127], ["GDT_Int16"], "GDT_Int16", "GDT_Int16"),
        ([0, 127], ["GDT_Byte"], "GDT_Byte", "GDT_Byte"),
        ([0, 127], ["GDT_Byte"], "GDT_Byte", "GDT_Byte"),
        ([0, 127], [], "GDT_Byte", "GDT_Byte"),
        ([0, 127], [], "GDT_Int8", "GDT_Int8"),
        ([0.333, 1.777], ["GDT_Float64"], "GDT_Float64", "GDT_Float64"),
        ([3 * 10**40, 1], ["GDT_Float64"], "GDT_Float64", "GDT_Float64"),
        ([3 * 10**37, 1], ["GDT_Float32"], "GDT_Float32", "GDT_Float32"),
        ([-128, 127], ["GDT_Int8", "GDT_Byte"], "GDT_Int8", "GDT_Int8"),
        ([0, 32767], ["GDT_Int16"], "GDT_Int16", "GDT_Int16"),
        ([0, 32767], ["GDT_UInt16"], "GDT_UInt16", "GDT_UInt16"),
    )
    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers = current_test_data[0]
        minimum_gdal_type = current_test_data[1]
        user_defined_minimum_gdal_type = current_test_data[2]
        min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers,
            minimum_gdal_type_list=minimum_gdal_type,
            user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
        )
        min_data_type_assertion = current_test_data[3]
        assert min_data_type == min_data_type_assertion, (
            f"min_data_type: {min_data_type} != min_data_type_assertion: {min_data_type_assertion} for test case:\n{current_test_data}"
        )
        numpy_data_type_string = MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]
        np.dtype(numpy_data_type_string)
        min_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[0])
        max_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[1])
        pass


def test_minimum_c_data_handler_different_external_minimum_data_type():
    test_data_list = (
        ([-128, 130], ["GDT_Int8"], "GDT_Byte", "GDT_Int16"),
        ([-128, 127], ["GDT_Byte"], "GDT_Byte", "GDT_Int8"),
        ([0, 255], [], "GDT_Int8", "GDT_Byte"),
        ([3 * 10**40, 1], ["GDT_Int8"], "GDT_Int8", "GDT_Float64"),
        ([3 * 10**40, 1], ["GDT_Int16"], "GDT_Float32", "GDT_Float64"),
        ([3 * 10**40, 1], ["GDT_UInt16"], "GDT_Int32", "GDT_Float64"),
        ([3 * 10**40, 1], ["GDT_Int64"], "GDT_Int8", "GDT_Float64"),
        ([3 * 10**37, 1], ["GDT_Int8"], "GDT_UInt64", "GDT_Float32"),
        ([3 * 10**37, 1], ["GDT_Int64"], "GDT_UInt64", "GDT_Float32"),
    )
    for current_test_data in test_data_list:
        print(current_test_data)
        list_of_numbers = current_test_data[0]
        minimum_gdal_type = current_test_data[1]
        user_defined_minimum_gdal_type = current_test_data[2]
        with pytest.warns(UserWarning):
            min_data_type = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(
                list_of_numbers=list_of_numbers,
                minimum_gdal_type_list=minimum_gdal_type,
                user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
            )
            min_data_type_assertion = current_test_data[3]
            assert min_data_type == min_data_type_assertion, (
                f"min_data_type: {min_data_type} != min_data_type_assertion: {min_data_type_assertion} for test case:\n{current_test_data}"
            )
            numpy_data_type_string = MinimumCDataTypeHandler.gdal_to_numpy_data_type_conversion_dict[min_data_type]
            np.dtype(numpy_data_type_string)
            min_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[0])
            max_value_numpy = np.dtype(numpy_data_type_string).type(list_of_numbers[1])


test_minimum_c_data_handler_min_max_values()
test_minimum_c_data_handler_external_minimum_data_type_exact_match()
