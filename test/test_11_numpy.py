# import numpy as np


def get_datype():
    # Example 1: Integers and Floats
    inputs1 = [10, 3.14, 255]
    # The result will be float64 because it needs to hold 3.14
    common_type1 = np.array(inputs1).dtype

    # Example 2: Small vs. Large Integers
    inputs2 = [5, 65536, -1]
    # The result will be int32 (or int64 depending on the system/NumPy config)
    # because int16 (max 32767) is too small for 65536.
    common_type2 = np.array(inputs2).dtype

    # Example 3: Mixed types including a complex number
    inputs3 = [1, 2.0, 3 + 4j]
    # The result will be complex128
    common_type3 = np.array(inputs3).dtype

    print(f"Inputs 1 Common Type: {common_type1}")
    print(f"Inputs 2 Common Type: {common_type2}")
    print(f"Inputs 3 Common Type: {common_type3}")


# def test_min_max_values_np_data_types():
#     min_max_data_type_dictionary = {}
#     for current_data_type in integer_data_types:
#         current_lower_data = str(current_data_type).lower()
#         numpy_data_type = getattr(np, current_lower_data)
#         min_max_data_type_dictionary[current_lower_data] = {}
#         min_max_data_type_dictionary[current_lower_data]["min"] = np.iinfo(numpy_data_type).min
#         min_max_data_type_dictionary[current_lower_data]["max"] = np.iinfo(numpy_data_type).max

#     for current_data_type in float_data_types:
#         current_lower_data = str(current_data_type).lower()
#         numpy_data_type = getattr(np, current_lower_data)
#         min_max_data_type_dictionary[current_lower_data] = {}
#         min_max_data_type_dictionary[current_lower_data]["min"] = np.finfo(numpy_data_type).min
#         min_max_data_type_dictionary[current_lower_data]["max"] = np.finfo(numpy_data_type).max

#     print(min_max_data_type_dictionary)

#     min_max_data_type_dictionary_assertion = {
#         "int8": {"min": -128, "max": 127},
#         "uint16": {"min": 0, "max": 65535},
#         "int16": {"min": -32768, "max": 32767},
#         "uint32": {"min": 0, "max": 4294967295},
#         "int32": {"min": -2147483648, "max": 2147483647},
#         "uint64": {"min": 0, "max": 18446744073709551615},
#         "int64": {"min": -9223372036854775808, "max": 9223372036854775807},
#         "float32": {"min": np.float32(-3.4028235e38), "max": np.float32(3.4028235e38)},
#         "float64": {"min": np.float64(-1.7976931348623157e308), "max": np.float64(1.7976931348623157e308)},
#     }
#     assert min_max_data_type_dictionary == min_max_data_type_dictionary_assertion


# def test_get_min_datatype():
#     list_of_datatypes_compare = ["uint16", "float64"]


# def test_result_type():
#     list_of_list = [
#         (1, 1),
#         (1.5, 2.5),
#         (1, 2.5),
#     ]
#     for current_type_list in list_of_list:
#         b = np.ndarray(current_type_list)
#         a = b.dtype
#         # a = np.min_scalar_type(current_type_list)
#         print(a)


# def promote_dtype():
#     """Promote two numpy dtypes to a common dtype."""
#     list_of_list = [
#         ("int8", "int16", "int16"),
#         ("int8", "uint8", "uint16"),
#         ("int16", "uint16", "uint16"),
#         ("int32", "uint32", "uint32"),
#         ("int64", "uint64", "uint64"),
#         ("float32", "float64", "float64"),
#         ("int32", "float32", "float32"),
#         ("uint32", "float32", "float32"),
#         # (1.5, 2.5),
#         # (1, 2.5),
#     ]
#     for current_type_list in list_of_list:
#         a = np.promote_types(current_type_list[0], current_type_list[1])

#         # a = np.min_scalar_type(current_type_list)
#         print(a)

#         try:
#             assert a == current_type_list[2]
#         except AssertionError as e:
#             e.args += (a, "!=", current_type_list[2])
#             raise


# def test_numpy_min_scalar():
#     test_data = [
#         ("int8", -128),
#         ("int16", -32768),
#         ("uint16", 65535),
#         ("int32", -2147483648),
#         ("uint32", 4294967295),
#         ("int64", -9223372036854775808),
#         ("uint64", 18446744073709551615),
#         # ("float32", -3.4028235e39), # is bugged https://github.com/numpy/numpy/issues/30342
#         ("float32", -3.0e39),
#         ("float64", -1.7976931348623157e308),
#     ]
#     for current_test_data in test_data:
#         numpy_datatype_min = np.min_scalar_type(current_test_data[1])

#         numpy_datatype_str_min = str(numpy_datatype_min)

#         assert current_test_data[0] == numpy_datatype_str_min
