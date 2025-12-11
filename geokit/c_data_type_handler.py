from dataclasses import dataclass
from typing import Literal, Union
import warnings

import numpy as np
from osgeo import gdal
from geokit.error import GeoKitCDataError
from geokit.data_types import (
    gdal_abbreviation_mapper_dict,
    gdal_c_raster_data_types_literal,
    gdal_c_raster_data_types_with_abbreviations_literal,
    geokit_c_data_types_literal,
    integer_data_types_literal,
    _gdal_c_raster_data_types_list,
    numpy_data_types_list_literal,
)


@dataclass
class CIntegerDataTypeInfo:
    (
        """Class to hold information about C integer data types.
    It contains the data """
        ""
    )

    gdal_data_type_string: str
    numpy_data_type_string: str
    bits: int
    signed: bool

    def __post_init__(self):
        if self.signed is True:
            self.min_value: int = self.min_value_int_signed()
            self.max_value: int = self.max_value_int_signed()
        else:
            self.min_value: int = self.min_value_int_unsigned()
            self.max_value: int = self.max_value_int_unsigned()

    def max_value_int_signed(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def min_value_int_signed(
        self,
    ) -> int:
        return -(2 ** (self.bits - 1))

    def max_value_int_unsigned(self) -> int:
        return 2 ** (self.bits) - 1

    def min_value_int_unsigned(
        self,
    ) -> int:
        return 0

    def get_unsigned_gdal_integer_string(self) -> str:
        if self.signed is True:
            unsigned_string = self.gdal_data_type_string
        elif "GDT_UInt" in self.gdal_data_type_string:
            unsigned_string = self.gdal_data_type_string.replace("GDT_UInt", "GDT_Int")
        elif "GDT_Byte" in self.gdal_data_type_string:
            return "GDT_Int8"
        else:
            raise GeoKitCDataError(f"Could not convert {self.gdal_data_type_string} to unsigned integer string.")

        return unsigned_string


@dataclass
class CFloatDataTypeInfo:
    gdal_data_type_string: str
    numpy_data_type_string: str
    bits: int


class MinimumCDataTypeHandler:
    int8bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_Int8",
        numpy_data_type_string="int8",
        bits=8,
        signed=True,
    )
    unint8bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_Byte",
        numpy_data_type_string="uint8",
        bits=8,
        signed=False,
    )
    int16bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_Int16",
        numpy_data_type_string="int16",
        bits=16,
        signed=True,
    )
    unint16bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_UInt16", numpy_data_type_string="uint16", bits=16, signed=False
    )
    int32bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_Int32", numpy_data_type_string="int32", bits=32, signed=True
    )
    unint32bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_UInt32", numpy_data_type_string="uint32", bits=32, signed=False
    )
    int64bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_Int64", numpy_data_type_string="int64", bits=64, signed=True
    )
    unint64bit: CIntegerDataTypeInfo = CIntegerDataTypeInfo(
        gdal_data_type_string="GDT_UInt64", numpy_data_type_string="uint64", bits=64, signed=False
    )
    gdal_implemented_integer_data_types_dict: dict[str, CIntegerDataTypeInfo] = {
        int8bit.gdal_data_type_string: int8bit,
        unint8bit.gdal_data_type_string: unint8bit,
        int16bit.gdal_data_type_string: int16bit,
        unint16bit.gdal_data_type_string: unint16bit,
        int32bit.gdal_data_type_string: int32bit,
        unint32bit.gdal_data_type_string: unint32bit,
        int64bit.gdal_data_type_string: int64bit,
        unint64bit.gdal_data_type_string: unint64bit,
    }
    gdal_implemented_integer_data_types_list: list[CIntegerDataTypeInfo] = [
        *gdal_implemented_integer_data_types_dict.values()
    ]
    bit_to_gdal_name_int: dict[int, str] = {
        int8bit.bits: int8bit.gdal_data_type_string,
        unint8bit.bits: unint8bit.gdal_data_type_string,
        int16bit.bits: int16bit.gdal_data_type_string,
        unint16bit.bits: unint16bit.gdal_data_type_string,
        int32bit.bits: int32bit.gdal_data_type_string,
        unint32bit.bits: unint32bit.gdal_data_type_string,
        int64bit.bits: int64bit.gdal_data_type_string,
        unint64bit.bits: unint64bit.gdal_data_type_string,
    }
    float32bit: CFloatDataTypeInfo = CFloatDataTypeInfo(
        gdal_data_type_string="GDT_Float32", numpy_data_type_string="float32", bits=32
    )
    float64bit: CFloatDataTypeInfo = CFloatDataTypeInfo(
        gdal_data_type_string="GDT_Float64", numpy_data_type_string="float64", bits=64
    )
    gdal_implemented_float_dict: dict[str, CFloatDataTypeInfo] = {
        float32bit.gdal_data_type_string: float32bit,
        float64bit.gdal_data_type_string: float64bit,
    }

    gdal_implemented_float_list: list[CFloatDataTypeInfo] = [*gdal_implemented_float_dict.values()]
    bit_to_gdal_name_float: dict[int, str] = {
        float32bit.bits: float32bit.gdal_data_type_string,
        float64bit.bits: float64bit.gdal_data_type_string,
    }
    gdal_implemented_complex_list = []
    numpy_data_to_gdal_type_conversion_dict = {
        "bool": "GDT_Int8",
        "uint8": "GDT_Byte",
        "int8": "GDT_Int8",
        "uint16": "GDT_UInt16",
        "int16": "GDT_Int16",
        "uint32": "GDT_UInt32",
        "int32": "GDT_Int32",
        "uint64": "GDT_UInt64",
        "int64": "GDT_Int64",
        "float16": "GDT_Float32",
        "float32": "GDT_Float32",
        "float64": "GDT_Float64",
        # "float96": None,
        # "float128": None,
        # "complex64": "GDT_CFloat64",
        # "complex128": None,
        # "complex192": None,
        # "complex256": None,
        # "object": None,
    }

    gdal_to_numpy_data_type_conversion_dict = {
        "GDT_Byte": "uint8",
        "GDT_Int8": "int8",
        "GDT_UInt16": "uint16",
        "GDT_Int16": "int16",
        "GDT_UInt32": "uint32",
        "GDT_Int32": "int32",
        "GDT_UInt64": "uint64",
        "GDT_Int64": "int64",
        "GDT_Float32": "float32",
        "GDT_Float64": "float64",
        "GDT_CFloat32": "complex64",
        "GDT_CFloat64": "complex64",
    }

    @classmethod
    def _get_valid_data_types_integer(cls, int_to_check: int | np.integer) -> list[str]:
        valid_data_types_list: list[str] = []
        for relevant_data_type in cls.gdal_implemented_integer_data_types_list:
            if relevant_data_type.min_value <= int_to_check <= relevant_data_type.max_value:
                valid_data_types_list.append(relevant_data_type.gdal_data_type_string)
        return valid_data_types_list

    @classmethod
    def get_highest_common_integer_type(
        cls, list_of_integer_strings: list[integer_data_types_literal]
    ) -> CIntegerDataTypeInfo:
        list_of_bits: list[int] = []
        for current_integer_string in list_of_integer_strings:
            int_d_type_class = cls.gdal_implemented_integer_data_types_dict[current_integer_string]
            list_of_bits.append(int_d_type_class.bits)
        highest_common_bits = max(list_of_bits)
        highest_common_data_type_string = cls.bit_to_gdal_name_int[highest_common_bits]
        highest_common_integer_type = cls.gdal_implemented_integer_data_types_dict[highest_common_data_type_string]
        return highest_common_integer_type

    @classmethod
    def _get_valid_data_types_from_integer_list(
        cls,
        int_list_to_check: list[int | np.integer | bool],
        minimum_integer_type_list: list[integer_data_types_literal] | None,
        user_defined_minimum_gdal_type: integer_data_types_literal | None,
    ) -> str:
        valid_data_types_list: list[str] = []
        for relevant_data_type in cls.gdal_implemented_integer_data_types_list:
            if cls._all_between(
                nums=int_list_to_check, lower=relevant_data_type.min_value, upper=relevant_data_type.max_value
            ):
                valid_data_types_list.append(relevant_data_type.gdal_data_type_string)
        if not valid_data_types_list:
            raise GeoKitCDataError(f"Could find a common data type to display {int_list_to_check} in the same raster")

        # if isinstance(user_defined_minimum_gdal_type,str):
        if isinstance(minimum_integer_type_list, list):
            minimum_data_type_info = cls.gdal_implemented_integer_data_types_dict[valid_data_types_list[0]]
            data_types_to_consider: list[str] = [valid_data_types_list[0], *minimum_integer_type_list]
            highest_common_integer_type = cls.get_highest_common_integer_type(
                list_of_integer_strings=data_types_to_consider
            )

            if highest_common_integer_type.bits > minimum_data_type_info.bits:
                output_data_type_string = highest_common_integer_type.get_unsigned_gdal_integer_string()

            else:
                output_data_type_string = minimum_data_type_info.gdal_data_type_string
        else:
            output_data_type_string = valid_data_types_list[0]

        if user_defined_minimum_gdal_type in valid_data_types_list:
            user_requested_bits = cls.gdal_implemented_integer_data_types_dict[output_data_type_string].bits
            required_bit_size = cls.gdal_implemented_integer_data_types_dict[output_data_type_string].bits
            if user_requested_bits == required_bit_size:
                output_data_type_string = user_defined_minimum_gdal_type

        return output_data_type_string

    @staticmethod
    def _all_between(nums: list[int | np.integer], lower: int | np.integer, upper: int | np.integer):
        return all(lower <= n <= upper for n in nums)

    @classmethod
    def _get_minimum_float_type(
        cls,
        list_of_float_data_types: list[np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType],
        list_of_integer: list[int | np.integer],
        minimum_required_datatype_list: None | list[gdal_c_raster_data_types_literal],
    ) -> str:
        list_of_bits: list[int] = []
        for current_integer in list_of_integer:
            current_converted_float = float(current_integer)
            list_of_float_data_types.append(np.min_scalar_type(current_converted_float))

        if isinstance(minimum_required_datatype_list, list):
            for current_datatype in minimum_required_datatype_list:
                if current_datatype in cls.gdal_implemented_integer_data_types_dict:
                    pass
                elif current_datatype in cls.gdal_implemented_float_dict:
                    bits_of_minimum_required_data_type = cls.gdal_implemented_float_dict[current_datatype].bits
                    list_of_bits.append(bits_of_minimum_required_data_type)
        for current_float_data_type in list_of_float_data_types:
            float_type_numpy_string = str(current_float_data_type)
            float_gdal_string = cls.numpy_data_to_gdal_type_conversion_dict[float_type_numpy_string]
            bits_of_current_float_type = cls.gdal_implemented_float_dict[float_gdal_string].bits
            list_of_bits.append(bits_of_current_float_type)
        if not list_of_float_data_types:
            return cls.float64bit.gdal_data_type_string
        required_bit_length = max(list_of_bits)
        required_data_type = cls.bit_to_gdal_name_float[required_bit_length]
        return required_data_type

    @classmethod
    def _determine_if_gdal_data_type_is_int_float_or_complex(
        cls, gdal_data_type_string_list: list[gdal_c_raster_data_types_literal]
    ) -> Literal["GDT_Byte", "GDT_Int", "GDT_Float", "GDT_CInt", "GDT_CFloat"]:
        output_data_type_list: list[Literal["GDT_Byte", "GDT_Int", "GDT_Float", "GDT_CInt", "GDT_CFloat"]] = []
        for current_data_type_to_check in gdal_data_type_string_list:
            if current_data_type_to_check[0:8] == "GDT_Byte":
                current_data_type = "GDT_Int"
            elif current_data_type_to_check[0:7] == "GDT_Int" or current_data_type_to_check[0:8] == "GDT_UInt":
                current_data_type = "GDT_Int"
            elif current_data_type_to_check[0:9] == "GDT_Float":
                current_data_type = "GDT_Float"
            elif current_data_type_to_check[0:8] == "GDT_CInt":
                current_data_type = "GDT_CInt"
            elif current_data_type_to_check[0:10] == "GDT_CFloat":
                current_data_type = "GDT_CFloat"
            else:
                raise GeoKitCDataError(
                    f"Could not identify string as Byte, int, float, complex integer or complex float. Got: {current_data_type_to_check}"
                )
            output_data_type_list.append(current_data_type)
        if "GDT_CFloat" in output_data_type_list:
            return "GDT_CFloat"
        if "GDT_CInt" in output_data_type_list:
            return "GDT_CInt"
        if "GDT_Float" in output_data_type_list:
            return "GDT_Float"
        if "GDT_Int" in output_data_type_list:
            return "GDT_Int"
        if "GDT_Int" in output_data_type_list:
            return "GDT_Int"
        raise GeoKitCDataError(
            f"Could not identify string as Byte, int, float, complex integer or complex float. Got: {output_data_type_list}"
        )

    @classmethod
    def _check_if_integers_can_be_displayed_as_integers(
        cls, list_of_numbers: list[int | np.integer]
    ) -> tuple[list[np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType], list[int | np.integer]]:
        list_of_output_integers: list[int | np.integer] = []
        list_of_output_floats: list[np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType] = []
        if all(list_of_numbers) >= 0:
            max_integer_site = cls.unint64bit.max_value
        else:
            max_integer_site = cls.int64bit.max_value
        for current_number in list_of_numbers:
            if abs(current_number) <= max_integer_site:
                list_of_output_integers.append(current_number)
            else:
                list_of_output_floats.append(np.min_scalar_type(float(current_number)))
        return list_of_output_floats, list_of_output_integers

    @classmethod
    def _convert_abbreviation_to_gdal_data_type(
        cls,
        input_string_list: list[geokit_c_data_types_literal] | None,
    ) -> list[gdal_c_raster_data_types_literal] | None:
        """Converts abbreviactions and numpy repsentation of gdal datat types to rigorous gdal datatypes.

        Parameters
        ----------
        input_string_list : list[geokit_c_data_types_literal] | None
            Data types provided as abbreviations, numpy data type strings or rigorous gdal data type strings.

        Returns
        -------
        list[gdal_c_raster_data_types_literal] | None
            Rigorous string repesentation of gdal data types.

        """
        if input_string_list is None:
            return input_string_list
        output_gdal_data_type_string_list: list[gdal_c_raster_data_types_literal] = []
        for input_string in input_string_list:
            if input_string in gdal_abbreviation_mapper_dict:
                gdal_data_type_string = gdal_abbreviation_mapper_dict[input_string]
                output_gdal_data_type_string_list.append(gdal_data_type_string)
            elif input_string in _gdal_c_raster_data_types_list:
                output_gdal_data_type_string_list.append(input_string)
            elif input_string in cls.numpy_data_to_gdal_type_conversion_dict:
                gdal_data_type_string = cls.numpy_data_to_gdal_type_conversion_dict[input_string]
                output_gdal_data_type_string_list.append(gdal_data_type_string)
            else:
                raise GeoKitCDataError(
                    f"Got invalid gdal data type or abbreviation: {input_string}. Expected one of {geokit_c_data_types_literal}"
                )
        return output_gdal_data_type_string_list

    @classmethod
    def check_if_user_requested_data_type_deviates(
        cls,
        user_defined_minimum_gdal_type: geokit_c_data_types_literal | None,
        automatically_determined_minimum_gdal_type: gdal_c_raster_data_types_literal,
    ):
        """Checks if the user defined minimum gdal data type deviates from the automatically determined minimum gdal data type. Raises a warning if the two data types differ.

        Parameters
        ----------
        user_defined_minimum_gdal_type : geokit_c_data_types_literal | None
            The user defined minimum gdal data type.
        automatically_determined_minimum_gdal_type : gdal_c_raster_data_types_literal
            The automatically determined minimum gdal data type.
        """
        if user_defined_minimum_gdal_type is not None:
            converted_user_defined_minimum_gdal_type_list = cls._convert_abbreviation_to_gdal_data_type(
                input_string_list=[user_defined_minimum_gdal_type]
            )
            converted_user_defined_minimum_gdal_type = converted_user_defined_minimum_gdal_type_list[0]
            if automatically_determined_minimum_gdal_type != converted_user_defined_minimum_gdal_type:
                warning_message = (
                    f"The user-defined minimum GDAL data type: {user_defined_minimum_gdal_type},"
                    f" which is understood as the rigorous GDAL datatype: {converted_user_defined_minimum_gdal_type},"
                    f" differs from automatically determined data type: {automatically_determined_minimum_gdal_type}."
                    " To silence this warning check the configuration of your function call for configurations that might cause an unintentional overflow error."
                    " Otherwise you can just set the data type to the automatically determined data type or to the Python object None."
                )
                warnings.warn(
                    message=warning_message,
                    category=UserWarning,
                )

    @classmethod
    def get_valid_gdal_data_type_as_string(
        cls,
        list_of_numbers: list[float | int | np.integer | np.floating | np.complexfloating | bool | np.bool_],
        minimum_gdal_type_list: list[geokit_c_data_types_literal] | None = None,
        user_defined_minimum_gdal_type: geokit_c_data_types_literal | None = None,
    ) -> str:
        """This Function determines the minimum required data type to store all numbers provided list_of_numbers in the same data type.

        Additionally the user can provide a list of minimum required data types that need to be considered as well as a user defined minimum data type.
        The data types in the gdal_type_list can be provided to get at least the data type provided in the list or a bigger one if needed.
        The user defined minimum data type is only used to warn the user if the automatically determined data type differs from the user defined one.
        This is useful identify a potential misconfiguration.

        Parameters
        ----------
        list_of_numbers : list[float  |  int  |  np.integer  |  np.floating  |  np.complexfloating  |  bool  |  np.bool_]
            A list of numbers for which the minimum required gdal data type should be determined.
        minimum_gdal_type_list : list[geokit_c_data_types_literal] | None, optional
            This is a list of the minimum required GDAL data types that need to be considered. If a larger data type is required, the input is ignored.
            If no data types are provided, they are automatically determined from the list of numbers. by default None
        user_defined_minimum_gdal_type : geokit_c_data_types_literal | None, optional
            The data type that the user expects. If a different data type is determined automatically a warning is raised. If set to None no warning is raised., by default None

        Returns
        -------
        str
            A a string of the number data type that can be used in gdal to store all numbers provided in list_of_numbers.

        """
        minimum_gdal_type_list_converted = cls._convert_abbreviation_to_gdal_data_type(
            input_string_list=minimum_gdal_type_list
        )
        if user_defined_minimum_gdal_type is None:
            user_defined_minimum_gdal_type_converted = cls._convert_abbreviation_to_gdal_data_type(
                input_string_list=user_defined_minimum_gdal_type
            )
        elif isinstance(user_defined_minimum_gdal_type, str):
            user_defined_minimum_gdal_type_converted = cls._convert_abbreviation_to_gdal_data_type(
                input_string_list=[user_defined_minimum_gdal_type]
            )
        else:
            raise GeoKitCDataError(
                "For argument: user_defined_minimum_gdal_type either a string with a valid Gdal datatype or None is allowed. However the following type has been provided: {user_defined_minimum_gdal_type}"
            )

        list_of_integer: list[int | np.integer] = []
        list_of_float_data_types: list[np.dtype] = []
        list_of_complex_floats = []
        for current_number in list_of_numbers:
            # if type(current_number) == type(bool):
            if isinstance(current_number, (bool, np.bool_)):
                if isinstance(minimum_gdal_type_list, list):
                    minimum_gdal_type_list.append("GDT_Int8")
                else:
                    minimum_gdal_type_list = ["GDT_Int8"]
            elif isinstance(current_number, (int, np.integer)):
                list_of_integer.append(current_number)
            elif np.isnan(current_number) or np.isinf(current_number):
                minimum_data_type_float = np.dtype("float16")
                list_of_float_data_types.append(minimum_data_type_float)
            elif isinstance(current_number, float):
                integer_float: int = int(current_number)
                if current_number == integer_float:
                    list_of_integer.append(integer_float)
                else:
                    list_of_float_data_types.append(np.min_scalar_type(current_number))
            elif isinstance(current_number, np.number):
                if np.issubdtype(current_number, np.floating):
                    list_of_float_data_types.append(np.min_scalar_type(current_number))
                elif np.issubdtype(current_number, np.complexfloating):
                    list_of_complex_floats.append(current_number)
                else:
                    raise GeoKitCDataError(
                        f"GeoKit only supports the integers, floats and complex floats as numpy data types. Nonetheless the following data type has been provided {type(current_number)} "
                    )
            else:
                raise GeoKitCDataError(
                    f"GeoKit only supports the integers, floats and complex floats as data types. Nonetheless the following data type has been provided {type(current_number)} "
                )
        list_of_output_float_data_types, list_of_output_integers = cls._check_if_integers_can_be_displayed_as_integers(
            list_of_numbers=list_of_integer
        )

        list_of_float_data_types.extend(list_of_output_float_data_types)

        if isinstance(minimum_gdal_type_list_converted, list):
            if minimum_gdal_type_list_converted:
                c_number_category = cls._determine_if_gdal_data_type_is_int_float_or_complex(
                    minimum_gdal_type_list_converted
                )
            else:
                c_number_category = None
        elif minimum_gdal_type_list_converted is None:
            c_number_category = None
        else:
            raise GeoKitCDataError(
                "For argument: minimum_gdal_type either a list of strings with valid Gdal datatypes or None is allowed. However the following type has been provided: {minimum_gdal_type}"
            )

        if list_of_complex_floats or c_number_category == "GDT_CInt" or c_number_category == "GDT_CFloat":
            raise GeoKitCDataError("Complex Data Types are not implemented yet.")
        if not list_of_float_data_types and not list_of_output_integers and c_number_category is None:
            cls.check_if_user_requested_data_type_deviates(
                user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
                automatically_determined_minimum_gdal_type=cls.int8bit.gdal_data_type_string,
            )
            return cls.int8bit.gdal_data_type_string
        if list_of_float_data_types or c_number_category == "GDT_Float":
            minimum_float_type = cls._get_minimum_float_type(
                list_of_float_data_types=list_of_float_data_types,
                list_of_integer=list_of_integer,
                minimum_required_datatype_list=minimum_gdal_type_list_converted,
            )
            cls.check_if_user_requested_data_type_deviates(
                user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
                automatically_determined_minimum_gdal_type=minimum_float_type,
            )
            return minimum_float_type
        if list_of_output_integers or c_number_category == "GDT_Int":
            if isinstance(user_defined_minimum_gdal_type_converted, list):
                user_defined_c_number_category = cls._determine_if_gdal_data_type_is_int_float_or_complex(
                    user_defined_minimum_gdal_type_converted
                )
                if user_defined_c_number_category == "GDT_Int":
                    user_defined_minimum_gdal_type_int = user_defined_minimum_gdal_type
                else:
                    user_defined_minimum_gdal_type_int = None
            else:
                user_defined_minimum_gdal_type_int = None

            minimum_integer_type = cls._get_valid_data_types_from_integer_list(
                int_list_to_check=list_of_output_integers,
                minimum_integer_type_list=minimum_gdal_type_list_converted,
                user_defined_minimum_gdal_type=user_defined_minimum_gdal_type_int,
            )
            cls.check_if_user_requested_data_type_deviates(
                user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
                automatically_determined_minimum_gdal_type=minimum_integer_type,
            )
            return minimum_integer_type
        else:
            raise GeoKitCDataError("No datatypes could be determined.")

    @staticmethod
    def get_gdal_constant_from_string(input_string: str) -> int:
        """This function converts a gdal data type string to the corresponding gdal constant that repsresents the data type.

        Parameters
        ----------
        input_string : str
            String that should be converted to gdal constant.

        Returns
        -------
        int
            Integer representing the gdal data type. This can be passed to gdal functions that require a data type constant.

        """
        if input_string[0:4] == "GDT_":
            constant_string = input_string
        else:
            constant_string = "GDT_" + input_string

        if hasattr(gdal, constant_string):
            gdal_data_type_constant = getattr(gdal, constant_string)
            return gdal_data_type_constant
        else:
            raise ValueError(f"Data type {input_string} is not a valid GDAL data type")

    @classmethod
    def get_valid_gdal_data_type_as_constant(
        cls,
        list_of_numbers: list[float | int | np.integer | np.floating | np.complexfloating | bool | np.bool_],
        minimum_gdal_type_list: list[geokit_c_data_types_literal] | None = None,
        user_defined_minimum_gdal_type: geokit_c_data_types_literal | None = None,
    ) -> int:
        """This Function determines the minimum required data type to store all numbers provided list_of_numbers in the same data type.

        Additionally the user can provide a list of minimum required data types that need to be considered as well as a user defined minimum data type.
        The data types in the gdal_type_list can be provided to get at least the data type provided in the list or a bigger one if needed.
        The user defined minimum data type is only used to warn the user if the automatically determined data type differs from the user defined one.
        This is useful identify a potential misconfiguration.

        Parameters
        ----------
        list_of_numbers : list[float  |  int  |  np.integer  |  np.floating  |  np.complexfloating  |  bool  |  np.bool_]
            A list of numbers for which the minimum required gdal data type should be determined.
        minimum_gdal_type_list : list[geokit_c_data_types_literal] | None, optional
            This is a list of the minimum required GDAL data types that need to be considered. If a larger data type is required, the input is ignored.
            If no data types are provided, they are automatically determined from the list of numbers. by default None
        user_defined_minimum_gdal_type : geokit_c_data_types_literal | None, optional
            The data type that the user expects. If a different data type is determined automatically a warning is raised. If set to None no warning is raised., by default None

        Returns
        -------
        int
            An integer that represents represents a C datatype that must be passed to certain GDAL functions. To convert it to a human readable string use the function get_gdal_constant_from_string
            or use the get_valid_gdal_data_type_as_string instead of this function.
        """
        data_type_string = cls.get_valid_gdal_data_type_as_string(
            list_of_numbers=list_of_numbers,
            minimum_gdal_type_list=minimum_gdal_type_list,
            user_defined_minimum_gdal_type=user_defined_minimum_gdal_type,
        )
        data_type_constant = cls.get_gdal_constant_from_string(input_string=data_type_string)
        return data_type_constant
