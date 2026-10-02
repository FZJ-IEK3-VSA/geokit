"""Smoke tests of the deprecated data-type handler.

``MinimumCDataTypeHandler`` is replaced by ``geokit.dtypes`` (ADR 7). It stays importable with a FutureWarning for
one minor release; these tests only make sure that the deprecated path still works and warns.
"""

import importlib
import warnings

import pytest

# the names geokit.data_types keeps for one minor release, each with a FutureWarning
DEPRECATED_TYPE_ALIASES = [
    "float_data_types_literal",
    "float_data_types_with_abbreviations_literal",
    "gdal_abbreviation_mapper_dict",
    "gdal_c_raster_data_types_literal",
    "gdal_c_raster_data_types_with_abbreviations_literal",
    "geokit_c_data_types_literal",
    "integer_data_types_literal",
    "integer_data_types_with_abbreviations_literal",
    "numpy_data_types_list_literal",
]


def test_importing_the_handler_warns_that_it_is_deprecated():
    """Importing geokit.c_data_type_handler issues a FutureWarning that points to geokit.dtypes."""
    with pytest.warns(FutureWarning, match="geokit.dtypes"):
        import geokit.c_data_type_handler as handler_module

        importlib.reload(handler_module)


def test_the_handler_still_chooses_a_type():
    """The deprecated handler still returns the GDAL type name it chose before: Int16 for 1 and 300."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from geokit.c_data_type_handler import MinimumCDataTypeHandler

    chosen = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(list_of_numbers=[1, 300])

    assert chosen == "GDT_Int16"


@pytest.mark.parametrize("alias_name", DEPRECATED_TYPE_ALIASES)
def test_the_type_aliases_warn_that_they_are_deprecated(alias_name):
    """Each type alias of the former handler still resolves, with a FutureWarning that points to dtype_input."""
    import geokit.data_types as data_types

    with pytest.warns(FutureWarning, match="dtype_input"):
        alias = getattr(data_types, alias_name)

    assert alias is not None


def test_an_unknown_name_of_geokit_data_types_raises_attribute_error():
    """A name that geokit.data_types does not have raises AttributeError, as for any module."""
    import geokit.data_types as data_types

    with pytest.raises(AttributeError):
        data_types.no_such_alias
