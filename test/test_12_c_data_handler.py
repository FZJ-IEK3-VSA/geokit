"""Smoke tests of the deprecated data-type handler.

``MinimumCDataTypeHandler`` is replaced by ``geokit.dtypes`` (ADR 7). It stays importable with a FutureWarning for
one minor release; these tests only make sure that the deprecated path still works and warns.
"""

import importlib
import warnings

import pytest


def test_importing_the_handler_warns_that_it_is_deprecated():
    """Importing geokit.c_data_type_handler issues a FutureWarning that points to geokit.dtypes."""
    with pytest.warns(FutureWarning, match="geokit.dtypes"):
        import geokit.c_data_type_handler as handler_module

        importlib.reload(handler_module)


def test_the_handler_still_chooses_a_type():
    """The deprecated handler still returns a GDAL type name for a list of numbers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from geokit.c_data_type_handler import MinimumCDataTypeHandler

    chosen = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_string(list_of_numbers=[1, 300])

    assert chosen in ("GDT_Int16", "GDT_UInt16")


def test_the_type_aliases_warn_that_they_are_deprecated():
    """The type aliases of geokit.data_types warn on access and still resolve."""
    import geokit.data_types as data_types

    with pytest.warns(FutureWarning, match="dtype_input"):
        alias = data_types.geokit_c_data_types_literal

    assert alias is not None
    with pytest.raises(AttributeError):
        data_types.no_such_alias
