"""The GeoKit library is a collection of general geospatial operations"""

# maybe set GDAL_DATA variable
import pathlib
from os import environ as _environ
from os.path import basename as _basename
from os.path import dirname as _dirname
from os.path import join as _join

# from warnings import warn as _warn

if not "GDAL_DATA" in _environ:
    from os.path import isdir as _isdir
    from sys import executable as _executable

    for d in [
        _join(
            _dirname(_executable), "Library", "share", "gdal"
        ),  # Common location on windows
        _join(_dirname(_executable), "..", "share", "gdal"),
    ]:  # Common location on linux
        if _isdir(d):
            # _warn("Setting GDAL_DATA to: "+d, UserWarning)
            _environ["GDAL_DATA"] = d
            break

    if not "GDAL_DATA" in _environ:
        raise RuntimeError(
            "Could not locate GDAL_DATA folder. Please set this as an environment variable pointing to the GDAL static files"
        )


# Add useful paths for testing and stuff
from collections import OrderedDict as _OrderedDict
from glob import glob as _glob

# import the special algorithms
import geokit.algorithms
import geokit.geom
import geokit.raster
import geokit.srs

# import the utilities
import geokit.util
import geokit.vector
from geokit.core.extent import Extent
from geokit.core.geom import drawGeoms
from geokit.core.get_test_data import get_all_test_data_dict

# import the main objects
from geokit.core.location import Location, LocationSet
from geokit.core.raster import drawRaster, drawSmopyMap
from geokit.core.regionmask import RegionMask

# import vidualizing functions to top level since they are
from geokit.core.util import drawImage

_test_data_ = get_all_test_data_dict()
