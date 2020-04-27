"""The GeoKit library is a collection of general geospatial operations"""

__version__ = "1.2.4"

# maybe set GDAL_DATA variable
from os import environ as _environ
from os.path import join as _join, dirname as _dirname, basename as _basename
# from warnings import warn as _warn

if not "GDAL_DATA" in _environ:
    from os.path import isdir as _isdir
    from sys import executable as _executable

    for d in [_join(_dirname(_executable), "Library", "share", "gdal"),  # Common location on windows
              _join(_dirname(_executable), "..", "share", "gdal")]:  # Common location on linux
        if _isdir(d):
            # _warn("Setting GDAL_DATA to: "+d, UserWarning)
            _environ["GDAL_DATA"] = d
            break

    if not "GDAL_DATA" in _environ:
        raise RuntimeError(
            "Could not locate GDAL_DATA folder. Please set this as an environment variable pointing to the GDAL static files")


# import the utilities
import geokit.util
import geokit.srs
import geokit.geom
import geokit.raster
import geokit.vector

# import the main objects
from geokit.core.location import Location, LocationSet
from geokit.core.extent import Extent
from geokit.core.regionmask import RegionMask

# import vidualizing functions to top level since they are
from geokit.core.util import drawImage
from geokit.core.geom import drawGeoms
from geokit.core.raster import drawRaster

# import the special algorithms
import geokit.algorithms

# Add useful paths for testing and stuff
from collections import OrderedDict as _OrderedDict
from glob import glob as _glob
_test_data_ = _OrderedDict()

for f in _glob(_join(_dirname(__file__), "test", "data", "*")):
    _test_data_[_basename(f)] = f
