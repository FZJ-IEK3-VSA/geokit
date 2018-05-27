"""The GeoKit library is a collection of general geospatial operations"""

__version__ = "1.1.1"

# maybe set GDAL_DATA variable
from os import environ as _environ
from os.path import join as _join, dirname as _dirname, basename as _basename

if not "GDAL_DATA" in _environ:
	from os.path import isdir as _isdir
	from sys import executable as _executable
	
	testDir = _join(_executable,"..","Library","share","gdal")

	if not _isdir(testDir):
		raise RuntimeError("Could not locate GDAL data folder at :", testDir)
	_environ["GDAL_DATA"] = testDir

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
#import geokit.algorithms

# Add useful paths for testing and stuff
from collections import OrderedDict as _OrderedDict
from glob import glob as _glob
_test_data_ = _OrderedDict()

for f in _glob(_join(_dirname(__file__), "test", "data", "*")):
    _test_data_[_basename(f)] = f
