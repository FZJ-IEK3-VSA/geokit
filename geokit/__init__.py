"""The GeoKit library!"""

#__version__ = 0.0.2

# import the utilities
import geokit.util
import geokit.srs
import geokit.geom
import geokit.raster
import geokit.vector

# import the main objects
from geokit._core.location import Location
from geokit._core.extent import Extent
from geokit._core.regionmask import RegionMask

# import vidualizing functions to top level since they are 
from geokit._core.rasterutil import drawImage
from geokit._core.geomutil import drawGeoms

# import the special algorithms
import geokit.algorithms

# Import the CLI
import __main__
