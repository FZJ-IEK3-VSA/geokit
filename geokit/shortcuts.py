"""
A module which automatically imports all geokit modules under shorter names

Map:
	geokit.util -> gku
	geokit.srs -> gks
	geokit.geom -> gkg
	geokit.raster -> gkr
	geokit.vector -> gkv
	geokit.RegionMask -> RegionMask
	geokit.Extent -> Extent
	geokit.algorithms -> gk
"""

import geokit.util as gku
import geokit.srs  as gks
import geokit.geom  as gkg
import geokit.raster  as gkr
import geokit.vector  as gkv
from geokit._core.extent import Extent
from geokit._core.regionmask import RegionMask
import geokit.algorithms as gka