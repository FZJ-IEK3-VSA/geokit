import warnings

warnings.warn(
    "Importing from 'geokit.core.geom' is deprecated and will be removed in a future version. "
    "Use 'from geokit.geom import ...' or 'from geokit import geom' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.geom import *  # noqa: F401, F403, E402
from geokit.geom import POINT, MULTIPOINT, LINE, MULTILINE, POLYGON, MULTIPOLYGON  # noqa: F401, E402
