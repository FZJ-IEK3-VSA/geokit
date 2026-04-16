import warnings

warnings.warn(
    "Importing from 'geokit.core.raster' is deprecated and will be removed in a future version. "
    "Use 'from geokit.raster import ...' or 'from geokit import raster' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.raster import *  # noqa: F401, F403, E402
