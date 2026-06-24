import warnings

warnings.warn(
    "Importing from 'geokit.core.regionmask' is deprecated and will be removed in a future version. "
    "Use 'from geokit.regionmask import ...' or 'from geokit import regionmask' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.regionmask import *  # noqa: F401, F403, E402
