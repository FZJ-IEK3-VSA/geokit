import warnings

warnings.warn(
    "Importing from 'geokit.core.srs' is deprecated and will be removed in a future version. "
    "Use 'from geokit.srs import ...' or 'from geokit import srs' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.srs import *  # noqa: F401, F403, E402
from geokit.srs import EPSG3035, EPSG3857, EPSG4326, SRSCOMMON  # noqa: F401, E402
