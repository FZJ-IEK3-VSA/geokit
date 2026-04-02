import warnings

warnings.warn(
    "Importing from 'geokit.core.location' is deprecated and will be removed in a future version. "
    "Use 'from geokit.location import ...' or 'from geokit import location' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.location import *  # noqa: F401, F403, E402
