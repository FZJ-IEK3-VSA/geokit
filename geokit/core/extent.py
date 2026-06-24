import warnings

warnings.warn(
    "Importing from 'geokit.core.extent' is deprecated and will be removed in a future version. "
    "Use 'from geokit.extent import ...' or 'from geokit import extent' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.extent import *  # noqa: F401, F403, E402
