import warnings

warnings.warn(
    "Importing from 'geokit.core.vector' is deprecated and will be removed in a future version. "
    "Use 'from geokit.vector import ...' or 'from geokit import vector' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.vector import *  # noqa: F401, F403, E402
