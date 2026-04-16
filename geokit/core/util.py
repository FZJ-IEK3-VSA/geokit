import warnings

warnings.warn(
    "Importing from 'geokit.core.util' is deprecated and will be removed in a future version. "
    "Use 'from geokit.util import ...' or 'from geokit import util' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.util import *  # noqa: F401, F403, E402
from geokit.util import Feature  # noqa: F401, E402
