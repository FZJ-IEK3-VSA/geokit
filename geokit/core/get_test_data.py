import warnings

warnings.warn(
    "Importing from 'geokit.core.get_test_data' is deprecated and will be removed in a future version. "
    "Use 'from geokit.get_test_data import ...' instead.",
    FutureWarning,
    stacklevel=2,
)

from geokit.get_test_data import *  # noqa: F401, F403, E402
