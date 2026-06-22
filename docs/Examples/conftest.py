"""Keep the rasterio/geopandas comparison notebooks out of the nbval CI run.

CI executes every notebook under ``docs/Examples/`` with ``pytest --nbval``. The
GDAL and GeoPandas+Rasterio comparison notebooks need rasterio/geopandas (not in
the GeoKit test env) and are rendered from committed outputs instead. They are
the same notebooks excluded from the docs build in
``docs/scripts/execute_notebooks.py`` (CACHED_NOTEBOOKS) -- reuse that one list
so the exclusion stays in a single place.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from execute_notebooks import CACHED_NOTEBOOKS, DOCS_DIR  # noqa: E402

collect_ignore = [str(DOCS_DIR / rel) for rel in CACHED_NOTEBOOKS]
