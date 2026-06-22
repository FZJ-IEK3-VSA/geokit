"""Generate the documentation home page (``index.md``) from the repository
``README.md`` at build time, so the README is the single source of truth.

The README lives at the repository root and refers to in-repo assets and pages
with a ``docs/`` prefix (e.g. ``./docs/visualizations/...``,
``[guide](docs/example_articles/...)``). Inside the built site those same files
live at the docs root, so the prefix is stripped here. External URLs such as
``https://geokit.readthedocs.io/`` are untouched (they contain no ``docs/``
path segment).
"""

import re
from pathlib import Path

import mkdocs_gen_files

README = Path("README.md")

# Content wrapped in these HTML comments is shown on GitHub (where the comments
# are invisible) but removed from the generated docs home page. Used for things
# that are redundant on the docs site itself, e.g. the "read the docs" banner.
README_ONLY = re.compile(r"<!-- readme-only:start -->.*?<!-- readme-only:end -->\n{0,2}", re.DOTALL)


def _readme_to_index(text: str) -> str:
    # Drop a leading UTF-8 BOM if present.
    text = text.lstrip("﻿")
    # Remove README-only blocks (redundant on the docs site).
    text = README_ONLY.sub("", text)
    # Rewrite the repo-root 'docs/' prefixes to be relative to the docs root.
    text = text.replace("./docs/", "./")  # HTML/Markdown asset paths
    text = text.replace("](docs/", "](")  # Markdown links
    text = text.replace("`docs/", "`")    # inline-code folder mentions
    return text


with mkdocs_gen_files.open("index.md", "w") as fd:
    fd.write(_readme_to_index(README.read_text(encoding="utf-8")))

# "Edit this page" should point at the source of truth.
mkdocs_gen_files.set_edit_path("index.md", "README.md")
