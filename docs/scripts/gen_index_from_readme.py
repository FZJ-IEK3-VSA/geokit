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

# --- Theme-aware logos -------------------------------------------------------
# On GitHub the README switches logos between light and dark with the HTML
# <picture> element (GitHub's supported mechanism, which follows the GitHub
# theme). Material for MkDocs does not use <picture> for this -- it switches
# images via the "#only-light" / "#only-dark" URL-fragment convention, which
# follows the docs palette toggle. So on the docs home page we swap each
# <picture> logo for its Material-native two-<img> equivalent.
#
# Every swappable logo is wrapped in the README with
#     <!-- logo:NAME:start --> ... <!-- logo:NAME:end -->
# (invisible HTML comments on GitHub). LOGO_BLOCKS maps NAME to the Material
# version that replaces everything between those markers. To tweak a docs logo
# (URL, size, alt) edit its string below and keep it in sync with the <picture>
# version in the README, which is what GitHub renders.
_JSA_LOGOS = "https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos"
_HELMHOLTZ_LOGOS = "https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos"

LOGO_BLOCKS = {
    "jsa": f"""\
<a href="https://www.fz-juelich.de/en/ice/ice-2">
    <img src="{_JSA_LOGOS}/JSA-Header.svg#only-light" alt="Jülich Systems Analysis" height="80">
    <img src="{_JSA_LOGOS}/JSA-Header-dark.svg#only-dark" alt="Jülich Systems Analysis" height="80">
  </a>""",
    "helmholtz": f"""\
<a href="https://www.fz-juelich.de/en/ice/ice-2">
    <img src="{_HELMHOLTZ_LOGOS}/Helmholtz-Logo-Dark-Blue-RGB.svg#only-light" alt="Helmholtz Logo" height="80">
    <img src="{_HELMHOLTZ_LOGOS}/Helmholtz-Logo-White-RGB.svg#only-dark" alt="Helmholtz Logo" height="80">
  </a>""",
}


def _swap_logo_blocks(text: str) -> str:
    """Replace each ``<!-- logo:NAME:start --> ... <!-- logo:NAME:end -->``
    region (markers included) with its Material-native version from
    ``LOGO_BLOCKS``. Plain string slicing, no regex; a missing marker pair
    leaves the text untouched.
    """
    for name, replacement in LOGO_BLOCKS.items():
        start = f"<!-- logo:{name}:start -->"
        end = f"<!-- logo:{name}:end -->"
        while start in text and end in text:
            before, _, rest = text.partition(start)
            _, _, after = rest.partition(end)
            text = before + replacement + after
    return text


def _readme_to_index(text: str) -> str:
    # Drop a leading UTF-8 BOM if present.
    text = text.lstrip("﻿")
    # Remove README-only blocks (redundant on the docs site).
    text = README_ONLY.sub("", text)
    # Swap each <picture> logo for its Material-native light/dark <img> pair.
    text = _swap_logo_blocks(text)
    # Rewrite the repo-root 'docs/' prefixes to be relative to the docs root.
    text = text.replace("./docs/", "./")  # HTML/Markdown asset paths
    text = text.replace("](docs/", "](")  # Markdown links
    text = text.replace("`docs/", "`")  # inline-code folder mentions
    return text


with mkdocs_gen_files.open("index.md", "w") as fd:
    fd.write(_readme_to_index(README.read_text(encoding="utf-8")))

# "Edit this page" should point at the source of truth.
mkdocs_gen_files.set_edit_path("index.md", "README.md")
