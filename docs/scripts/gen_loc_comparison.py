"""Generate the code-length comparison page at documentation-build time.

Run by mkdocs-gen-files (see mkdocs.yml). It imports :mod:`count_loc` and writes
a virtual Markdown page rendered directly from the comparison notebooks, so the
published numbers always match the notebooks and no notebook is executed.
"""

import sys
from pathlib import Path

# mkdocs-gen-files runs scripts with the repo root as CWD; make the sibling
# ``count_loc`` module importable regardless of how the script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import count_loc  # noqa: E402

import mkdocs_gen_files  # noqa: E402

PAGE_PATH = "Examples/_06_application_examples/code_length_comparison.md"

with mkdocs_gen_files.open(PAGE_PATH, "w") as fd:
    fd.write(count_loc.render_markdown())

# "Edit this page" should point at the source of truth, not the virtual file.
mkdocs_gen_files.set_edit_path(PAGE_PATH, "docs/scripts/count_loc.py")
