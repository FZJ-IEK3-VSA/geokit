"""Generate the API reference pages automatically.

This script is run by mkdocs-gen-files during the build process.
It discovers all Python modules in the geokit package and generates
corresponding Markdown files with mkdocstrings directives.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Directories to skip
SKIP_DIRS = {"__pycache__", "data"}

for path in sorted(Path("geokit").rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to("geokit").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__, data directories, and other non-module files
    if any(skip in parts for skip in SKIP_DIRS):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    # Skip the root package index (geokit/__init__.py → just "geokit")
    # We still want it — it shows the top-level API
    if len(parts) == 0:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

# Write the navigation file for literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
