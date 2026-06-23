"""Single source of truth for the code-length comparison between ETHOS.GeoKit,
pure GDAL/OGR, and GeoPandas + Rasterio.

The same offshore-wind / North-Sea workflow is implemented three times in the
notebooks under ``docs/Examples/_06_application_examples/``. This module counts
the lines of code of each implementation **directly from the notebook JSON** so
the numbers can never drift from the notebooks and never require executing them
(no GDAL / rasterio / geopandas import, no data download).

Counting rule
-------------
Every code cell of every comparison notebook carries exactly one ``loc-*`` tag
in its cell metadata:

* ``loc-setup``    -- library imports / path configuration   (excluded)
* ``loc-download`` -- data download boilerplate, identical    (excluded)
                      across all three notebooks
* ``loc-workflow`` -- the actual data-processing workflow     (counted: processing)
* ``loc-plotting`` -- visualization / plotting               (counted: plotting)

A "line of code" is a physical source line that, after stripping whitespace, is
neither empty nor a pure comment (it does not start with ``#``).

The comparison reports three figures per implementation:

* **data processing** -- ``loc-workflow`` only; the core geospatial work and the
  fairest head-to-head comparison;
* **plotting** -- ``loc-plotting`` only;
* **total** -- processing + plotting.

``loc-setup`` and ``loc-download`` are always excluded because they are identical
boilerplate across all three implementations and would only add noise.

Usage
-----
    python docs/scripts/count_loc.py            # print the summary table
    python docs/scripts/count_loc.py --markdown # print the full generated page
    python docs/scripts/count_loc.py --write     # (re)write loc_numbers.json
    python docs/scripts/count_loc.py --check     # fail if loc_numbers.json is stale

The module is also imported by ``gen_loc_comparison.py`` (mkdocs-gen-files) to
render the documentation page at build time, and by the test suite to guard
against drift. It depends only on the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repo root resolved from this file: docs/scripts/count_loc.py -> parents[2].
# Guard against __file__ being undefined (e.g. if accidentally executed in a
# notebook-conversion context); this file is a build helper, not a doc page.
try:
    _HERE = Path(__file__).resolve()
except NameError:
    _HERE = (Path.cwd() / "docs" / "scripts" / "count_loc.py").resolve()

REPO_ROOT = _HERE.parents[2]
EXAMPLE_DIR = REPO_ROOT / "docs" / "Examples" / "_06_application_examples"
NUMBERS_FILE = _HERE.parent / "loc_numbers.json"

# Ordered so that GeoKit is the baseline the others are compared against.
NOTEBOOKS = [
    {"label": "ETHOS.GeoKit", "file": "_1_determine_offshore_turbine_depths.ipynb"},
    {"label": "GeoPandas + Rasterio", "file": "_3_determine_offshore_turbine_depths_geopandas_rasterio.ipynb"},
    {"label": "Pure GDAL/OGR", "file": "_2_determine_offshore_turbine_depths_gdal.ipynb"},
]

PROCESSING_TAG = "loc-workflow"
PLOTTING_TAG = "loc-plotting"
EXCLUDED_TAGS = ("loc-setup", "loc-download")

# Tuple (not set) so iteration order is deterministic across runs.
CATEGORY_TAGS = ("loc-setup", "loc-download", PROCESSING_TAG, PLOTTING_TAG)

# ---------------------------------------------------------------------------
# Core counting logic
# ---------------------------------------------------------------------------


def count_code_lines(source: list[str]) -> int:
    """Count non-blank, non-comment-only physical lines in a cell source."""
    n = 0
    for line in source:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        n += 1
    return n


def _cell_category(cell: dict) -> str | None:
    """Return the single ``loc-*`` category tag of a code cell, validated."""
    tags = [t for t in cell.get("metadata", {}).get("tags", []) if t in CATEGORY_TAGS]
    if len(tags) != 1:
        return None
    return tags[0]


def analyse_notebook(path: Path) -> dict:
    """Parse a notebook and return its per-cell breakdown and category totals."""
    nb = json.loads(path.read_text(encoding="utf-8"))

    cells: list[dict] = []
    totals = {tag: 0 for tag in CATEGORY_TAGS}
    last_header = ""
    code_index = 0

    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            for line in cell["source"]:
                if line.strip().startswith("#"):
                    last_header = line.strip().lstrip("#").strip()
            continue
        if cell["cell_type"] != "code":
            continue

        category = _cell_category(cell)
        if category is None:
            raise ValueError(
                f"{path.name}: code cell #{code_index} must carry exactly one of {list(CATEGORY_TAGS)} in metadata.tags"
            )

        loc = count_code_lines(cell["source"])
        totals[category] += loc
        cells.append(
            {
                "index": code_index,
                "section": last_header,
                "category": category,
                "loc": loc,
                "source": "".join(cell["source"]),
            }
        )
        code_index += 1

    processing = totals[PROCESSING_TAG]
    plotting = totals[PLOTTING_TAG]
    excluded = sum(totals[t] for t in EXCLUDED_TAGS)
    return {
        "cells": cells,
        "totals": totals,
        "processing_loc": processing,
        "plotting_loc": plotting,
        "total_loc": processing + plotting,
        "excluded_loc": excluded,
    }


def _ratio(value: int, baseline: int) -> float | None:
    return round(value / baseline, 2) if baseline else None


def compute() -> dict:
    """Compute the full comparison result for all notebooks."""
    results = []
    for spec in NOTEBOOKS:
        path = EXAMPLE_DIR / spec["file"]
        analysis = analyse_notebook(path)
        results.append({"label": spec["label"], "file": spec["file"], **analysis})

    base_proc = results[0]["processing_loc"]
    base_total = results[0]["total_loc"]
    for r in results:
        r["processing_ratio"] = _ratio(r["processing_loc"], base_proc)
        r["total_ratio"] = _ratio(r["total_loc"], base_total)

    return {"baseline_label": results[0]["label"], "notebooks": results}


def numbers_snapshot(report: dict | None = None) -> dict:
    """Compact, stable dict used as the committed drift-check snapshot."""
    report = report or compute()
    return {
        "baseline_label": report["baseline_label"],
        "notebooks": [
            {
                "label": nb["label"],
                "file": nb["file"],
                "processing_loc": nb["processing_loc"],
                "plotting_loc": nb["plotting_loc"],
                "total_loc": nb["total_loc"],
                "processing_ratio": nb["processing_ratio"],
                "total_ratio": nb["total_ratio"],
                "totals": nb["totals"],
            }
            for nb in report["notebooks"]
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _summary_table(report: dict) -> str:
    lines = [
        "| Implementation | Data processing | Plotting | Total | Processing vs GeoKit | Total vs GeoKit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nb in report["notebooks"]:
        lines.append(
            f"| {nb['label']} | {nb['processing_loc']} | {nb['plotting_loc']} | {nb['total_loc']} "
            f"| {nb['processing_ratio']:.2f}× | {nb['total_ratio']:.2f}× |"
        )
    return "\n".join(lines)


INTRO = """\


# Application Example

The [following example](../Examples/_06_application_examples/_1_determine_offshore_turbine_depths.ipynb) was created to showcase how ETHOS.GeoKit 
can be used to answer real-world questions in the field of energy system analysis. It demonstrates
how the ocean depth at offshore wind turbine locations in the North Sea can be determined. This kind of information is
important for estimating the investment required to install these turbines and for assessing future electricity costs and prices.


# Code-length comparison: GeoKit vs. GDAL vs. GeoPandas + Rasterio

The same end-to-end workflow — estimating the seabed depth at plausible 2050
offshore-wind turbine locations in the North Sea by combining the IHO sea-area
vector boundaries, the GEBCO bathymetry raster, and the turbine point dataset —
is implemented two additional times using other geospatial Python tools to showcase the boilerplate reduction:

* **GeoPandas + Rasterio** (`_3_..._geopandas_rasterio.ipynb`)
* **Pure GDAL/OGR** (`_2_..._gdal.ipynb`)

The two non-GeoKit notebooks require `rasterio`/`geopandas`, which are not
dependencies of GeoKit; they are therefore excluded from the documentation build
and shown here only as a transparency reference.

## How the lines of code are counted

Every code cell in each notebook carries exactly one category tag in its cell
metadata:

| Tag | Meaning | Counted as |
| --- | --- | :---: |
| `loc-setup` | library imports / path configuration | excluded |
| `loc-download` | data-download boilerplate (identical across all three) | excluded |
| `loc-workflow` | the actual data-processing workflow | **data processing** |
| `loc-plotting` | visualization / plotting | **plotting** |

A *line of code* is a physical source line that is neither blank nor a
comment-only line. We report **data processing** and **plotting** separately,
plus their **total**. Imports and the (identical) data-download boilerplate are
excluded because they are identical across all three implementations and would only
add noise; excluding them is conservative rather than flattering to GeoKit.

Reporting processing and plotting separately makes the comparison honest: it
shows that GeoKit's advantage is concentrated in the **data-processing** code —
the core geospatial work — rather than in plotting helpers.
"""


def render_markdown(report: dict | None = None) -> str:
    report = report or compute()
    return "\n".join([INTRO, "## Summary", _summary_table(report)]) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(report: dict) -> None:
    print(f"Baseline: {report['baseline_label']}")
    for nb in report["notebooks"]:
        print(
            f"  {nb['label']:22} processing={nb['processing_loc']:>4} "
            f"plotting={nb['plotting_loc']:>4} total={nb['total_loc']:>4}  "
            f"(proc {nb['processing_ratio']:.2f}x, total {nb['total_ratio']:.2f}x)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Code-length comparison (single source of truth).")
    parser.add_argument("--markdown", action="store_true", help="print the full generated docs page")
    parser.add_argument("--write", action="store_true", help="(re)write the committed loc_numbers.json snapshot")
    parser.add_argument("--check", action="store_true", help="exit non-zero if loc_numbers.json is stale")
    args = parser.parse_args(argv)

    report = compute()
    snapshot = numbers_snapshot(report)

    if args.markdown:
        print(render_markdown(report))
        return 0

    if args.write:
        NUMBERS_FILE.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {NUMBERS_FILE.relative_to(REPO_ROOT)}")
        return 0

    if args.check:
        if not NUMBERS_FILE.exists():
            print(f"ERROR: {NUMBERS_FILE} is missing; run: python docs/scripts/count_loc.py --write", file=sys.stderr)
            return 1
        committed = json.loads(NUMBERS_FILE.read_text(encoding="utf-8"))
        if committed != snapshot:
            print(
                "ERROR: loc_numbers.json is out of date with the notebooks.\n"
                "Run: python docs/scripts/count_loc.py --write\n"
                f"  committed: {json.dumps(committed)}\n"
                f"  current:   {json.dumps(snapshot)}",
                file=sys.stderr,
            )
            return 1
        print("loc_numbers.json is up to date.")
        return 0

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
