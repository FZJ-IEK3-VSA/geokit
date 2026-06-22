"""Execute the documentation example notebooks in parallel and write their
outputs back in place.

This runs as a documentation **pre-build** step (see ``.readthedocs.yaml``).
``mkdocs-jupyter`` is configured with ``execute: False``, so it simply renders
the outputs produced here. Running execution as a dedicated, parallel step makes
the build faster than mkdocs-jupyter's sequential execution.

Caching policy
--------------
The GDAL (``_2``) and GeoPandas + Rasterio (``_3``) comparison notebooks are
**never executed in CI**: they require ``rasterio`` / ``geopandas`` (not GeoKit
or documentation dependencies) and are instead *cached* as committed outputs.
They are listed in ``CACHED_NOTEBOOKS`` and skipped here; refresh them manually
with the rasterio environment (see ``--help``). Every other example notebook is
output-free in git and executed fresh on each build.

Usage
-----
    python docs/scripts/execute_notebooks.py            # execute all (parallel)
    python docs/scripts/execute_notebooks.py --jobs 4   # set parallel worker count
    python docs/scripts/execute_notebooks.py --list     # list what would run
    python docs/scripts/execute_notebooks.py --only '*_05_*'   # glob filter
    python docs/scripts/execute_notebooks.py --include-cached   # also run _2/_3
                                                                # (needs rasterio)
    python docs/scripts/execute_notebooks.py --strip    # strip outputs (no execute)
    python docs/scripts/execute_notebooks.py --check-clean  # CI guard for outputs

The number of parallel processes is set with ``--jobs N`` (alias ``-j N``);
the default is ``min(CPU count, number of notebooks)``. If you hit transient
kernel-startup errors under high parallelism, lower ``--jobs`` and/or raise
``--retries`` / ``--startup-timeout``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = DOCS_DIR / "Examples"

# Rendered from committed outputs; never executed in CI (require rasterio).
CACHED_NOTEBOOKS = {
    "Examples/_06_application_examples/_2_determine_offshore_turbine_depths_gdal.ipynb",
    "Examples/_06_application_examples/_3_determine_offshore_turbine_depths_geopandas_rasterio.ipynb",
}

DEFAULT_TIMEOUT = 1800  # seconds per notebook (cell execution)
DEFAULT_STARTUP_TIMEOUT = 120  # seconds to wait for a kernel to come up
DEFAULT_RETRIES = 3  # retries for transient kernel-startup failures under load


def discover(include_cached: bool = False, only: str | None = None) -> list[Path]:
    """Return the notebooks to execute, sorted, excluding checkpoints (and the
    cached comparison notebooks unless ``include_cached``)."""
    notebooks = []
    for path in sorted(EXAMPLES_DIR.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        rel = path.relative_to(DOCS_DIR).as_posix()
        if not include_cached and rel in CACHED_NOTEBOOKS:
            continue
        if only and not fnmatch.fnmatch(rel, only):
            continue
        notebooks.append(path)
    return notebooks


# Cells whose stdout/results are boilerplate (and may print local absolute
# paths). Their outputs are cleared after execution so nothing machine-specific
# is committed/rendered.
_CLEAR_OUTPUT_TAGS = {"loc-setup", "loc-download"}


def _clear_boilerplate_outputs(nb) -> None:
    """Clear outputs of setup/download cells to avoid committing local paths."""
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        if _CLEAR_OUTPUT_TAGS & set(cell.get("metadata", {}).get("tags", [])):
            cell["outputs"] = []
            cell["execution_count"] = None


def _normalize_paths_in_outputs(nb) -> None:
    """Rewrite absolute repo paths in cell outputs to repo-relative form, so no
    machine-specific path is committed/rendered (e.g. when a cell prints a list
    of data file paths)."""
    root = str(REPO_ROOT) + os.sep

    def fix(value):
        if isinstance(value, str):
            return value.replace(root, "")
        if isinstance(value, list):
            return [fix(item) for item in value]
        return value

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = fix(out["text"])
            data = out.get("data", {})
            for key in ("text/plain", "text/html"):
                if key in data:
                    data[key] = fix(data[key])


def execute_one(path_str: str, timeout: int, startup_timeout: int, retries: int) -> tuple[str, bool, str]:
    """Execute a single notebook in its own directory, writing outputs in place.

    Returns (relative_path, ok, message). Runs in a worker process.

    Starting many Jupyter kernels at once races on TCP port allocation, which
    surfaces as transient ``ZMQError: Address already in use`` / ``Kernel died
    before replying to kernel_info``. We stagger the first launch with a small
    random jitter and retry such infrastructure failures; genuine notebook errors
    (a cell raising) are reported immediately without retry.
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = Path(path_str)
    rel = path.relative_to(DOCS_DIR).as_posix()

    # De-synchronise the initial burst of kernel startups across workers.
    time.sleep(random.uniform(0.0, 1.5))

    last_err = ""
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            nb = nbformat.read(path, as_version=4)
            # Preserve the committed language_info so execution doesn't bump the
            # (volatile, env-specific) Python patch version, which would churn the
            # committed cache / dirty the working tree.
            original_language_info = nb.get("metadata", {}).get("language_info")
            kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name") or "python3"
            client = NotebookClient(
                nb,
                timeout=timeout,
                startup_timeout=startup_timeout,
                kernel_name=kernel,
                allow_errors=False,
                # Don't write per-cell execution timestamps into cell metadata;
                # they are volatile and would create noise in the committed cache.
                record_timing=False,
                # Execute with the notebook's own directory as CWD, like Jupyter /
                # mkdocs-jupyter do, so relative data paths resolve correctly.
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute()
            if original_language_info is not None:
                nb["metadata"]["language_info"] = original_language_info
            _normalize_paths_in_outputs(nb)
            _clear_boilerplate_outputs(nb)
            nbformat.write(nb, path)
            note = f"{time.time() - started:.0f}s"
            if attempt > 1:
                note += f", attempt {attempt}"
            return rel, True, note
        except CellExecutionError as exc:
            # A cell raised: a real notebook failure, retrying won't help.
            return rel, False, f"CellExecutionError: {exc}"
        except Exception as exc:  # noqa: BLE001 - transient kernel/infra failures
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(random.uniform(1.0, 3.0) * attempt)
                continue
            return rel, False, f"{last_err} (after {attempt} attempts)\n{traceback.format_exc()}"
    return rel, False, last_err  # unreachable, keeps type checkers happy


def has_committed_outputs(path: Path) -> bool:
    """True if any code cell in the notebook has stored outputs (stdlib only)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return any(c.get("cell_type") == "code" and c.get("outputs") for c in nb["cells"])


def strip_outputs(path: Path) -> bool:
    """Clear all code-cell outputs/execution counts in a notebook in place.

    Returns True if the file changed. Uses stdlib json, matching nbformat's
    on-disk style (indent=1) to keep diffs minimal.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        meta = cell.get("metadata", {})
        if cell.get("outputs") or cell.get("execution_count") is not None or "execution" in meta:
            cell["outputs"] = []
            cell["execution_count"] = None
            meta.pop("execution", None)  # volatile per-cell run timestamps
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def cmd_strip() -> int:
    """Strip outputs from every non-cached example notebook (keeps _2/_3)."""
    stripped = [nb.relative_to(DOCS_DIR).as_posix() for nb in discover(include_cached=False) if strip_outputs(nb)]
    if stripped:
        print(f"Stripped outputs from {len(stripped)} notebook(s):")
        for rel in stripped:
            print(f"  {rel}")
    else:
        print("No outputs to strip; all non-cached notebooks are already clean.")
    return 0


def cmd_check_clean() -> int:
    """Fail if any non-cached notebook has committed outputs, or a cached one is
    missing them. Intended as a CI guard. Stdlib only."""
    dirty = [nb.relative_to(DOCS_DIR).as_posix() for nb in discover(include_cached=False) if has_committed_outputs(nb)]
    missing = [rel for rel in sorted(CACHED_NOTEBOOKS) if not has_committed_outputs(DOCS_DIR / rel)]
    ok = True
    if dirty:
        ok = False
        print("ERROR: these notebooks must NOT carry committed outputs "
              "(run: python docs/scripts/execute_notebooks.py --strip):", file=sys.stderr)
        for rel in dirty:
            print(f"  {rel}", file=sys.stderr)
    if missing:
        ok = False
        print("ERROR: these cached comparison notebooks are missing their committed outputs "
              "(refresh with the rasterio env: --include-cached --only '*_determine_offshore_turbine_depths_*'):", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
    if ok:
        print("Notebook outputs are clean: only the cached comparison notebooks carry outputs.")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="number of parallel worker processes (default: min(CPU count, #notebooks))")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-cell execution timeout in seconds")
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT,
                        help="seconds to wait for each kernel to start")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="retries for transient kernel-startup failures under high parallelism")
    parser.add_argument("--list", action="store_true", help="list the notebooks that would be executed and exit")
    parser.add_argument("--only", default=None, help="glob (relative to docs/) selecting a subset of notebooks")
    parser.add_argument(
        "--include-cached",
        action="store_true",
        help="also execute the cached comparison notebooks (_2/_3); requires rasterio/geopandas",
    )
    parser.add_argument("--strip", action="store_true",
                        help="strip outputs from non-cached notebooks (does not execute); for pre-commit cleanup")
    parser.add_argument("--check-clean", action="store_true",
                        help="CI guard: fail if non-cached notebooks carry committed outputs (or cached ones don't)")
    args = parser.parse_args(argv)

    if args.strip:
        return cmd_strip()
    if args.check_clean:
        return cmd_check_clean()

    notebooks = discover(include_cached=args.include_cached, only=args.only)
    if not notebooks:
        print("No notebooks to execute.")
        return 0

    if args.list:
        for nb in notebooks:
            print(nb.relative_to(DOCS_DIR).as_posix())
        return 0

    jobs = args.jobs or min(os.cpu_count() or 1, len(notebooks))
    print(f"Executing {len(notebooks)} notebook(s) with {jobs} worker process(es)…", flush=True)

    failures: list[tuple[str, str]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(execute_one, str(nb), args.timeout, args.startup_timeout, args.retries): nb
            for nb in notebooks
        }
        for fut in as_completed(futures):
            rel, ok, msg = fut.result()
            if ok:
                print(f"  ✓ {rel}  ({msg})", flush=True)
            else:
                print(f"  ✗ {rel}\n{msg}", flush=True)
                failures.append((rel, msg))

    elapsed = time.time() - started
    if failures:
        print(f"\n{len(failures)} notebook(s) failed after {elapsed:.0f}s:", file=sys.stderr)
        for rel, _ in failures:
            print(f"  - {rel}", file=sys.stderr)
        return 1

    print(f"\nAll {len(notebooks)} notebook(s) executed successfully in {elapsed:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
