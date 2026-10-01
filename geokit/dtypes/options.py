"""The data-type options: one switch, ``checks``, that turns the ``GeoKitDataTypeWarning`` warnings off (ADR 10).

``set_options`` changes the setting for the whole process. ``options`` changes it inside a ``with`` block only
and restores it afterwards; it is local to the current thread because it uses a ``contextvars.ContextVar``.
The option never changes a type or a value, and ``GeoKitDataTypeError`` is raised in any case.
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace

from geokit.error import GeoKitDataTypeWarning

__all__ = ["Options", "get_options", "issue_warning", "options", "set_options"]


@dataclass(frozen=True)
class Options:
    """The data-type options. ``checks`` turns the ``GeoKitDataTypeWarning`` warnings on or off."""

    checks: bool = True


_process_wide_options = Options()
_context_options: contextvars.ContextVar[Options | None] = contextvars.ContextVar("geokit_dtypes_options", default=None)


def get_options() -> Options:
    """Return the options in effect: those of the innermost :func:`options` block, else the process-wide ones."""
    context_options = _context_options.get()
    if context_options is not None:
        return context_options
    return _process_wide_options


def set_options(*, checks: bool | None = None) -> None:
    """Change the data-type options for the whole process.

    ``checks=False`` turns every ``GeoKitDataTypeWarning`` off. The types and the values stay the same, and
    ``GeoKitDataTypeError`` is still raised for a request GeoKit cannot carry out.
    """
    global _process_wide_options
    if checks is not None:
        _process_wide_options = replace(_process_wide_options, checks=bool(checks))


@contextlib.contextmanager
def options(*, checks: bool | None = None) -> Iterator[Options]:
    """Change the data-type options inside a ``with`` block only, and restore them afterwards.

    The change is local to the current thread (it uses a ``contextvars.ContextVar``).
    """
    changed = get_options()
    if checks is not None:
        changed = replace(changed, checks=bool(checks))
    token = _context_options.set(changed)
    try:
        yield changed
    finally:
        _context_options.reset(token)


def issue_warning(message: str) -> None:
    """Issue a ``GeoKitDataTypeWarning`` unless the checks are turned off."""
    if not get_options().checks:
        return
    warnings.warn(message, GeoKitDataTypeWarning, stacklevel=3)
