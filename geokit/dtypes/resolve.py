"""Choosing the output type of an operation: the ``dtype`` modes, the operation rules and the scalar widening.

``resolve_dtype`` is called once per public call (ADR 6) and implements the modes ``"auto"``,
``"preserve_input"`` and ``"smallest"`` and explicit types (ADR 1, ADR 2), the rule of the operation (ADR 3)
and the widening for the noData, fill and burn values GeoKit writes itself (ADR 4).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from geokit.dtypes.conversions import (
    DTYPES_EXACT_IN_FLOAT32,
    DTYPE_MODES,
    can_hold,
    describe_range,
    dtype_for_integer_range,
    dtype_for_value,
    gdal_type_name,
    is_number,
    is_whole_number,
    promote_dtypes,
    to_dtype,
    to_gdal,
)
from geokit.error import GeoKitDataTypeError, GeoKitDataTypeWarning

__all__ = ["DTYPE_MODES", "DTYPE_PARAMETER_DOCSTRING", "DtypeRule", "ResolvedDtype", "dtype_mode", "resolve_dtype"]

DTYPE_PARAMETER_DOCSTRING = """dtype : str, numpy.dtype, type or None, optional
    The data type of the output raster. By default (None or "auto"), GeoKit chooses a type that holds
    every value the operation can produce.

    - "auto": a type that holds every possible result, chosen from the input types and the operation.
    - "preserve_input": the type of the input. GeoKit does not check the results. Results that this
      type cannot hold are clipped (overflow), fractional results are rounded, and precision can be
      lost, without a warning.
    - "smallest": as "auto", then the smallest type that stores every result exactly. Reads the
      output once.
    - an explicit type, such as "Byte", "Float32" or np.uint16: used as given. GeoKit does not check
      the results, so the same losses as under "preserve_input" can occur without a warning.

    A noData, fill or burn value that the type cannot store raises a GeoKitDataTypeError."""
"""The docstring block of the ``dtype`` parameter, shared by every function that writes a raster."""


def dtype_mode(dtype) -> str:
    """Return the mode that a ``dtype`` argument selects: ``"auto"``, ``"preserve_input"``, ``"smallest"`` or ``"explicit"``."""
    if dtype is None:
        return "auto"
    if isinstance(dtype, str):
        lowered = dtype.strip().lower()
        if lowered in DTYPE_MODES:
            return lowered
    return "explicit"


class DtypeRule(Enum):
    """The effect of an operation on the value range, which decides the type under ``"auto"``.

    ``SUBSET`` (``warp`` with ``near``, ``mode``, ``min``, ``max``, ``med``, ``q1``, ``q3``), ``IDENTITY``
    (``createRaster``, ``createRasterLike``, ``saveRasterAsTif``, ``rasterize``) and ``UNION`` (mosaics,
    ``combineSimilarRasters``) keep the promoted input type. ``FRACTIONAL`` (``warp`` with ``bilinear``,
    ``average``, ``cubic``, ``cubicspline``, ``lanczos``, ``rms``) gives ``float32`` when the input type is
    exact in ``float32`` and ``float64`` otherwise. ``SUM`` (``warp`` with ``sum``) and ``DERIVATIVE``
    (``gradient``) give ``float64``. ``SUM_OF_BURNS`` (``rasterize(add=True)``) gives the narrowest integer type
    that holds the feature count times the burn value. ``USER_FUNCTION`` (``mutateRaster``,
    ``combineSimilarRasters(combiningFunc=...)``) keeps the type of the array the function returned.
    """

    SUBSET = "subset"
    IDENTITY = "identity"
    UNION = "union"
    FRACTIONAL = "fractional"
    SUM = "sum"
    SUM_OF_BURNS = "sum of burns"
    DERIVATIVE = "derivative"
    USER_FUNCTION = "user function"


@dataclass(frozen=True)
class ResolvedDtype:
    """The type chosen for an output and the mode that chose it.

    Under ``"smallest"`` the caller runs the operation in ``dtype`` and then shrinks the finished output with
    ``geokit.dtypes.shrink_dataset`` or ``geokit.dtypes.smallest_dtype_for_array``.
    """

    dtype: np.dtype
    mode: str

    @property
    def shrink_output(self) -> bool:
        """Whether the caller has to shrink the finished output."""
        return self.mode == "smallest"

    @property
    def gdal_constant(self) -> int:
        """The GDAL pixel type constant of ``dtype``."""
        return to_gdal(self.dtype)


def resolve_dtype(
    input_dtypes: Iterable = (),
    rule: DtypeRule = DtypeRule.IDENTITY,
    *,
    input_values: Iterable = (),
    scalars: Mapping[str, object] | None = None,
    dtype=None,
    sum_count: int = 1,
    context: str = "",
) -> ResolvedDtype:
    """Choose the output type of an operation, once per public call.

    Parameters
    ----------
    input_dtypes : iterable of types
        The types of the inputs, for example the band types of the source rasters.
    rule : DtypeRule
        The effect of the operation on the value range.
    input_values : iterable of numbers
        Inputs whose exact value is known in advance, such as a constant burn value. Each counts as an input
        of the smallest type that holds it.
    scalars : mapping of name to value
        The values GeoKit writes itself: ``noData``, ``fill`` and the burn value. ``None`` values are skipped.
        Under ``"auto"`` and ``"smallest"`` the type widens until every scalar fits; under ``"preserve_input"``
        and with an explicit type a scalar that does not fit raises a ``GeoKitDataTypeError``.
    dtype : the ``dtype`` argument of the public function
        ``None`` or ``"auto"``, ``"preserve_input"``, ``"smallest"``, or an explicit type.
    sum_count : int
        For ``DtypeRule.SUM_OF_BURNS``: how many burns can add up in one pixel (the feature count).
    context : str
        The name of the public function, for messages.

    Returns
    -------
    ResolvedDtype
        The type to create the output with, and whether to shrink the output afterwards.
    """
    mode = dtype_mode(dtype)
    named_scalars = _numeric_scalars(scalars)
    promoted_input = _promote_inputs(input_dtypes, input_values)

    if mode == "explicit":
        chosen = to_dtype(dtype)
        _require_scalars_fit(chosen, named_scalars, mode=mode, context=context)
        return ResolvedDtype(chosen, mode)

    if promoted_input is None:
        promoted_input = np.dtype(np.uint8)  # a raster made from nothing is a Byte raster

    if mode == "preserve_input":
        _require_scalars_fit(promoted_input, named_scalars, mode=mode, context=context)
        return ResolvedDtype(promoted_input, mode)

    chosen = _apply_rule(promoted_input, rule, input_dtypes, input_values, sum_count)
    chosen = _widen_for_scalars(chosen, named_scalars, context=context)
    return ResolvedDtype(chosen, mode)


def _numeric_scalars(scalars: Mapping[str, object] | None) -> dict[str, object]:
    """Return the scalars with a value, checked to be numbers."""
    if scalars is None:
        return {}
    named_scalars = {}
    for name, value in scalars.items():
        if value is None:
            continue
        if not is_number(value):
            raise GeoKitDataTypeError(f"{name}={value!r} is not a number.")
        named_scalars[name] = value
    return named_scalars


def _promote_inputs(input_dtypes: Iterable, input_values: Iterable) -> np.dtype | None:
    """Return the promoted type of the input types and values, or None when there are neither."""
    dtypes_of_inputs = [to_dtype(input_dtype) for input_dtype in input_dtypes]
    for value in input_values:
        dtypes_of_inputs.append(dtype_for_value(value))
    return promote_dtypes(dtypes_of_inputs)


def _apply_rule(
    promoted_input: np.dtype, rule: DtypeRule, input_dtypes: Iterable, input_values: Iterable, sum_count: int
) -> np.dtype:
    """Return the output type that ``rule`` gives for the promoted input type."""
    if rule in (DtypeRule.SUBSET, DtypeRule.IDENTITY, DtypeRule.UNION, DtypeRule.USER_FUNCTION):
        return promoted_input
    if rule == DtypeRule.FRACTIONAL:
        if promoted_input in DTYPES_EXACT_IN_FLOAT32:
            return np.dtype(np.float32)
        return np.dtype(np.float64)
    if rule in (DtypeRule.SUM, DtypeRule.DERIVATIVE):
        return np.dtype(np.float64)
    if rule == DtypeRule.SUM_OF_BURNS:
        return _dtype_for_sum_of_burns(input_dtypes, input_values, sum_count)
    raise GeoKitDataTypeError(f"Unknown rule {rule!r}.")


def _dtype_for_sum_of_burns(input_dtypes: Iterable, input_values: Iterable, sum_count: int) -> np.dtype:
    """Return the type that holds ``sum_count`` burns of every input value, or of the whole input type range."""
    count = max(int(sum_count), 1)
    summed_dtypes = []
    for value in input_values:
        summed_dtypes.append(dtype_for_value(_times(value, count)))
    for input_dtype in input_dtypes:
        summed_dtypes.append(_dtype_for_summed_range(to_dtype(input_dtype), count))
    summed = promote_dtypes(summed_dtypes)
    if summed is None:
        return np.dtype(np.uint8)
    return summed


def _times(value, count: int):
    """Return ``value`` multiplied by ``count``, as an int when ``value`` is a whole number."""
    if is_whole_number(value):
        return int(value) * count
    return float(value) * count


def _dtype_for_summed_range(input_dtype: np.dtype, count: int) -> np.dtype:
    """Return the type that holds ``count`` copies of the full range of ``input_dtype``."""
    if input_dtype.kind == "f":  # "f" = floating point
        return np.dtype(np.float64)
    limits = np.iinfo(input_dtype)
    low = min(int(limits.min) * count, 0)
    high = int(limits.max) * count
    integer_dtype = dtype_for_integer_range(low, high)
    if integer_dtype is None:
        return np.dtype(np.float64)
    return integer_dtype


def _widen_for_scalars(chosen: np.dtype, named_scalars: Mapping[str, object], context: str) -> np.dtype:
    """Widen the type until every scalar fits; warn when a 64-bit integer type has to become a float type."""
    for scalar_name, value in named_scalars.items():
        if can_hold(chosen, value):
            continue
        widened = promote_dtypes([chosen, dtype_for_value(value)])
        # "i"/"u" = signed/unsigned integer kinds, "f" = floating point
        loses_whole_numbers = chosen.kind in "iu" and chosen.itemsize == 8 and widened.kind == "f"
        if loses_whole_numbers:
            warnings.warn(
                f"{_prefix(context)}{scalar_name}={value!r} does not fit {gdal_type_name(chosen)}. The output dtype "
                f"becomes {gdal_type_name(widened)}, which stores whole numbers exactly only up to 2**53. Pass a "
                f'{scalar_name} that fits, or dtype="{gdal_type_name(chosen)}", to keep the integer type.',
                GeoKitDataTypeWarning,
                stacklevel=2,
            )
        chosen = widened
    return chosen


def _require_scalars_fit(chosen: np.dtype, named_scalars: Mapping[str, object], mode: str, context: str) -> None:
    """Raise a ``GeoKitDataTypeError`` for the first scalar that does not fit ``chosen``."""
    for scalar_name, value in named_scalars.items():
        if can_hold(chosen, value):
            continue
        raise _scalar_does_not_fit_error(chosen, scalar_name, value, mode=mode, context=context)


def _scalar_does_not_fit_error(
    chosen: np.dtype, scalar_name: str, value, mode: str, context: str
) -> GeoKitDataTypeError:
    """Build the error for a scalar that ``chosen`` cannot hold."""
    type_description = f"{gdal_type_name(chosen)} ({describe_range(chosen)})"
    if mode == "explicit":
        where = f"the requested dtype {type_description}"
    else:
        where = f'the input dtype {type_description}, which dtype="preserve_input" keeps'
    holding_dtype = promote_dtypes([chosen, dtype_for_value(value)])
    return GeoKitDataTypeError(
        f"{_prefix(context)}{scalar_name}={value!r} cannot be stored in {where}. Pass a {scalar_name} value that "
        f'fits, a type that holds it such as dtype="{gdal_type_name(holding_dtype)}", or dtype="auto" to let '
        f"GeoKit choose."
    )


def _prefix(context: str) -> str:
    """Return ``"context: "`` for an error or warning message, or an empty string when there is no context."""
    if context:
        return f"{context}: "
    return ""
