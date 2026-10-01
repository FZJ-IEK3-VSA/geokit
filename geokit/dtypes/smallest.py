"""Shrinking a finished output to the smallest lossless type: the ``"smallest"`` mode of ADR 1.

The output is read once. Whole numbers, also whole numbers stored as floats, go to the first integer type of
the order of choice that holds every value and the noData value. Other values go to ``float32`` if every value
is exact in ``float32``, and otherwise to ``float64``. Whole numbers with a NaN noData stay float, because no
integer type stores NaN. The shrink is lossless, so it never warns.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from osgeo import gdal

from geokit.dtypes.conversions import can_hold, dtype_for_integer_range, from_band, is_whole_number, to_gdal

__all__ = ["shrink_dataset", "smallest_dtype_for_array", "smallest_dtype_for_dataset"]


@dataclass(frozen=True)
class _ValueSummary:
    """What the ``"smallest"`` mode needs to know about the data values of an output."""

    minimum: float | None  # None when no data value was seen
    maximum: float | None
    all_whole_numbers: bool
    all_exact_in_float32: bool
    any_not_finite: bool  # NaN or infinity among the data values


_EMPTY_SUMMARY = _ValueSummary(None, None, True, True, False)


def _summarize_values(values: np.ndarray, no_data) -> _ValueSummary:
    flat_values = np.asarray(values).ravel()
    flat_values = _without_no_data(flat_values, no_data)
    if flat_values.size == 0:
        return _EMPTY_SUMMARY
    if flat_values.dtype.kind == "b":
        flat_values = flat_values.astype(np.uint8)
    if flat_values.dtype.kind in "iu":
        minimum = int(flat_values.min())
        maximum = int(flat_values.max())
        exact_in_float32 = max(abs(minimum), abs(maximum)) <= 2**24
        return _ValueSummary(minimum, maximum, True, exact_in_float32, False)

    is_finite = np.isfinite(flat_values)
    any_not_finite = not bool(is_finite.all())
    finite_values = flat_values[is_finite]
    if finite_values.size == 0:
        return _ValueSummary(None, None, True, True, any_not_finite)
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    all_whole_numbers = bool(np.all(finite_values == np.floor(finite_values)))
    round_trip = finite_values.astype(np.float32).astype(finite_values.dtype)
    all_exact_in_float32 = bool(np.all(round_trip == finite_values))
    return _ValueSummary(minimum, maximum, all_whole_numbers, all_exact_in_float32, any_not_finite)


def _without_no_data(flat_values: np.ndarray, no_data) -> np.ndarray:
    if no_data is None:
        return flat_values
    if is_whole_number(no_data) or flat_values.dtype.kind in "iu":
        return flat_values[flat_values != no_data]
    if math.isnan(float(no_data)):
        return flat_values[~np.isnan(flat_values)]
    return flat_values[flat_values != no_data]


def _merge_summaries(first: _ValueSummary, second: _ValueSummary) -> _ValueSummary:
    seen_minima = [value for value in (first.minimum, second.minimum) if value is not None]
    seen_maxima = [value for value in (first.maximum, second.maximum) if value is not None]
    return _ValueSummary(
        minimum=min(seen_minima) if seen_minima else None,
        maximum=max(seen_maxima) if seen_maxima else None,
        all_whole_numbers=first.all_whole_numbers and second.all_whole_numbers,
        all_exact_in_float32=first.all_exact_in_float32 and second.all_exact_in_float32,
        any_not_finite=first.any_not_finite or second.any_not_finite,
    )


def _smallest_dtype(summary: _ValueSummary, no_data) -> np.dtype:
    """Return the smallest type that stores every summarised value and the noData value exactly."""
    no_data_is_nan = no_data is not None and not is_whole_number(no_data) and math.isnan(float(no_data))
    no_data_is_whole = no_data is None or is_whole_number(no_data)
    can_be_integer = summary.all_whole_numbers and not summary.any_not_finite and no_data_is_whole
    if can_be_integer:
        bounds = [value for value in (summary.minimum, summary.maximum, no_data) if value is not None]
        if not bounds:
            return np.dtype(np.uint8)
        integer_dtype = dtype_for_integer_range(int(min(bounds)), int(max(bounds)))
        if integer_dtype is not None:
            return integer_dtype
        return np.dtype(np.float64)
    no_data_fits_float32 = no_data is None or no_data_is_nan or can_hold(np.float32, no_data)
    if summary.all_exact_in_float32 and no_data_fits_float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def smallest_dtype_for_array(values: np.ndarray, no_data=None) -> np.dtype:
    """Return the smallest type that stores every value of ``values`` and ``no_data`` exactly.

    Whole numbers, also whole numbers stored as floats, go to the first integer type of the order of choice
    that holds every value and the noData value. Other values go to ``float32`` if every value is exact in
    ``float32``, and otherwise to ``float64``. Whole numbers with a NaN noData stay float, because no integer
    type stores NaN.
    """
    summary = _summarize_values(values, no_data)
    return _smallest_dtype(summary, no_data)


def smallest_dtype_for_dataset(dataset: gdal.Dataset, rows_per_block: int | None = None) -> np.dtype:
    """Return the smallest type that stores every pixel of ``dataset`` and its noData value exactly.

    The dataset is read once, block by block, so large outputs on disk do not have to fit in memory.
    """
    no_data = dataset.GetRasterBand(1).GetNoDataValue()
    summary = _EMPTY_SUMMARY
    for band_index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(band_index)
        for block in _read_in_row_blocks(band, rows_per_block):
            block_summary = _summarize_values(block, no_data)
            summary = _merge_summaries(summary, block_summary)
    return _smallest_dtype(summary, no_data)


def _read_in_row_blocks(band: gdal.Band, rows_per_block: int | None) -> Iterator[np.ndarray]:
    if rows_per_block is None:
        bytes_per_row = band.XSize * from_band(band).itemsize
        rows_per_block = max(1, (64 * 2**20) // max(bytes_per_row, 1))
    for first_row in range(0, band.YSize, rows_per_block):
        row_count = min(rows_per_block, band.YSize - first_row)
        yield band.ReadAsArray(xoff=0, yoff=first_row, win_xsize=band.XSize, win_ysize=row_count)


def shrink_dataset(dataset: gdal.Dataset) -> gdal.Dataset:
    """Return ``dataset`` in the smallest type that stores every pixel and the noData value exactly.

    The dataset itself is returned when it already has that type. Otherwise a new in-memory dataset is
    returned; the caller writes it to its destination.
    """
    smallest = smallest_dtype_for_dataset(dataset)
    if smallest == from_band(dataset):
        return dataset
    return gdal.Translate("", dataset, format="MEM", outputType=to_gdal(smallest))
