"""Measure decorator for registering measures with a DataModel."""

from __future__ import annotations

from typing import Any, Callable

from datasubway.dataframe import MeasureDataFrame
from datasubway.query_context import QueryContext


def measure(data_model: Any) -> Callable:
    """Factory decorator that registers a measure function with a DataModel.

    Usage:
        @measure(dm)
        def revenue(qc):
            return (dm.table("orders")
                .aggregate(
                    group_by=allow("*", qc.groups),
                    aggs=[{"col": "amount", "func": "sum", "alias": "revenue"}]
                ))

    The decorated function must accept a QueryContext and return a MeasureDataFrame
    whose last operation is .aggregate().
    """

    def decorator(fn: Callable) -> Callable:
        name = fn.__name__

        if name in data_model.measures:
            raise ValueError(f"Measure '{name}' is already registered")

        # Probe the measure with an empty QueryContext to extract output columns.
        empty_qc = QueryContext({"measures": [name]})
        probe_result = None
        output_cols: list[str] = []

        try:
            probe_result = fn(empty_qc)
        except Exception:
            # If the measure fails with empty context, that's ok —
            # it may depend on filters/groups that are empty.
            # We still register it; runtime errors will surface at query time.
            pass

        if probe_result is not None:
            if not isinstance(probe_result, MeasureDataFrame):
                raise TypeError(
                    f"Measure '{name}' must return a MeasureDataFrame (use dm.table())"
                )
            if probe_result._last_op != "aggregate":
                raise ValueError(f"Measure '{name}' must end with .aggregate()")
            output_cols = probe_result.columns()
            data_model.measure_grouping_contexts[name] = (
                probe_result._grouping_context or {}
            )
        else:
            data_model.measure_grouping_contexts[name] = {}

        # Register the measure
        data_model.measures[name] = fn
        data_model.measure_output_cols[name] = output_cols
        data_model.measure_docstrings[name] = fn.__doc__ or ""

        return fn

    return decorator
