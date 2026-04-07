"""Measure decorator for registering measures with a DataModel."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import datafusion
from datafusion.substrait import Producer

from datasubway._engine import QueryContext


def measure(data_model: Any) -> Callable:
    """Factory decorator that registers a measure function with a DataModel.

    Usage:
        @measure(dm)
        def revenue(qc):
            return (dm.table("orders")
                .filter(allow("*", qc.filters))
                .aggregate(
                    group_by=allow("*", qc.groups),
                    aggs=[F.sum(col("amount")).alias("revenue")]
                ))

    The decorated function must accept a QueryContext and return a datafusion.DataFrame
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
        except Exception as e:
            # Measure may depend on filters/groups that are empty during probe.
            # Warn so failures are visible rather than silently breaking validation.
            warnings.warn(
                f"Measure '{name}' probe failed: {e}. "
                f"Output columns unknown — sort/having validation may reject valid columns.",
                stacklevel=2,
            )

        if probe_result is not None:
            if not isinstance(probe_result, datafusion.DataFrame):
                raise TypeError(
                    f"Measure '{name}' must return a datafusion.DataFrame (use dm.table())"
                )
            # Validate plan ends with aggregate via Rust plan inspection
            substrait_plan = Producer.to_substrait_plan(
                probe_result.logical_plan(), data_model.py_ctx
            )
            if not data_model.engine.is_aggregate_plan(substrait_plan.encode()):
                raise ValueError(f"Measure '{name}' must end with .aggregate()")
            output_cols = [f.name for f in probe_result.schema()]

        # Register the measure
        data_model.measures[name] = fn
        data_model.measure_output_cols[name] = output_cols
        data_model.measure_docstrings[name] = fn.__doc__ or ""

        return fn

    return decorator
