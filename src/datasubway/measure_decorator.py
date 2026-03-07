from __future__ import annotations

from typing import Callable

from datasubway.data_model import DataModel
from datasubway.polars_wrappers.proxy import LazyFrameProxy
from datasubway.query_context import QueryContext


def measure(data_model: DataModel) -> Callable:

    def decorator(func: Callable) -> Callable:

        func_name = func.__name__

        # Make sure the measure name is unique
        if func_name in data_model.measures:
            raise ValueError(
                f"Measure '{func_name}' already exists in data_model.measures"
            )

        # Call with empty QueryContext to record the proxy chain at decoration time
        mock_qc = QueryContext({"measures": []})
        proxy = func(mock_qc)

        if not isinstance(proxy, LazyFrameProxy):
            raise ValueError(
                f"Measure '{func_name}' must return a LazyFrameProxy via dm.table(). "
                f"Got: {type(proxy).__name__}"
            )

        proxy.validate_measure_chain()

        assert proxy.grouping_context is not None
        data_model.measure_grouping_contexts[func_name] = proxy.grouping_context

        output_cols = [expr.meta.output_name() for expr in proxy.agg_exprs]
        data_model.measure_output_cols[func_name] = output_cols

        data_model.measures[func_name] = func
        data_model.measure_docstrings[func_name] = func.__doc__ or func_name.replace("_", " ")

        return func

    return decorator
