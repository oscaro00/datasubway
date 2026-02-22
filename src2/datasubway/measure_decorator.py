from __future__ import annotations

import inspect
import textwrap
from typing import Callable

from datasubway.data_model import DataModel
from datasubway.libcst.measure_grouping_context import (
    validate_and_extract_grouping_context,
)


def measure(data_model: DataModel) -> Callable:

    def decorator(func: Callable) -> Callable:

        func_name = func.__name__

        # Make sure the measure name is unique
        if func_name in data_model.measures:
            raise ValueError(
                f"Measure '{func_name}' already exists in data_model.measures"
            )

        source_code = inspect.getsource(func)
        dedented_source = textwrap.dedent(source_code)

        # Validate the measure ends in some type of group_by, then agg
        # and get the grouping context
        grouping_context = validate_and_extract_grouping_context(
            dedented_source, func_name
        )

        # Add the grouping context to the data model instance
        data_model.grouping_contexts[func_name] = grouping_context

        # add the measure to the list of measures in the function
        data_model.measures[func_name] = func

        return func

    return decorator
