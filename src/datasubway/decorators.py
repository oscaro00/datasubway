from typing import Callable
import inspect
import textwrap

from datasubway.data_model import DataModel
from datasubway.cst.visitors.validate_measure_method_chain import validate_measure_method_chain
from datasubway.cst.visitors.get_last_grouping_context import get_last_grouping_context


def measure(data_model_instance: DataModel) -> Callable:
    """
    Decorator to register and validate measure functions in a DataModel.

    This decorator:
    1. Validates that the function name is unique (not already in data_model.measures)
    2. Validates that the last polars method chain ends with .group_by().agg()
       (or .group_by_dynamic().agg() / .rolling().agg())
    3. Registers the function in data_model.measures

    All validation occurs at decoration time (when @measure is applied), ensuring
    fast failure if there are any issues.

    Usage:
        dm = DataModel(tables=..., joins=..., pre_aggregations=..., pre_agg_directory=...)

        @measure(dm)
        def revenue_by_item():
            return (
                dm.tables['sales']
                .group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

    Args:
        data_model_instance: The DataModel instance to register this measure in

    Returns:
        Decorator function that validates and registers the measure

    Raises:
        ValueError: If the function name already exists or validation fails
    """
    def decorator(func: Callable) -> Callable:
        """
        Inner decorator function that performs validation and registration.

        Args:
            func: The measure function to decorate

        Returns:
            The original function (unchanged)

        Raises:
            ValueError: If validation fails
        """
        func_name = func.__name__

        # Check 1: Validate function name is unique
        if func_name in data_model_instance.measures:
            raise ValueError(
                f"Measure '{func_name}' already exists in data_model.measures"
            )

        # Check 2: Validate method chain using libcst
        try:
            source_code = inspect.getsource(func)
            # Dedent to remove leading whitespace
            dedented_source = textwrap.dedent(source_code)

            is_valid, error_msg = validate_measure_method_chain(dedented_source, func_name)

            if not is_valid:
                raise ValueError(error_msg)

            # Extract and store grouping context
            try:
                grouping_context = get_last_grouping_context(dedented_source, func_name)
                data_model_instance.grouping_contexts[func_name] = grouping_context
            except Exception:
                data_model_instance.grouping_contexts[func_name] = None

        except OSError as e:
            # inspect.getsource() can fail for functions defined in interactive sessions
            raise ValueError(
                f"Cannot validate measure '{func_name}': unable to get source code. "
                f"Error: {str(e)}"
            )

        # All validations passed - register the measure
        data_model_instance.measures[func_name] = func

        # Return the original function unchanged
        return func

    return decorator
