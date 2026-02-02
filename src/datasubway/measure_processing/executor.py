"""Code execution helpers for measure processing."""

import inspect
import textwrap
from typing import Dict, Any, Optional, Callable, Tuple, TYPE_CHECKING

import polars as pl

from datasubway.validation.safe_literals import validate_safe_context, validate_all_strings_are_safe

if TYPE_CHECKING:
    from datasubway.data_model import DataModel
    from datasubway.query_context.query_context import QueryContext


def extract_measure_source(
    measure_func: Callable,
    measure_name: str
) -> Tuple[str, Optional[str]]:
    """Extract source code and decorator variable name for a measure.

    Handles:
    1. Getting source code via inspect.getsource
    2. Extracting decorator variable name (e.g., 'dm' from @measure(dm))
    3. Stripping decorator lines

    Args:
        measure_func: The measure function callable
        measure_name: Name of the registered measure

    Returns:
        Tuple of (source_code, decorator_variable_name)
        decorator_variable_name may be None if not found
    """
    from datasubway.cst.extractors.extract_decorator_variable import extract_decorator_variable_name

    source_code = textwrap.dedent(inspect.getsource(measure_func))

    # Extract decorator variable name BEFORE stripping decorator lines
    decorator_variable_name = extract_decorator_variable_name(
        source_code=source_code,
        function_name=measure_name
    )

    # Strip decorator lines (e.g., @measure(dm))
    lines = source_code.split('\n')
    def_line_idx = next(
        (i for i, line in enumerate(lines) if line.strip().startswith('def ')),
        0
    )
    source_code = '\n'.join(lines[def_line_idx:])

    return source_code, decorator_variable_name


def exec_transformed_code(
    measure_name: str,
    transformed_code: str,
    data_model: 'DataModel',
    query_context: 'QueryContext',
    decorator_variable_name: Optional[str] = None
) -> pl.LazyFrame:
    """Execute transformed measure code and return the resulting LazyFrame.

    This function is used after CST transformations are complete (either from
    sequential processing or from parallel workers).

    Args:
        measure_name: Name of the measure function
        transformed_code: Fully transformed Python source code
        data_model: DataModel instance to bind in exec namespace
        query_context: QueryContext instance
        decorator_variable_name: Optional custom variable name from decorator

    Returns:
        LazyFrame result from executing the measure

    Raises:
        ValueError: If measure doesn't return a LazyFrame
    """
    from datasubway.column_context import Allow, Exclude

    exec_namespace = {
        'pl': pl,
        'self': data_model,
        'dm': data_model,
        'data_model': data_model,
        'Allow': Allow,
        'Exclude': Exclude,
        'qc': query_context.context
    }

    if decorator_variable_name is not None:
        exec_namespace[decorator_variable_name] = data_model

    # Defense in depth: validate query_context before exec
    # This should already be validated in QueryContext.__init__, but we
    # re-validate here as a security safeguard
    validate_safe_context(query_context.context)
    validate_all_strings_are_safe(query_context.context)

    exec(transformed_code, exec_namespace)
    measure_func = exec_namespace[measure_name]
    lazy_frame = measure_func(query_context.context)

    if not isinstance(lazy_frame, pl.LazyFrame):
        raise ValueError(
            f"Measure '{measure_name}' must return pl.LazyFrame, "
            f"got: {type(lazy_frame)}"
        )

    return lazy_frame
