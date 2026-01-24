"""Parallel processing workers for measure transformation.

This module provides worker functions for ProcessPoolExecutor-based
parallel measure processing. Each worker process maintains its own
DataModel instance to avoid pickle issues with functions.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from datasubway.data_model import DataModel

# Module-level worker state for ProcessPoolExecutor
_worker_dm: Optional['DataModel'] = None


def init_worker(
    tables: Dict[str, pl.LazyFrame],
    joins: List[Dict[str, Any]],
    pre_aggs: Dict[str, Any],
    pre_agg_dir: Path,
    pre_agg_metadata: List[Dict[str, Any]],
    table_schemas: Dict[str, List[str]],
    join_lookup: Dict[str, Dict[str, Any]]
) -> None:
    """Initialize worker process with its own DataModel instance.

    Called once per worker by ProcessPoolExecutor. Creates a lightweight
    DataModel clone with all data needed for CST transformations, but
    without the measures dict (which contains unpicklable functions).

    Args:
        tables: Dict mapping table names to LazyFrames
        joins: List of join specifications
        pre_aggs: Pre-aggregation definitions
        pre_agg_dir: Directory for pre-aggregation parquet files
        pre_agg_metadata: List of pre-agg metadata dicts
        table_schemas: Dict mapping table names to column lists
        join_lookup: Pre-computed join paths between tables
    """
    global _worker_dm
    # Import here to avoid circular imports at module level
    from datasubway.data_model import DataModel

    _worker_dm = DataModel(tables, joins, pre_aggs, pre_agg_dir)
    _worker_dm._pre_agg_manager.metadata = pre_agg_metadata
    # Skip recomputation - use pre-computed values from main process
    _worker_dm.table_schemas = table_schemas
    _worker_dm.join_lookup = join_lookup


def transform_measure_worker(args: Tuple) -> Tuple[str, str]:
    """Worker function to transform a single measure's source code.

    Runs in a worker process. Applies all CST transformations using
    the worker's DataModel instance (_worker_dm).

    Args:
        args: Tuple of (measure_name, source_code, qc_context, decorator_var_name)

    Returns:
        Tuple of (measure_name, transformed_code)
    """
    from datasubway.cst.transformers.replace_context_with_table_columns import resolve_table_columns
    from datasubway.cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods
    from datasubway.cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions
    from datasubway.cst.transformers.replace_table_calls import replace_table_calls
    from datasubway.cst.transformers.inject_table_parameters import inject_table_parameters
    from datasubway.cst.transformers.strip_table_prefixes import strip_table_prefixes

    measure_name, source_code, qc_context, decorator_var_name = args
    global _worker_dm

    current_code = source_code

    # 1. Resolve Allow/Exclude to column lists
    current_code = resolve_table_columns(
        source_code=current_code,
        function_name=measure_name,
        runtime_context={'qc': qc_context},
        output_type='polar_col'
    )

    # 2. Inject parameters into table() calls
    valid_var_names = ['dm', 'self', 'data_model']
    if decorator_var_name is not None:
        valid_var_names.append(decorator_var_name)

    current_code = inject_table_parameters(
        source_code=current_code,
        function_name=measure_name,
        runtime_context={
            'qc': qc_context,
            'valid_var_names': valid_var_names,
            'table_schemas': _worker_dm.table_schemas
        }
    )

    # 3. Replace dm.table() calls with actual LazyFrame code
    replace_context = {
        'dm': _worker_dm,
        'self': _worker_dm,
        'data_model': _worker_dm,
        'qc': qc_context
    }
    if decorator_var_name is not None:
        replace_context[decorator_var_name] = _worker_dm

    current_code = replace_table_calls(
        source_code=current_code,
        function_name=measure_name,
        runtime_context=replace_context
    )

    # 4. Strip table prefixes from pl.col() calls
    current_code = strip_table_prefixes(
        source_code=current_code,
        function_name=measure_name
    )

    # 5. Remove empty polars methods
    current_code = remove_empty_polars_methods(
        source_code=current_code,
        function_name=measure_name
    )

    # 6. Transform pre-agg expressions (only if using pre-agg)
    if 'self.pre_agg_directory' in current_code:
        pre_agg_metadata = _worker_dm._pre_agg_manager.extract_metadata_from_code(current_code)
        current_code = transform_pre_agg_expressions(
            source_code=current_code,
            function_name=measure_name,
            pre_agg_metadata=pre_agg_metadata
        )

    return (measure_name, current_code)
