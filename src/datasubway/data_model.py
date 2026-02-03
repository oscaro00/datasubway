from typing import Self, Dict, List, Any, Optional, Union, Literal
from pathlib import Path
import os
import polars as pl
import libcst as cst

from datasubway.query_context.query_context import QueryContext
from datasubway.cst_builders import build_pre_agg_cst, build_table_access_cst, build_join_chain_cst
from datasubway.column_utils import resolve_column_table, get_join_specs_for_columns
from datasubway.query import combine_measure_results, apply_query_modifiers
from datasubway.join_graph import JoinGraph
from datasubway.pre_agg_manager import PreAggManager
from datasubway.measure_processing import (
    apply_transformation_pipeline,
    apply_transformation_pipeline_with_tracking,
    print_transformation_steps,
    extract_measure_source,
    exec_transformed_code,
    init_worker,
    transform_measure_worker
)

# Threshold for parallel vs sequential measure processing
# Below this count, process overhead exceeds parallelization benefit
PARALLEL_THRESHOLD = 10


class DataModel:

    def __init__(self: Self, tables: Dict[str, pl.LazyFrame], joins: List[Dict[str, Any]], pre_aggregations: Dict[str, Any], pre_agg_directory: Optional[Path], logging_directory: Optional[Path] = None) -> Self:
        """
        Expected join format:
        [
            {
                'left':'table1', 'right':'table2',
                'left_on':['col1', 'col3'], 'right_on':['col1', 'col2'],
                'how':'inner', 'direction':'right2left' # direction can also be 'both'
                # left joins only make sense if direction is right2left
            },
            {} # more join edges
        ]

        Expected pre_aggregations format:
        {
            'pre_agg1_name' : {
                'group_by' : ['tbl1.col10', 'col11'],
                'aggregations' : {
                    'tbl1.col1' : 'sum',           # Single function
                    'tbl1.col2' : ['max', 'min'],  # Multiple functions
                    'tbl2.col3' : 'mean'
                }
            }
        }

        Note: Aggregation values can be either:
        - A single function string (e.g., 'sum', 'max', 'mean')
        - A list of function strings (e.g., ['sum', 'max', 'mean'])
        Both formats are supported and will be normalized internally.

        Expected data in pre_agg_metadata:
        - name, file path, last modified timestamp, group by columns, aggregated columns with type of aggregation, row count (sort key)

        The pre_agg_metadata list should be sorted in ascending order of row count
        """

        self.tables = tables
        self.joins = joins
        self.pre_aggregations = pre_aggregations
        self.pre_agg_directory = pre_agg_directory or Path('_pre_aggregations/')

        self.table_schemas = {tbl_name : lf.collect_schema().names() for tbl_name, lf in self.tables.items()}

        self.measures = {}
        self.grouping_contexts = {}

        self.validate_tables()
        self.join_lookup = JoinGraph(self.tables, self.joins).build()

        # Initialize pre-aggregation manager
        self._pre_agg_manager = PreAggManager(
            self.tables,
            self.pre_aggregations,
            self.pre_agg_directory,
            self.join_lookup
        )
        self.pre_agg_metadata = self._pre_agg_manager.metadata

        # Initialize query logging
        self._logging_directory = logging_directory
        if logging_directory:
            logging_directory.mkdir(parents=True, exist_ok=True)


    def validate_tables(self: Self) -> None:
        for key, val in self.tables.items():
            if not isinstance(key, str) or key.find('.') != -1:
                raise TypeError('Table keys must be strings and cannot contain periods (.)')

            if not isinstance(val, pl.LazyFrame):
                raise TypeError('Table values must be lazy frame objects')

    def write_pre_aggregation(self: Self, write: Union[str, List[str]]) -> None:
        """Write pre-aggregations to parquet files.

        Delegates to PreAggManager and syncs metadata.
        """
        self._pre_agg_manager.write(write)
        self.pre_agg_metadata = self._pre_agg_manager.metadata

    def table(
        self: Self,
        original_table: str,
        group_by_cols: List[str],
        agg_cols: Dict[str, str],
        allow_pre_aggs: bool = True
    ) -> cst.BaseExpression:
        """
        Return CST node representing LazyFrame source code for use in measures.

        This method is called by libcst transformers to replace dm.table() calls
        with the actual LazyFrame source code. Routes to pre-aggregated tables
        when available, otherwise builds source code for tables with joins.

        Args:
            original_table: Primary table name (e.g., 'sales')
            group_by_cols: Columns used in .group_by() (with/without prefix)
            agg_cols: Dict mapping column -> agg function
                      e.g., {'revenue': 'sum', 'quantity': 'mean'}
            allow_pre_aggs: Whether to search for pre-aggregations

        Returns:
            libcst.BaseExpression node representing one of:
            - pl.scan_parquet(self.pre_agg_directory / 'pre_agg_name.parquet')
            - self.tables['sales'].join(self.tables['products'], ...)
            - self.tables['sales']

        Raises:
            KeyError: If original_table doesn't exist
            ValueError: If columns invalid or joins don't exist
        """
        # Validate inputs
        if original_table not in self.tables:
            raise KeyError(
                f"Table '{original_table}' not found. "
                f"Available: {list(self.tables.keys())}"
            )

        # Try pre-aggregation if allowed
        if allow_pre_aggs and self.pre_agg_metadata:
            matching_pre_agg = self._pre_agg_manager.find_matching(
                group_by_cols, agg_cols, original_table
            )

            if matching_pre_agg:
                pre_agg_path = Path(matching_pre_agg['path'])

                # Verify file exists
                if not pre_agg_path.exists():
                    import warnings
                    warnings.warn(
                        f"Pre-agg file not found: {pre_agg_path}. "
                        f"Falling back to source tables."
                    )
                else:
                    # Return CST for pre-agg scan
                    return build_pre_agg_cst(matching_pre_agg['name'])

        # Fallback: Build CST for tables with joins
        # Normalize columns to include table prefixes for proper join resolution
        normalized_group_cols = [
            resolve_column_table(col, original_table, self.table_schemas)
            for col in group_by_cols
        ]
        normalized_agg_cols_keys = [
            resolve_column_table(col, original_table, self.table_schemas)
            for col in agg_cols.keys()
        ]
        all_columns = normalized_group_cols + normalized_agg_cols_keys
        join_specs = get_join_specs_for_columns(all_columns, original_table, self.join_lookup, self.tables)

        if not join_specs:
            # Single table - no joins needed
            return build_table_access_cst(original_table)
        else:
            # Multiple tables - build join chain
            return build_join_chain_cst(original_table, join_specs)

    def query(
        self: Self,
        query_context: Dict[str, Any],
        output_type: Literal['explain', 'query', 'data'] = 'data'
    ) -> Union[str, Dict[str, str], pl.DataFrame]:
        """
        Execute measures with query context and return results.

        This method orchestrates the entire query pipeline:
        1. Validates inputs and query context
        2. Extracts and transforms measure source code
        3. Applies libcst transformations in correct order
        4. Executes transformed code
        5. Combines multiple measures via join
        6. Applies post-aggregation modifiers (having, sort, limit, offset)
        7. Returns result based on output_type

        Args:
            query_context: Query context dictionary with required 'measure' key
                          and optional 'filter', 'group', 'having', 'sort', 'limit', 'offset', 'allow_pre_aggs'
            output_type: Type of output to return:
                - 'explain': Polars query plan as string
                - 'query': Transformed source code (string or dict if multiple measures)
                - 'data': Executed data as DataFrame

        Returns:
            - str: If output_type is 'explain'
            - str or Dict[str, str]: If output_type is 'query'
            - pl.DataFrame: If output_type is 'data'

        Raises:
            KeyError: If measure names not registered
            ValueError: If query_context invalid or output_type invalid

        Example:
            >>> dm = DataModel(...)
            >>>
            >>> result = dm.query(
            ...     {'measure': ['revenue_by_store'], 'group': ['store_id']},
            ...     output_type='data'
            ... )
        """
        # Start timing for logging
        import time
        start_time = time.perf_counter()
        query_id = None
        if self._logging_directory:
            import uuid
            query_id = str(uuid.uuid4())

        # Validate output_type
        if output_type not in ['explain', 'query', 'data']:
            raise ValueError(
                f"output_type must be 'explain', 'query', or 'data', got: {output_type}"
            )

        # Validate and wrap query context
        qc = QueryContext(query_context)

        # Extract measure names
        if 'measure' not in qc.context:
            raise TypeError("Query context must include 'measure' key")
        measure_names = qc.context['measure']

        # Validate all measures exist
        for measure_name in measure_names:
            if measure_name not in self.measures:
                raise KeyError(
                    f"Measure '{measure_name}' not registered. "
                    f"Available: {list(self.measures.keys())}"
                )

        # Process measures (parallel or sequential based on count)
        used_parallel = len(measure_names) >= PARALLEL_THRESHOLD
        if used_parallel:
            results = self._process_measures_parallel(measure_names, qc)
        else:
            results = self._process_measures_sequential(measure_names, qc)

        # Unpack results
        transformed_codes = {}
        lazy_frames = []
        for i, (code, lazy_frame) in enumerate(results):
            transformed_codes[measure_names[i]] = code
            lazy_frames.append(lazy_frame)

        # Log query if logging is enabled
        if self._logging_directory:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._log_query(
                query_id=query_id,
                execution_time_ms=execution_time_ms,
                output_type=output_type,
                query_context=qc.context,
                transformed_codes=transformed_codes,
                used_parallel=used_parallel
            )

        # Handle 'query' output type
        if output_type == 'query':
            if len(measure_names) == 1:
                return transformed_codes[measure_names[0]]
            else:
                return transformed_codes

        # Combine multiple measures
        group_by_cols = qc.context.get('group')
        result = combine_measure_results(lazy_frames, group_by_cols)

        # Apply post-aggregation modifiers (having, sort, limit, offset)
        result = apply_query_modifiers(result, qc)

        # Return based on output_type
        if output_type == 'explain':
            return result.explain()

        # output_type == 'data'
        return result.collect()

    def show_measure_transformation(
        self: Self,
        query_context: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Show how a single measure is transformed through each step of the transformation pipeline.

        This debugging method applies each transformer sequentially and captures the code state
        after each transformation, making it easy to understand how a measure is parsed.

        Args:
            query_context: Query context dictionary with required 'measure' key containing
                          exactly ONE measure name. Also supports optional 'filter', 'group',
                          'sort', 'limit', 'offset', 'allow_pre_aggs' keys.
            verbose: If True, print each transformation step to console. If False, only return
                    the dictionary.

        Returns:
            Dictionary mapping transformer names to code state after each transformation.
            Keys are numbered (e.g., '0_original', '1_resolve_table_columns', etc.)
            Values are either the transformed code string or None for skipped steps.

        Raises:
            ValueError: If query_context doesn't contain exactly one measure
            KeyError: If the measure name is not registered
            Exception: If query_context is empty (from QueryContext validation)

        Example:
            >>> dm = DataModel(...)
            >>>
            >>> # Display transformation steps with verbose output
            >>> steps = dm.show_measure_transformation(
            ...     {'measure': ['total_revenue'], 'group': ['item_id']},
            ...     verbose=True
            ... )
            >>>
            >>> # Get steps programmatically without printing
            >>> steps = dm.show_measure_transformation(
            ...     {'measure': ['total_revenue'], 'group': ['item_id']},
            ...     verbose=False
            ... )
            >>> print(steps['3_replace_table_calls'])
        """
        # Validate and wrap query context
        qc = QueryContext(query_context)

        # Extract measure names
        if 'measure' not in qc.context:
            raise TypeError("Query context must include 'measure' key")

        measure_names = qc.context['measure']

        # Validate exactly one measure
        if not isinstance(measure_names, list):
            raise ValueError(
                f"show_measure_transformation() requires 'measure' to be a list, "
                f"got {type(measure_names).__name__}"
            )

        if len(measure_names) != 1:
            raise ValueError(
                f"show_measure_transformation() requires exactly one measure, "
                f"got {len(measure_names)}: {measure_names}"
            )

        measure_name = measure_names[0]

        # Validate measure exists
        if measure_name not in self.measures:
            raise KeyError(
                f"Measure '{measure_name}' not registered. "
                f"Available: {list(self.measures.keys())}"
            )

        # Extract source and get tracking steps
        source_code, decorator_var = extract_measure_source(
            self.measures[measure_name],
            measure_name
        )

        transformation_steps = apply_transformation_pipeline_with_tracking(
            source_code=source_code,
            measure_name=measure_name,
            qc_context=qc.context,
            table_schemas=self.table_schemas,
            data_model=self,
            decorator_variable_name=decorator_var
        )

        # Print if verbose
        if verbose:
            print_transformation_steps(measure_name, transformation_steps)

        return transformation_steps

    def _process_single_measure(
        self: Self,
        measure_name: str,
        query_context: QueryContext
    ) -> tuple[str, pl.LazyFrame]:
        """
        Process one measure through transformation pipeline and execution.

        Args:
            measure_name: Name of measure to process
            query_context: QueryContext instance

        Returns:
            Tuple of (transformed_code, lazy_frame)
        """
        # Extract source code
        source_code, decorator_var = extract_measure_source(
            self.measures[measure_name],
            measure_name
        )

        # Apply transformation pipeline
        transformed_code = apply_transformation_pipeline(
            source_code=source_code,
            measure_name=measure_name,
            qc_context=query_context.context,
            table_schemas=self.table_schemas,
            data_model=self,
            decorator_variable_name=decorator_var
        )

        # Execute transformed code
        lazy_frame = exec_transformed_code(
            measure_name=measure_name,
            transformed_code=transformed_code,
            data_model=self,
            query_context=query_context,
            decorator_variable_name=decorator_var
        )

        return transformed_code, lazy_frame

    def _process_measures_sequential(
        self: Self,
        measure_names: List[str],
        query_context: QueryContext
    ) -> List[tuple[str, pl.LazyFrame]]:
        """
        Process measures sequentially using existing single-measure logic.

        Used when measure count is below PARALLEL_THRESHOLD.

        Args:
            measure_names: List of measure names to process
            query_context: QueryContext instance

        Returns:
            List of (transformed_code, lazy_frame) tuples
        """
        return [
            self._process_single_measure(name, query_context)
            for name in measure_names
        ]

    def _process_measures_parallel(
        self: Self,
        measure_names: List[str],
        query_context: QueryContext
    ) -> List[tuple[str, pl.LazyFrame]]:
        """
        Process measures in parallel using ProcessPoolExecutor.

        Used when measure count >= PARALLEL_THRESHOLD. Each worker applies
        CST transformations independently, then main process executes the
        transformed code.

        Args:
            measure_names: List of measure names to process
            query_context: QueryContext instance

        Returns:
            List of (transformed_code, lazy_frame) tuples
        """
        from concurrent.futures import ProcessPoolExecutor

        # Extract source code in main process (requires access to self.measures)
        measure_sources = {}
        for name in measure_names:
            source, decorator_var = extract_measure_source(
                self.measures[name],
                name
            )
            measure_sources[name] = (source, decorator_var)

        # Prepare worker initialization data (all picklable)
        init_args = (
            self.tables,
            self.joins,
            self.pre_aggregations,
            self.pre_agg_directory,
            self.pre_agg_metadata,
            self.table_schemas,
            self.join_lookup
        )

        # Prepare per-measure work items
        work_items = [
            (name, measure_sources[name][0], query_context.context, measure_sources[name][1])
            for name in measure_names
        ]

        # Process in parallel
        max_workers = min(len(measure_names), os.cpu_count() or 4)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=init_args
        ) as executor:
            transformed_results = list(executor.map(transform_measure_worker, work_items))

        # Execute transformed code in main process and build results
        results = []
        for measure_name, transformed_code in transformed_results:
            decorator_var = measure_sources[measure_name][1]
            lazy_frame = exec_transformed_code(
                measure_name=measure_name,
                transformed_code=transformed_code,
                data_model=self,
                query_context=query_context,
                decorator_variable_name=decorator_var
            )
            results.append((transformed_code, lazy_frame))

        return results

    def _log_query(
        self: Self,
        query_id: str,
        execution_time_ms: float,
        output_type: str,
        query_context: Dict[str, Any],
        transformed_codes: Dict[str, str],
        used_parallel: bool
    ) -> None:
        """Write query log entry to parquet file.

        Each query is logged to a separate parquet file with timestamp and UUID.
        Files can be read together with: pl.scan_parquet("logs/*.parquet")

        Args:
            query_id: UUID string identifying this query
            execution_time_ms: Total query execution time in milliseconds
            output_type: One of 'explain', 'query', or 'data'
            query_context: The validated query context dictionary
            transformed_codes: Dict mapping measure names to transformed source code
            used_parallel: Whether parallel processing was used for measures
        """
        import json
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc)

        log_entry = pl.DataFrame({
            "query_id": [query_id],
            "timestamp": [timestamp],
            "execution_time_ms": [execution_time_ms],
            "output_type": [output_type],
            "measure_count": [len(transformed_codes)],
            "used_parallel": [used_parallel],
            "query_context": [json.dumps(query_context)],
            "transformed_measures": [json.dumps(transformed_codes)]
        })

        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{query_id[:8]}.parquet"
        log_entry.write_parquet(self._logging_directory / filename)
