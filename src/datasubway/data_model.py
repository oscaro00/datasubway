from typing import Self, Dict, List, Any, Optional, Union, Literal
from pathlib import Path
import os
import inspect
import textwrap
import polars as pl
import libcst as cst

from datasubway.query_context.query_context import QueryContext
from datasubway.cst.extractors.extract_decorator_variable import extract_decorator_variable_name

# Threshold for parallel vs sequential measure processing
# Below this count, process overhead exceeds parallelization benefit
PARALLEL_THRESHOLD = 10

# Module-level worker state for ProcessPoolExecutor
_worker_dm: Optional['DataModel'] = None


def _init_worker(
    tables: Dict[str, pl.LazyFrame],
    joins: List[Dict[str, Any]],
    pre_aggs: Dict[str, Any],
    pre_agg_dir: Path,
    pre_agg_metadata: List[Dict[str, Any]],
    table_schemas: Dict[str, List[str]],
    join_lookup: Dict[str, Dict[str, Any]]
) -> None:
    """
    Initialize worker process with its own DataModel instance.

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
    _worker_dm = DataModel(tables, joins, pre_aggs, pre_agg_dir)
    _worker_dm.pre_agg_metadata = pre_agg_metadata
    # Skip recomputation - use pre-computed values from main process
    _worker_dm.table_schemas = table_schemas
    _worker_dm.join_lookup = join_lookup


def _transform_measure_worker(args: tuple) -> tuple[str, str]:
    """
    Worker function to transform a single measure's source code.

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
    replace_context = {'dm': _worker_dm, 'self': _worker_dm, 'data_model': _worker_dm, 'qc': qc_context}
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
        pre_agg_metadata = _worker_dm._extract_pre_agg_metadata_from_code(current_code)
        current_code = transform_pre_agg_expressions(
            source_code=current_code,
            function_name=measure_name,
            pre_agg_metadata=pre_agg_metadata
        )

    return (measure_name, current_code)


# Type aliases for aggregation functions
AggFuncLiteral = Literal['sum', 'mean', 'min', 'max', 'count', 'len', 'null_count', 'first', 'last', 'n_unique', 'std', 'var']
AggFuncType = Union[AggFuncLiteral, List[AggFuncLiteral]]


class DataModel:

    def __init__(self: Self, tables: Dict[str, pl.LazyFrame], joins: List[Dict[str, Any]], pre_aggregations: Dict[str, Any], pre_agg_directory: Optional[Path]) -> Self:
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
        self.join_lookup = {}
        self.pre_agg_metadata = []

        self.validate_tables()
        self.build_join_lookup()


    def validate_tables(self: Self) -> None:
        for key, val in self.tables.items():
            if not isinstance(key, str) or key.find('.') != -1:
                raise TypeError('Table keys must be strings and cannot contain periods (.)')
            
            if not isinstance(val, pl.LazyFrame):
                raise TypeError('Table values must be lazy frame objects')

    def _build_graph_from_joins(self: Self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert join list to directed adjacency list representation.

        Returns:
            Dict mapping table name to list of edge dictionaries.
            Each edge contains 'target' table name and 'join_spec' dict.
        """
        # Initialize graph with all table names
        graph = {table: [] for table in self.tables.keys()}

        for join_spec in self.joins:
            left = join_spec['left']
            right = join_spec['right']
            direction = join_spec['direction']

            # Always add forward edge (left -> right)
            graph[left].append({
                'target': right,
                'join_spec': join_spec
            })

            # Add reverse edge if bidirectional
            if direction == 'both':
                # Create reverse join spec with swapped left/right and left_on/right_on
                reverse_spec = {
                    'left': right,
                    'right': left,
                    'left_on': join_spec['right_on'],
                    'right_on': join_spec['left_on'],
                    'how': join_spec['how'],
                    'direction': 'both'
                }
                graph[right].append({
                    'target': left,
                    'join_spec': reverse_spec
                })

        return graph

    def _detect_cycles(self: Self, graph: Dict[str, List[Dict[str, Any]]]) -> None:
        """Detect cycles in the directed graph using DFS with color marking.

        Note: Bidirectional edges (A <-> B) are allowed and not considered cycles.
        Only cycles involving 3 or more nodes are detected.

        Args:
            graph: Adjacency list representation of join graph

        Raises:
            ValueError: If a cycle (3+ nodes) is detected in the join graph
        """
        # Color states: WHITE (0) = unvisited, GRAY (1) = in progress, BLACK (2) = done
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph.keys()}

        def dfs(node: str, path: List[str], parent: Optional[str] = None) -> None:
            """Recursive DFS to detect cycles.

            Args:
                node: Current node being visited
                path: Current path from root
                parent: Parent node in DFS tree (to allow bidirectional edges)
            """
            color[node] = GRAY
            path.append(node)

            for edge in graph[node]:
                neighbor = edge['target']
                if color[neighbor] == GRAY:
                    # Found a back edge
                    # Allow ONLY bidirectional edges (2-node cycles: A -> B -> A)
                    # Detect self-loops (A -> A) and longer cycles (A -> B -> C -> A)
                    cycle_start = path.index(neighbor)
                    cycle_length = len(path) - cycle_start

                    if cycle_length != 2:  # Allow only 2-node cycles (bidirectional edges)
                        cycle_path = path[cycle_start:] + [neighbor]
                        raise ValueError(f"Cycle detected in join graph: {' -> '.join(cycle_path)}")
                    # else: it's a 2-node cycle (bidirectional edge), which is allowed

                elif color[neighbor] == WHITE:
                    dfs(neighbor, path, node)

            path.pop()
            color[node] = BLACK

        # Run DFS from each unvisited node (handles disconnected components)
        for node in graph.keys():
            if color[node] == WHITE:
                dfs(node, [], None)

    def _find_all_paths_bfs(self: Self, graph: Dict[str, List[Dict[str, Any]]], source: str) -> Dict[str, Dict[str, Any]]:
        """Find all reachable tables from source using BFS.

        Args:
            graph: Adjacency list representation of join graph
            source: Source table name to start from

        Returns:
            Dict mapping destination table to path info:
            {
                'dest_table': {
                    'path': ['source', 'intermediate', 'dest'],
                    'join_specs': [{join_spec1}, {join_spec2}]
                }
            }

        Raises:
            ValueError: If multiple paths exist to the same destination
        """
        from collections import deque

        result = {}
        visited = {source}
        queue = deque([{
            'current': source,
            'path': [source],
            'join_specs': []
        }])

        while queue:
            state = queue.popleft()
            current = state['current']
            path = state['path']
            specs = state['join_specs']

            for edge in graph[current]:
                neighbor = edge['target']

                # Check if neighbor is in current path (would create cycle)
                # Skip cycles to avoid false "multiple paths" errors
                if neighbor in path:
                    continue

                if neighbor in visited:
                    # Check if this creates a multiple path situation
                    if neighbor in result:
                        existing_path = result[neighbor]['path']
                        new_path = path + [neighbor]
                        raise ValueError(
                            f"Multiple paths from {source} to {neighbor}:\n"
                            f"  Path 1: {' -> '.join(existing_path)}\n"
                            f"  Path 2: {' -> '.join(new_path)}"
                        )
                    continue

                visited.add(neighbor)
                new_path = path + [neighbor]

                # Create a clean join spec without the 'direction' field
                clean_spec = {
                    'left': edge['join_spec']['left'],
                    'right': edge['join_spec']['right'],
                    'left_on': edge['join_spec']['left_on'],
                    'right_on': edge['join_spec']['right_on'],
                    'how': edge['join_spec']['how']
                }
                new_specs = specs + [clean_spec]

                result[neighbor] = {
                    'path': new_path,
                    'join_specs': new_specs
                }

                queue.append({
                    'current': neighbor,
                    'path': new_path,
                    'join_specs': new_specs
                })

        return result

    def build_join_lookup(self: Self) -> None:
        """
        This method should parse the self.joins object and build a dictionary of dictionaries to find the joins necessary to join
        a table with another (or error that it's not possible).

        There should also be a check to make sure there are no loops or multiple paths from table A to table B.

        The inner most dictionary's values should be polars join objects and a list of tables involved in the join chain.
        """
        # Validate all join specifications
        for join_dict in self.joins:
            if join_dict['left'] not in self.tables or join_dict['right'] not in self.tables:
                raise KeyError('left and right join tables must exist')
            if join_dict['how'] not in ['inner', 'left'] or join_dict['direction'] not in ['both', 'right2left']:
                raise Exception('how must be inner or left and direction must be both or right2left')
            if join_dict['how'] == 'left' and join_dict['direction'] == 'both':
                raise Exception('Left joins only make sense with direction=right2left')

        # Build directed graph from join specifications
        graph = self._build_graph_from_joins()

        # Detect cycles (fail fast if any exist)
        self._detect_cycles(graph)

        # Compute all paths from each table to all reachable tables
        self.join_lookup = {}
        for source_table in self.tables.keys():
            paths_from_source = self._find_all_paths_bfs(graph, source_table)
            if paths_from_source:
                self.join_lookup[source_table] = paths_from_source

    def write_pre_aggregation(self: Self, write: Union[str, List[str]]) -> None:
        """
        This method should allow users to write out one, several, or all pre aggregations to the given pre aggregation directory.

        When writing out tables, the pre_agg_metadata should be updated.
        """
        from datetime import datetime

        pre_aggs_to_write = []
        if write == 'all' or write == ['all']:
            pre_aggs_to_write = list(self.pre_aggregations.keys())
        elif isinstance(write, str):
            if write in self.pre_aggregations.keys():
                pre_aggs_to_write.append(write)
            else:
                raise Exception(f'{write} not in list of defined pre aggregations')
        else:
            for pre_agg_name in write:
                if pre_agg_name in self.pre_aggregations.keys():
                    pre_aggs_to_write.append(pre_agg_name)
                else:
                    raise Exception(f'{pre_agg_name} not in list of defined pre aggregations')

        # Ensure pre_agg_directory exists
        self.pre_agg_directory.mkdir(parents=True, exist_ok=True)

        # Write each pre-aggregation
        for pre_agg_name in pre_aggs_to_write:
            pre_agg_config = self.pre_aggregations[pre_agg_name]

            # Extract configuration
            group_by_cols = pre_agg_config['group_by']
            aggregations = pre_agg_config['aggregations']  # {'col': 'func' or ['func1', 'func2'], ...}
            normalized_aggs = self._normalize_aggregations(aggregations)

            # Build list of all expressions using _get_pre_agg_calculation
            all_exprs = []
            for col_name, agg_funcs in normalized_aggs.items():
                if not agg_funcs:
                    raise ValueError(
                        f"Pre-aggregation '{pre_agg_name}' has empty aggregation list for column '{col_name}'. "
                        f"Provide at least one aggregation function."
                    )
                for agg_func in agg_funcs:
                    exprs = self._get_pre_agg_calculation(col_name, agg_func)
                    all_exprs.extend(exprs)

            # Determine which tables are needed and build base LazyFrame
            # Pass aggregation columns first (to use first agg column's table as base)
            agg_columns = list(normalized_aggs.keys())
            all_columns = agg_columns + group_by_cols
            base_lf = self._resolve_tables_for_pre_agg(all_columns, base_table_hint=agg_columns[0])

            # Execute aggregation (strip table prefixes from group_by columns)
            group_by_col_names = [col.split('.')[-1] if '.' in col else col for col in group_by_cols]
            result = base_lf.group_by(group_by_col_names).agg(all_exprs)

            # Write to parquet
            output_path = self.pre_agg_directory / f'{pre_agg_name}.parquet'
            result.collect().write_parquet(output_path)

            # Get row count for metadata
            row_count = result.select(pl.len()).collect().item()

            # Update metadata
            self.pre_agg_metadata.append({
                'name': pre_agg_name,
                'path': str(output_path),
                'last_modified': datetime.now(),
                'group_by': group_by_cols,
                'aggregations': normalized_aggs,
                'row_count': row_count
            })

        # Sort metadata by row count (as per docstring requirement)
        self.pre_agg_metadata.sort(key=lambda x: x['row_count'])

    def _resolve_tables_for_pre_agg(self: Self, columns: List[str], base_table_hint: str) -> pl.LazyFrame:
        """Build LazyFrame with all tables needed for the given columns.

        Args:
            columns: List of column references (may include table prefixes like 'tbl1.col1')
            base_table_hint: Column name to extract base table from (typically first aggregation column)

        Returns:
            LazyFrame with all necessary tables joined together

        Raises:
            ValueError: If columns don't have table prefixes or if tables can't be joined
        """
        # Extract table names from column prefixes
        tables_needed = set()
        for col in columns:
            if '.' in col:
                table_name = col.split('.')[0]
                tables_needed.add(table_name)
            else:
                raise ValueError(f"Column '{col}' must have table prefix (e.g., 'table.column')")

        if not tables_needed:
            raise ValueError("Cannot determine tables needed - columns must have table prefix")

        if len(tables_needed) == 1:
            # Single table - no joins needed
            return self.tables[tables_needed.pop()]

        # Multiple tables - start with the table from base_table_hint
        # This is typically the fact table in a star schema
        base_table = base_table_hint.split('.')[0] if '.' in base_table_hint else base_table_hint
        remaining_tables = tables_needed - {base_table}

        # Start with the base table
        result = self.tables[base_table]

        # Join each remaining table to the base
        for target_table in remaining_tables:
            # Check if we can join from base to target
            if base_table not in self.join_lookup:
                raise ValueError(f"No join path exists from {base_table} to {target_table}")

            if target_table not in self.join_lookup[base_table]:
                raise ValueError(f"No join path found from {base_table} to {target_table}")

            # Get join information
            join_info = self.join_lookup[base_table][target_table]
            join_specs = join_info['join_specs']

            # Apply each join in the chain
            for join_spec in join_specs:
                right_table = join_spec['right']
                result = result.join(
                    self.tables[right_table],
                    left_on=join_spec['left_on'],
                    right_on=join_spec['right_on'],
                    how=join_spec['how']
                )

        return result


    def _get_pre_agg_calculation(
        self,
        col_name: str,
        agg_func: Literal['sum', 'mean', 'min', 'max', 'count', 'len', 'null_count', 'first', 'last', 'n_unique', 'std', 'var']
    ) -> List[pl.Expr]:
        """Generate expressions to store pre-aggregated components.

        Returns a list of Polars expressions that calculate and name the
        components needed to correctly re-aggregate this metric later.

        The purpose of this function is to enable aggregations on top of pre aggregations
        by calculating the necessary components of each metric.

        For example, think of a pre aggregation that takes the mean of a column and groups
        by a column. If you took the mean of that column once again, it would be meaningless.
        Hence, this logic stores the sum and count in the pre aggregation, then future mean
        calculations can sum the numerators and denominators before dividing.

        Args:
            col_name: Original column name (may include table prefix like 'tbl1.col1')
            agg_func: Aggregation function type

        Returns:
            List of pl.Expr with proper aliases for storage (table prefix stripped)
        """
        # Extract just column name (strip table prefix if present)
        output_col_name = col_name.split('.')[-1] if '.' in col_name else col_name
        # Use output_col_name for both column reference and alias
        col_ref = output_col_name

        match agg_func:
            # Simple aggregations (no decomposition needed)
            case 'sum':
                return [pl.col(col_ref).sum().alias(f'{output_col_name}-sum')]

            case 'min':
                return [pl.col(col_ref).min().alias(f'{output_col_name}-min')]

            case 'max':
                return [pl.col(col_ref).max().alias(f'{output_col_name}-max')]

            case 'count' | 'len':
                return [pl.len().alias(f'{output_col_name}-count')]

            case 'null_count':
                return [pl.col(col_ref).null_count().alias(f'{output_col_name}-null_count')]

            case 'first':
                return [pl.col(col_ref).first().alias(f'{output_col_name}-first')]

            case 'last':
                return [pl.col(col_ref).last().alias(f'{output_col_name}-last')]

            # Complex aggregations (require decomposition)
            case 'mean':
                return [
                    pl.col(col_ref).sum().alias(f'{output_col_name}-mean-sum'),
                    pl.len().alias(f'{output_col_name}-mean-count')
                ]

            case 'std':
                return [
                    pl.col(col_ref).sum().alias(f'{output_col_name}-std-sum'),
                    pl.col(col_ref).pow(2).sum().alias(f'{output_col_name}-std-sumsq'),
                    pl.len().alias(f'{output_col_name}-std-count')
                ]

            case 'var':
                return [
                    pl.col(col_ref).sum().alias(f'{output_col_name}-var-sum'),
                    pl.col(col_ref).pow(2).sum().alias(f'{output_col_name}-var-sumsq'),
                    pl.len().alias(f'{output_col_name}-var-count')
                ]

            # Special cases (expensive - store arrays)
            case 'n_unique':
                # Store array of unique values per group
                # Warning: Can get expensive with high cardinality
                return [pl.col(col_ref).unique().alias(f'{output_col_name}-unique-set')]

            case _:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")

    def _build_pre_agg_cst(self: Self, pre_agg_name: str) -> cst.BaseExpression:
        """
        Build CST node for: pl.scan_parquet(self.pre_agg_directory / 'pre_agg_name.parquet')

        This format allows later transformers to detect pre-agg usage by checking
        if the code contains self.pre_agg_directory.
        """
        return cst.Call(
            func=cst.Attribute(
                value=cst.Name("pl"),
                attr=cst.Name("scan_parquet")
            ),
            args=[
                cst.Arg(
                    value=cst.BinaryOperation(
                        left=cst.Attribute(
                            value=cst.Name("self"),
                            attr=cst.Name("pre_agg_directory")
                        ),
                        operator=cst.Divide(),  # / operator
                        right=cst.SimpleString(f"'{pre_agg_name}.parquet'")
                    )
                )
            ]
        )

    def _build_table_access_cst(self: Self, table_name: str) -> cst.BaseExpression:
        """
        Build CST node for: self.tables['table_name']
        """
        return cst.Subscript(
            value=cst.Attribute(
                value=cst.Name("self"),
                attr=cst.Name("tables")
            ),
            slice=[
                cst.SubscriptElement(
                    slice=cst.Index(
                        value=cst.SimpleString(f"'{table_name}'")
                    )
                )
            ]
        )

    def _build_join_chain_cst(
        self: Self,
        base_table: str,
        join_specs: List[Dict]
    ) -> cst.BaseExpression:
        """
        Build CST node for inline join chain:
        self.tables['base'].join(self.tables['t2'], left_on=['col1'], right_on=['col2'], how='inner')

        Args:
            base_table: Starting table name
            join_specs: List of join specifications from join_lookup

        Returns:
            CST node representing the chained join expression
        """
        # Start with base table access
        result = self._build_table_access_cst(base_table)

        # Chain each join
        for join_spec in join_specs:
            right_table = join_spec['right']
            left_on = join_spec['left_on']
            right_on = join_spec['right_on']
            how = join_spec['how']

            # Build join call
            result = cst.Call(
                func=cst.Attribute(
                    value=result,
                    attr=cst.Name("join")
                ),
                args=[
                    # First arg: self.tables['right_table']
                    cst.Arg(value=self._build_table_access_cst(right_table)),
                    # left_on keyword arg
                    cst.Arg(
                        keyword=cst.Name("left_on"),
                        value=cst.List([
                            cst.Element(value=cst.SimpleString(f"'{col}'"))
                            for col in left_on
                        ])
                    ),
                    # right_on keyword arg
                    cst.Arg(
                        keyword=cst.Name("right_on"),
                        value=cst.List([
                            cst.Element(value=cst.SimpleString(f"'{col}'"))
                            for col in right_on
                        ])
                    ),
                    # how keyword arg
                    cst.Arg(
                        keyword=cst.Name("how"),
                        value=cst.SimpleString(f"'{how}'")
                    )
                ]
            )

        return result

    def _normalize_column_name(self: Self, col: str, table: str) -> str:
        """Add table prefix to column if not present."""
        if '.' in col:
            return col
        return f"{table}.{col}"

    def _columns_match(self: Self, col1: str, col2: str) -> bool:
        """Check if two columns refer to same column (ignoring table prefix)."""
        clean1 = col1.split('.')[-1]
        clean2 = col2.split('.')[-1]
        return clean1 == clean2

    def _normalize_aggregations(
        self: Self,
        aggregations: Dict[str, AggFuncType]
    ) -> Dict[str, List[AggFuncLiteral]]:
        """Normalize aggregations dict to always use lists.

        Converts both formats to a consistent internal representation:
        - 'col': 'sum' -> 'col': ['sum']
        - 'col': ['sum', 'max'] -> 'col': ['sum', 'max']

        Args:
            aggregations: Dict mapping column names to agg functions (string or list)

        Returns:
            Dict mapping column names to lists of agg functions

        Example:
            >>> self._normalize_aggregations({'revenue': 'sum', 'price': ['min', 'max']})
            {'revenue': ['sum'], 'price': ['min', 'max']}
        """
        normalized = {}
        for col_name, agg_func in aggregations.items():
            if isinstance(agg_func, list):
                normalized[col_name] = agg_func
            else:
                normalized[col_name] = [agg_func]
        return normalized

    def _find_matching_pre_agg(
        self: Self,
        group_by_cols: List[str],
        agg_cols: Dict[str, str],
        original_table: str
    ) -> Optional[Dict]:
        """
        Find the smallest pre-agg that satisfies measure requirements.

        Matching criteria (ALL must be true):
        1. Pre-agg group_by must equal or be superset of measure group_by
        2. Pre-agg must contain all required aggregation columns
        3. Pre-agg aggregation functions must exactly match

        Returns:
            Pre-agg metadata dict or None
        """
        for pre_agg in self.pre_agg_metadata:  # Already sorted by row_count
            # Check 1: Pre-agg group_by must be superset (comparing without table prefixes)
            # All measure group_by columns must have a match in pre-agg group_by
            all_group_by_match = True
            for measure_col in group_by_cols:
                found_match = False
                for pre_col in pre_agg['group_by']:
                    if self._columns_match(measure_col, pre_col):
                        found_match = True
                        break
                if not found_match:
                    all_group_by_match = False
                    break

            if not all_group_by_match:
                continue

            # Check 2 & 3: Aggregation columns and functions must match
            pre_agg_aggs = pre_agg['aggregations']  # Now always Dict[str, List[str]]
            all_match = True

            for col, agg_func in agg_cols.items():
                # Find matching column in pre-agg
                found_match = False
                for pre_col, pre_funcs in pre_agg_aggs.items():
                    if self._columns_match(col, pre_col):
                        # pre_funcs is now always a list, check if query's function is in it
                        if agg_func in pre_funcs:
                            found_match = True
                            break
                        # Special case: 'rank' can use 'sum' pre-aggregations
                        # rank() operates on aggregated values, so it can work with summed data
                        elif agg_func == 'rank' and 'sum' in pre_funcs:
                            found_match = True
                            break

                if not found_match:
                    all_match = False
                    break

            if not all_match:
                continue

            # All checks passed!
            return pre_agg

        return None

    def _extract_pre_agg_metadata_from_code(self: Self, code: str) -> Optional[Dict[str, Any]]:
        """
        Extract pre-agg metadata from code by parsing the parquet filename.

        Looks for pattern: pl.scan_parquet(self.pre_agg_directory / 'filename.parquet')
        Returns the metadata for that pre-agg, or None if not found.

        Args:
            code: Source code to parse

        Returns:
            Pre-agg metadata dict or None if not found
        """
        import re

        # Find parquet filename in code
        pattern = r"self\.pre_agg_directory\s*/\s*['\"]([^'\"]+\.parquet)['\"]"
        match = re.search(pattern, code)

        if not match:
            return None

        parquet_filename = match.group(1)
        pre_agg_name = parquet_filename.replace('.parquet', '')

        # Find matching pre-agg metadata
        for pre_agg in self.pre_agg_metadata:
            if pre_agg['name'] == pre_agg_name:
                return pre_agg

        return None

    def _get_join_specs_for_columns(
        self: Self,
        columns: List[str],
        base_table: str
    ) -> List[Dict]:
        """
        Determine join specs needed for given columns.

        Similar to _resolve_tables_for_pre_agg but returns join specs instead
        of executing joins. More lenient:
        - Columns without table prefix assumed to be from base_table
        - Only joins tables that are explicitly referenced

        Args:
            columns: List of columns (may include table prefixes)
            base_table: Primary table to start from

        Returns:
            List of join specifications to build join chain, or empty list if single table

        Raises:
            ValueError: If joins don't exist or tables not found
        """
        # Extract unique table names
        tables_needed = {base_table}

        for col in columns:
            if '.' in col:
                table_name = col.split('.')[0]
                if table_name in self.tables:
                    tables_needed.add(table_name)
                else:
                    raise ValueError(f"Column '{col}' references unknown table '{table_name}'")

        # Single table - no joins needed
        if len(tables_needed) == 1:
            return []

        # Multiple tables - collect all join specs
        all_join_specs = []
        remaining_tables = tables_needed - {base_table}

        for target_table in remaining_tables:
            if base_table not in self.join_lookup:
                raise ValueError(f"No joins defined from '{base_table}'")

            if target_table not in self.join_lookup[base_table]:
                raise ValueError(
                    f"No join path from '{base_table}' to '{target_table}'. "
                    f"Available: {list(self.join_lookup[base_table].keys())}"
                )

            join_info = self.join_lookup[base_table][target_table]
            all_join_specs.extend(join_info['join_specs'])

        return all_join_specs

    def _resolve_column_table(self, col: str, base_table: str) -> str:
        """
        Add table prefix to column if missing by looking up table schemas.

        When columns are provided without table prefixes (e.g., 'category' instead of
        'products.category'), this method searches through table schemas to find which
        table contains the column.

        Args:
            col: Column name (may or may not have table prefix)
            base_table: Base table to check first (optimization)

        Returns:
            Column name with table prefix (e.g., 'products.category')

        Example:
            >>> self._resolve_column_table('category', 'sales')
            'products.category'  # Found in products table schema
        """
        # Already has table prefix - return as-is
        if '.' in col:
            return col

        # Check if column exists in base table first (common case)
        if base_table in self.table_schemas:
            if col in self.table_schemas[base_table]:
                return f"{base_table}.{col}"

        # Search other tables for the column
        for table_name, schema in self.table_schemas.items():
            if col in schema:
                return f"{table_name}.{col}"

        # Column not found in any schema - assume it belongs to base table
        # This allows for dynamic columns or columns not in schema
        return f"{base_table}.{col}"

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
            matching_pre_agg = self._find_matching_pre_agg(
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
                    return self._build_pre_agg_cst(matching_pre_agg['name'])

        # Fallback: Build CST for tables with joins
        # Normalize columns to include table prefixes for proper join resolution
        normalized_group_cols = [
            self._resolve_column_table(col, original_table)
            for col in group_by_cols
        ]
        normalized_agg_cols_keys = [
            self._resolve_column_table(col, original_table)
            for col in agg_cols.keys()
        ]
        all_columns = normalized_group_cols + normalized_agg_cols_keys
        join_specs = self._get_join_specs_for_columns(all_columns, original_table)

        if not join_specs:
            # Single table - no joins needed
            return self._build_table_access_cst(original_table)
        else:
            # Multiple tables - build join chain
            return self._build_join_chain_cst(original_table, join_specs)

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
        from datasubway.query_context.query_context import QueryContext

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
        if len(measure_names) >= PARALLEL_THRESHOLD:
            results = self._process_measures_parallel(measure_names, qc)
        else:
            results = self._process_measures_sequential(measure_names, qc)

        # Unpack results
        transformed_codes = {}
        lazy_frames = []
        for i, (code, lazy_frame) in enumerate(results):
            transformed_codes[measure_names[i]] = code
            lazy_frames.append(lazy_frame)

        # Handle 'query' output type
        if output_type == 'query':
            if len(measure_names) == 1:
                return transformed_codes[measure_names[0]]
            else:
                return transformed_codes

        # Combine multiple measures
        group_by_cols = qc.context.get('group')
        result = self._combine_measure_results(lazy_frames, group_by_cols)

        # Apply post-aggregation modifiers (having, sort, limit, offset)
        result = self._apply_query_modifiers(result, qc)

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
        from datasubway.query_context.query_context import QueryContext

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

        # Process measure with tracking
        transformation_steps = self._process_single_measure_with_tracking(measure_name, qc)

        # Print if verbose
        if verbose:
            self._print_transformation_steps(measure_name, transformation_steps)

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
        import inspect
        import textwrap
        from datasubway.cst.transformers.replace_context_with_table_columns import resolve_table_columns
        from datasubway.cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods
        from datasubway.cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions
        from datasubway.cst.transformers.replace_table_calls import replace_table_calls
        from datasubway.column_context import Allow, Exclude

        # Extract source code
        measure_func = self.measures[measure_name]
        source_code = textwrap.dedent(inspect.getsource(measure_func))

        # Extract decorator variable name BEFORE stripping decorator lines
        # This allows users to use any variable name in @measure(variable_name)
        decorator_variable_name = extract_decorator_variable_name(
            source_code=source_code,
            function_name=measure_name
        )

        # Strip decorator lines (e.g., @measure(dm))
        # Decorators start with @ and appear before the def line
        lines = source_code.split('\n')
        def_line_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), 0)
        source_code = '\n'.join(lines[def_line_idx:])

        # Apply transformation pipeline
        current_code = source_code

        # 1. Resolve Allow/Exclude to column lists (PRESERVING table prefixes)
        current_code = resolve_table_columns(
            source_code=current_code,
            function_name=measure_name,
            runtime_context={'qc': query_context.context},
            output_type='polar_col'
        )

        # 2. Inject parameters into table() calls based on method chain analysis
        from datasubway.cst.transformers.inject_table_parameters import inject_table_parameters
        # Build list of valid variable names for DataModel
        valid_var_names = ['dm', 'self', 'data_model']
        if decorator_variable_name is not None:
            valid_var_names.append(decorator_variable_name)

        current_code = inject_table_parameters(
            source_code=current_code,
            function_name=measure_name,
            runtime_context={
                'qc': query_context.context,
                'valid_var_names': valid_var_names,
                'table_schemas': self.table_schemas
            }
        )

        # 3. Replace dm.table() calls with actual LazyFrame code (joins)
        # Build runtime context with standard aliases and custom decorator variable
        replace_context = {'dm': self, 'self': self, 'data_model': self, 'qc': query_context.context}
        if decorator_variable_name is not None:
            replace_context[decorator_variable_name] = self

        current_code = replace_table_calls(
            source_code=current_code,
            function_name=measure_name,
            runtime_context=replace_context
        )

        # 4. Strip table prefixes from pl.col() calls for Polars execution
        from datasubway.cst.transformers.strip_table_prefixes import strip_table_prefixes
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
            # Extract pre-agg metadata from code to know which columns exist in pre-agg
            pre_agg_metadata = self._extract_pre_agg_metadata_from_code(current_code)
            current_code = transform_pre_agg_expressions(
                source_code=current_code,
                function_name=measure_name,
                pre_agg_metadata=pre_agg_metadata
            )

        # Execute transformed code
        exec_namespace = {
            'pl': pl,
            'self': self,
            'dm': self,
            'data_model': self,
            'Allow': Allow,
            'Exclude': Exclude,
            'qc': query_context.context
        }

        # Add decorator variable name as alias (if found)
        # This allows users to use any variable name in @measure(variable_name)
        if decorator_variable_name is not None:
            exec_namespace[decorator_variable_name] = self

        exec(current_code, exec_namespace)
        measure_func = exec_namespace[measure_name]
        lazy_frame = measure_func(query_context.context)

        if not isinstance(lazy_frame, pl.LazyFrame):
            raise ValueError(
                f"Measure '{measure_name}' must return pl.LazyFrame, "
                f"got: {type(lazy_frame)}"
            )

        return current_code, lazy_frame

    def _extract_measure_source(
        self: Self,
        measure_name: str
    ) -> tuple[str, Optional[str]]:
        """
        Extract source code and decorator variable name for a measure.

        Handles:
        1. Getting source code via inspect.getsource
        2. Extracting decorator variable name (e.g., 'dm' from @measure(dm))
        3. Stripping decorator lines

        Args:
            measure_name: Name of the registered measure

        Returns:
            Tuple of (source_code, decorator_variable_name)
            decorator_variable_name may be None if not found
        """
        measure_func = self.measures[measure_name]
        source_code = textwrap.dedent(inspect.getsource(measure_func))

        # Extract decorator variable name BEFORE stripping decorator lines
        decorator_variable_name = extract_decorator_variable_name(
            source_code=source_code,
            function_name=measure_name
        )

        # Strip decorator lines (e.g., @measure(dm))
        lines = source_code.split('\n')
        def_line_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), 0)
        source_code = '\n'.join(lines[def_line_idx:])

        return source_code, decorator_variable_name

    def _exec_transformed_code(
        self: Self,
        measure_name: str,
        transformed_code: str,
        query_context: QueryContext,
        decorator_variable_name: Optional[str] = None
    ) -> pl.LazyFrame:
        """
        Execute transformed measure code and return the resulting LazyFrame.

        This method is used after CST transformations are complete (either from
        sequential processing or from parallel workers).

        Args:
            measure_name: Name of the measure function
            transformed_code: Fully transformed Python source code
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
            'self': self,
            'dm': self,
            'data_model': self,
            'Allow': Allow,
            'Exclude': Exclude,
            'qc': query_context.context
        }

        if decorator_variable_name is not None:
            exec_namespace[decorator_variable_name] = self

        exec(transformed_code, exec_namespace)
        measure_func = exec_namespace[measure_name]
        lazy_frame = measure_func(query_context.context)

        if not isinstance(lazy_frame, pl.LazyFrame):
            raise ValueError(
                f"Measure '{measure_name}' must return pl.LazyFrame, "
                f"got: {type(lazy_frame)}"
            )

        return lazy_frame

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
            source, decorator_var = self._extract_measure_source(name)
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
            initializer=_init_worker,
            initargs=init_args
        ) as executor:
            transformed_results = list(executor.map(_transform_measure_worker, work_items))

        # Execute transformed code in main process and build results
        results = []
        for measure_name, transformed_code in transformed_results:
            decorator_var = measure_sources[measure_name][1]
            lazy_frame = self._exec_transformed_code(
                measure_name,
                transformed_code,
                query_context,
                decorator_var
            )
            results.append((transformed_code, lazy_frame))

        return results

    def _process_single_measure_with_tracking(
        self: Self,
        measure_name: str,
        query_context: QueryContext
    ) -> Dict[str, str]:
        """
        Process one measure through transformation pipeline, tracking each step.

        This method duplicates the transformation pipeline from _process_single_measure()
        but stores the code state after each transformation step instead of executing
        the final code. Used by show_measure_transformation() for debugging.

        Args:
            measure_name: Name of measure to process
            query_context: QueryContext instance

        Returns:
            Dictionary mapping step names to code state after each transformation.
            Keys are numbered (e.g., '0_original', '1_resolve_table_columns').
            Values are either code strings or None for skipped steps.
        """
        import inspect
        import textwrap
        from datasubway.cst.transformers.replace_context_with_table_columns import resolve_table_columns
        from datasubway.cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods
        from datasubway.cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions
        from datasubway.cst.transformers.replace_table_calls import replace_table_calls

        steps = {}

        # Extract source code
        measure_func = self.measures[measure_name]
        source_code = textwrap.dedent(inspect.getsource(measure_func))

        # Extract decorator variable name BEFORE stripping decorator lines
        decorator_variable_name = extract_decorator_variable_name(
            source_code=source_code,
            function_name=measure_name
        )

        # Strip decorator lines (e.g., @measure(dm))
        lines = source_code.split('\n')
        def_line_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), 0)
        source_code = '\n'.join(lines[def_line_idx:])

        # STEP 0: Original source code (after decorator strip)
        current_code = source_code
        steps['0_original'] = current_code

        # STEP 1: Resolve Allow/Exclude to column lists
        current_code = resolve_table_columns(
            source_code=current_code,
            function_name=measure_name,
            runtime_context={'qc': query_context.context},
            output_type='polar_col'
        )
        steps['1_resolve_table_columns'] = current_code

        # STEP 2: Inject parameters into table() calls
        from datasubway.cst.transformers.inject_table_parameters import inject_table_parameters
        valid_var_names = ['dm', 'self', 'data_model']
        if decorator_variable_name is not None:
            valid_var_names.append(decorator_variable_name)

        current_code = inject_table_parameters(
            source_code=current_code,
            function_name=measure_name,
            runtime_context={
                'qc': query_context.context,
                'valid_var_names': valid_var_names,
                'table_schemas': self.table_schemas
            }
        )
        steps['2_inject_table_parameters'] = current_code

        # STEP 3: Replace dm.table() calls with actual LazyFrame code
        replace_context = {'dm': self, 'self': self, 'data_model': self, 'qc': query_context.context}
        if decorator_variable_name is not None:
            replace_context[decorator_variable_name] = self

        current_code = replace_table_calls(
            source_code=current_code,
            function_name=measure_name,
            runtime_context=replace_context
        )
        steps['3_replace_table_calls'] = current_code

        # STEP 4: Strip table prefixes from pl.col() calls
        from datasubway.cst.transformers.strip_table_prefixes import strip_table_prefixes
        current_code = strip_table_prefixes(
            source_code=current_code,
            function_name=measure_name
        )
        steps['4_strip_table_prefixes'] = current_code

        # STEP 5: Remove empty polars methods
        current_code = remove_empty_polars_methods(
            source_code=current_code,
            function_name=measure_name
        )
        steps['5_remove_empty_polars_methods'] = current_code

        # STEP 6: Transform pre-agg expressions (conditional)
        if 'self.pre_agg_directory' in current_code:
            # Extract pre-agg metadata from code to know which columns exist in pre-agg
            pre_agg_metadata = self._extract_pre_agg_metadata_from_code(current_code)
            current_code = transform_pre_agg_expressions(
                source_code=current_code,
                function_name=measure_name,
                pre_agg_metadata=pre_agg_metadata
            )
            steps['6_transform_pre_agg_expressions'] = current_code
        else:
            steps['6_transform_pre_agg_expressions'] = None  # Mark as skipped

        return steps

    def _print_transformation_steps(
        self: Self,
        measure_name: str,
        steps: Dict[str, str]
    ) -> None:
        """
        Pretty-print transformation steps to console.

        Args:
            measure_name: Name of the measure being transformed
            steps: Dictionary of transformation steps from _process_single_measure_with_tracking()
        """
        print("=" * 79)
        print(f"Transformation Pipeline for Measure: '{measure_name}'")
        print("=" * 79)
        print()

        for i, (step_name, code) in enumerate(steps.items()):
            # Extract readable step name (remove numbered prefix)
            readable_name = step_name.split('_', 1)[1] if '_' in step_name else step_name

            if code is None:
                # Skipped step
                print(f"[STEP {i}] After: {readable_name} [SKIPPED]")
                print("-" * 79)
                print()
            else:
                # Show transformed code
                print(f"[STEP {i}] After: {readable_name}")
                print("-" * 79)
                print(code)
                print()

        print("=" * 79)
        print("Transformation Complete")
        print("=" * 79)

    def _combine_measure_results(
        self: Self,
        measure_results: List[pl.LazyFrame],
        group_by_cols: Optional[List[str]]
    ) -> pl.LazyFrame:
        """
        Combine multiple measure results via outer join or cross join.

        Args:
            measure_results: List of LazyFrames from different measures
            group_by_cols: Columns to join on (from query_context['group']), or None

        Returns:
            Combined LazyFrame
        """
        if len(measure_results) == 1:
            return measure_results[0]

        result = measure_results[0]
        for i, subsequent in enumerate(measure_results[1:], 1):
            if group_by_cols:
                # Outer join on group by columns with coalesce to avoid duplicate join columns
                result = result.join(subsequent, on=group_by_cols, how='full', coalesce=True, suffix=f'_{i}')
            else:
                # Cross join (cartesian product) when no group by
                result = result.join(subsequent, how='cross', suffix=f'_{i}')

        return result

    def _having_to_polars(self, having_expr: Any) -> pl.Expr:
        """
        Convert having expression to Polars boolean expression.
        Uses same syntax as filter clause.

        Args:
            having_expr: Having expression (same format as filter)

        Returns:
            Polars boolean expression for filtering

        Examples:
            ('total_revenue', '>', 1000) → pl.col('total_revenue') > 1000
            {'AND': [('total_revenue', '>', 1000), ('count', '>=', 10)]}
                → (pl.col('total_revenue') > 1000) & (pl.col('count') >= 10)
        """
        from datasubway.column_context import OPERATOR_MAP

        # Simple condition (tuple)
        if isinstance(having_expr, tuple):
            column, operator, value = having_expr
            # Strip table prefix if present (post-agg columns might not have prefixes)
            column_name = column.split('.')[-1] if '.' in column else column
            col_expr = pl.col(column_name)

            if operator not in OPERATOR_MAP:
                raise ValueError(f"Unsupported operator in having clause: {operator}")

            return OPERATOR_MAP[operator](col_expr, value)

        # AND/OR dict
        if isinstance(having_expr, dict):
            key = next(iter(having_expr.keys()))
            conditions = having_expr[key]

            if key == 'AND':
                # Combine with &
                result = self._having_to_polars(conditions[0])
                for cond in conditions[1:]:
                    result = result & self._having_to_polars(cond)
                return result

            elif key == 'OR':
                # Combine with |
                result = self._having_to_polars(conditions[0])
                for cond in conditions[1:]:
                    result = result | self._having_to_polars(cond)
                return result

        raise ValueError(f"Invalid having expression: {having_expr}")

    def _apply_query_modifiers(
        self: Self,
        lazy_frame: pl.LazyFrame,
        qc: 'QueryContext'
    ) -> pl.LazyFrame:
        """
        Apply post-aggregation query modifiers to combined result.

        Applies modifiers in SQL-compliant order:
        1. HAVING - filter aggregated results
        2. ORDER BY - sort results
        3. LIMIT/OFFSET - slice results

        Args:
            lazy_frame: Combined LazyFrame from _combine_measure_results
            qc: QueryContext with having, sort, limit, offset parameters

        Returns:
            Modified LazyFrame with all post-aggregation operations applied

        Raises:
            ValueError: If having/sort columns don't exist in result
        """
        result = lazy_frame

        # Step 1: Apply HAVING clause (post-aggregation filtering)
        having = qc.context.get('having')
        if having is not None:
            try:
                having_expr = self._having_to_polars(having)
                result = result.filter(having_expr)
            except Exception as e:
                raise ValueError(f"Failed to apply having clause: {e}") from e

        # Step 2: Apply ORDER BY (sorting)
        sort = qc.context.get('sort')
        if sort is not None and len(sort) > 0:
            try:
                # Extract column names and directions
                columns = [col_name for col_name, _ in sort]
                # Build descending list: True for 'desc', False for 'asc'
                descending = [direction == 'desc' for _, direction in sort]

                # Strip table prefixes from column names
                # (post-aggregation, columns may not have table prefixes)
                clean_columns = [
                    col.split('.')[-1] if '.' in col else col
                    for col in columns
                ]

                result = result.sort(by=clean_columns, descending=descending)
            except Exception as e:
                raise ValueError(f"Failed to apply sort: {e}") from e

        # Step 3: Apply LIMIT and OFFSET (slicing)
        limit = qc.context.get('limit', 10000)
        offset = qc.context.get('offset', 0)

        # Only apply slice if limit is positive and meaningful
        if limit is not None and limit > 0:
            result = result.slice(offset=offset, length=limit)
        elif offset > 0:
            # Edge case: offset without limit means "skip first N rows, return all remaining"
            result = result.slice(offset=offset, length=None)

        return result