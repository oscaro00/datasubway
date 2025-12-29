from typing import Self, Dict, List, Any, Optional, Union, Literal
from pathlib import Path
import polars as pl
import libcst as cst

from query_context.query_context import QueryContext

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

                if not found_match:
                    all_match = False
                    break

            if not all_match:
                continue

            # All checks passed!
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

        if not agg_cols:
            raise ValueError("agg_cols cannot be empty")

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
        all_columns = group_by_cols + list(agg_cols.keys())
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
        6. Returns result based on output_type

        Args:
            query_context: Query context dictionary with required 'measure' key
                          and optional 'filter', 'group', 'sort', 'limit', 'offset', 'allow_pre_aggs'
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
        import inspect
        import textwrap
        from query_context.query_context import QueryContext

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

        # Process each measure
        transformed_codes = {}
        lazy_frames = []

        for measure_name in measure_names:
            code, lazy_frame = self._process_single_measure(measure_name, qc)
            transformed_codes[measure_name] = code
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

        # Return based on output_type
        if output_type == 'explain':
            return result.explain()

        # output_type == 'data'
        return result.collect()

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
        from cst.transformers.replace_context_with_table_columns import resolve_table_columns
        from cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods
        from cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions
        from cst.transformers.replace_table_calls import replace_table_calls

        # Extract source code
        measure_func = self.measures[measure_name]
        source_code = textwrap.dedent(inspect.getsource(measure_func))

        # Strip decorator lines (e.g., @measure(dm))
        # Decorators start with @ and appear before the def line
        lines = source_code.split('\n')
        def_line_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), 0)
        source_code = '\n'.join(lines[def_line_idx:])

        # Apply transformation pipeline
        current_code = source_code

        # 1. Replace dm.table() calls with actual LazyFrame code
        current_code = replace_table_calls(
            source_code=current_code,
            function_name=measure_name,
            runtime_context={'dm': self, 'self': self, 'qc': query_context.context}
        )

        # 2. Resolve Allow/Exclude to column lists
        current_code = resolve_table_columns(
            source_code=current_code,
            function_name=measure_name,
            runtime_context={'qc': query_context.context},
            output_type='polar_col'
        )

        # 3. Remove empty polars methods
        current_code = remove_empty_polars_methods(
            source_code=current_code,
            function_name=measure_name
        )

        # 4. Transform pre-agg expressions (only if using pre-agg)
        if 'self.pre_agg_directory' in current_code:
            current_code = transform_pre_agg_expressions(
                source_code=current_code,
                function_name=measure_name
            )

        # Execute transformed code
        exec_namespace = {
            'pl': pl,
            'self': self,
            'dm': self,
            'qc': query_context.context
        }

        exec(current_code, exec_namespace)
        measure_func = exec_namespace[measure_name]
        lazy_frame = measure_func(query_context.context)

        if not isinstance(lazy_frame, pl.LazyFrame):
            raise ValueError(
                f"Measure '{measure_name}' must return pl.LazyFrame, "
                f"got: {type(lazy_frame)}"
            )

        return current_code, lazy_frame

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