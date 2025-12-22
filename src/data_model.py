from typing import Self, Dict, List, Any, Optional, Union, Literal
from pathlib import Path
import polars as pl


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
                    'tbl1.col1' : 'sum',
                    'tbl1.col2' : 'max',
                    'tbl2.col3' : 'min'
                }
            }
        }

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
            aggregations = pre_agg_config['aggregations']  # {'col': 'agg_func', ...}

            # Build list of all expressions using _get_pre_agg_calculation
            all_exprs = []
            for col_name, agg_func in aggregations.items():
                exprs = self._get_pre_agg_calculation(col_name, agg_func)
                all_exprs.extend(exprs)

            # Determine which tables are needed and build base LazyFrame
            # Pass aggregation columns first (to use first agg column's table as base)
            agg_columns = list(aggregations.keys())
            all_columns = agg_columns + group_by_cols
            base_lf = self._resolve_tables_for_pre_agg(all_columns, base_table_hint=agg_columns[0])

            # Execute aggregation
            result = base_lf.group_by(group_by_cols).agg(all_exprs)

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
                'aggregations': aggregations,
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
        # Extract just column name (strip table prefix if present) for output
        output_col_name = col_name.split('.')[-1] if '.' in col_name else col_name

        match agg_func:
            # Simple aggregations (no decomposition needed)
            case 'sum':
                return [pl.col(col_name).sum().alias(f'{output_col_name}-sum')]

            case 'min':
                return [pl.col(col_name).min().alias(f'{output_col_name}-min')]

            case 'max':
                return [pl.col(col_name).max().alias(f'{output_col_name}-max')]

            case 'count' | 'len':
                return [pl.len().alias(f'{output_col_name}-count')]

            case 'null_count':
                return [pl.col(col_name).null_count().alias(f'{output_col_name}-null_count')]

            case 'first':
                return [pl.col(col_name).first().alias(f'{output_col_name}-first')]

            case 'last':
                return [pl.col(col_name).last().alias(f'{output_col_name}-last')]

            # Complex aggregations (require decomposition)
            case 'mean':
                return [
                    pl.col(col_name).sum().alias(f'{output_col_name}-mean-sum'),
                    pl.len().alias(f'{output_col_name}-mean-count')
                ]

            case 'std':
                return [
                    pl.col(col_name).sum().alias(f'{output_col_name}-std-sum'),
                    pl.col(col_name).pow(2).sum().alias(f'{output_col_name}-std-sumsq'),
                    pl.len().alias(f'{output_col_name}-std-count')
                ]

            case 'var':
                return [
                    pl.col(col_name).sum().alias(f'{output_col_name}-var-sum'),
                    pl.col(col_name).pow(2).sum().alias(f'{output_col_name}-var-sumsq'),
                    pl.len().alias(f'{output_col_name}-var-count')
                ]

            # Special cases (expensive - store arrays)
            case 'n_unique':
                # Store array of unique values per group
                # Warning: Can get expensive with high cardinality
                return [pl.col(col_name).unique().alias(f'{output_col_name}-unique-set')]

            case _:
                raise ValueError(f"Unsupported aggregation function: {agg_func}") 

    def table(self: Self, original_table: str, needed_columns: List[str], allow_pre_aggs: bool = True):
        """
        This method should be inserted into measures using libcst in place of LazyFrames at the beginning of polars method chains.
        The method will return an object cst based on the cases below

        If allow_pre_aggs is true, then search for the smallest pre_aggregation that has the necessary columns.
        This process will involve using libcst to update aggregations to work with pre aggregated columns.

        If a pre aggregation does not exist or allow_pre_aggs is false, then return a lazy frame.
        This lazy frame may potentially need other tables to be joined on
        """
        pass