from typing import Self, Dict, List
from pathlib import Path
import polars as pl


# TODO create a measure decorator to add measure to a specific data model
# The measure decorator should eventually be where the measure validation occurs (must end with group_by() and agg())


class DataModel:

    def __init__(self: Self, tables: Dict[pl.LazyFrame], joins: Dict, pre_aggregations: Dict, pre_agg_directory: Path) -> Self:
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

    def _build_graph_from_joins(self: Self) -> Dict:
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

    def _detect_cycles(self: Self, graph: Dict) -> None:
        """Detect cycles in the directed graph using DFS with color marking.

        Args:
            graph: Adjacency list representation of join graph

        Raises:
            ValueError: If a cycle is detected in the join graph
        """
        # Color states: WHITE (0) = unvisited, GRAY (1) = in progress, BLACK (2) = done
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph.keys()}

        def dfs(node: str, path: List[str]) -> None:
            """Recursive DFS to detect cycles."""
            color[node] = GRAY
            path.append(node)

            for edge in graph[node]:
                neighbor = edge['target']
                if color[neighbor] == GRAY:
                    # Found a back edge - cycle detected
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    raise ValueError(f"Cycle detected in join graph: {' -> '.join(cycle_path)}")
                elif color[neighbor] == WHITE:
                    dfs(neighbor, path)

            path.pop()
            color[node] = BLACK

        # Run DFS from each unvisited node (handles disconnected components)
        for node in graph.keys():
            if color[node] == WHITE:
                dfs(node, [])

    def _find_all_paths_bfs(self: Self, graph: Dict, source: str) -> Dict:
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

    def write_pre_aggregation(self: Self) -> None:
        """
        This method should allow users to write out one, several, or all pre aggregations to the given pre aggregation directory.

        When writing out tables, the pre_agg_metadata should be updated.
        """
        pass

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