"""JoinGraph class for building and validating join relationships between tables."""

from typing import Dict, List, Any, Optional
from collections import deque

import polars as pl


class JoinGraph:
    """Builds and validates join relationships between tables.

    Validates join specifications, detects cycles, and computes paths
    between all pairs of tables.
    """

    def __init__(self, tables: Dict[str, pl.LazyFrame], joins: List[Dict[str, Any]]):
        """Initialize JoinGraph with tables and join specifications.

        Args:
            tables: Dictionary mapping table names to LazyFrames
            joins: List of join specification dictionaries
        """
        self.tables = tables
        self.joins = joins

    def build(self) -> Dict[str, Dict[str, Any]]:
        """Validate joins, build graph, detect cycles, and compute all paths.

        Returns:
            Dictionary mapping source table to dict of reachable tables with paths.

        Raises:
            KeyError: If join references non-existent table
            Exception: If join specification is invalid
            ValueError: If cycles or multiple paths detected
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
        join_lookup = {}
        for source_table in self.tables.keys():
            paths_from_source = self._find_all_paths_bfs(graph, source_table)
            if paths_from_source:
                join_lookup[source_table] = paths_from_source

        return join_lookup

    def _build_graph_from_joins(self) -> Dict[str, List[Dict[str, Any]]]:
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

    def _detect_cycles(self, graph: Dict[str, List[Dict[str, Any]]]) -> None:
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

    def _find_all_paths_bfs(self, graph: Dict[str, List[Dict[str, Any]]], source: str) -> Dict[str, Dict[str, Any]]:
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
