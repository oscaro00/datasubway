from typing import Tuple, List, Set, Optional
import libcst as cst


# Polars-specific methods that indicate a polars method chain
POLARS_METHODS = {
    'select', 'filter', 'with_columns', 'drop', 'rename', 'sort',
    'limit', 'offset', 'group_by', 'group_by_dynamic', 'rolling',
    'agg', 'join', 'join_asof', 'collect', 'lazy', 'head', 'tail',
    'unique', 'drop_nulls', 'fill_null', 'fill_nan', 'with_row_index',
    'cast', 'explode', 'melt', 'pivot', 'unpivot', 'sample', 'shuffle',
    'reverse', 'slice', 'take', 'gather', 'rechunk', 'describe',
    'hstack', 'vstack', 'extend', 'clone'
}

# Group by variants that are valid before .agg()
GROUP_BY_VARIANTS = {'group_by', 'group_by_dynamic', 'rolling'}


def _remove_subchains(chains: List['MethodChain']) -> List['MethodChain']:
    """
    Remove chains that are subchains of other chains on the same line.

    A chain A is a subchain of chain B if:
    - They're on the same line
    - A.methods is a suffix of B.methods (or prefix)

    We keep only the longest chain for each line.

    Args:
        chains: List of MethodChain objects

    Returns:
        Filtered list with subchains removed
    """
    if not chains:
        return chains

    # Group chains by line number
    from collections import defaultdict
    chains_by_line = defaultdict(list)
    for chain in chains:
        chains_by_line[chain.line].append(chain)

    # For each line, keep only the longest chain(s)
    result = []
    for line, line_chains in chains_by_line.items():
        if len(line_chains) == 1:
            result.append(line_chains[0])
        else:
            # Sort by length (longest first) and keep the longest
            line_chains.sort(key=lambda c: len(c.methods), reverse=True)
            longest_length = len(line_chains[0].methods)

            # Keep all chains with the longest length (in case of ties)
            for chain in line_chains:
                if len(chain.methods) == longest_length:
                    result.append(chain)
                else:
                    break  # Rest are shorter

    return result


class MethodChain:
    """Represents a method call chain with its methods and position."""

    def __init__(self, methods: List[str], line: int):
        """
        Args:
            methods: List of method names in order (first to last)
            line: Line number where the chain ends (for sorting)
        """
        self.methods = methods
        self.line = line

    def contains_polars_methods(self) -> bool:
        """Check if this chain contains any polars methods."""
        return any(method in POLARS_METHODS for method in self.methods)

    def ends_with_group_by_agg(self) -> bool:
        """
        Check if chain ends with group_by variant followed by agg.

        Returns:
            True if the last two methods are (group_by_variant, agg)
        """
        if len(self.methods) < 2:
            return False

        second_to_last = self.methods[-2]
        last = self.methods[-1]

        return second_to_last in GROUP_BY_VARIANTS and last == 'agg'

    def has_methods_after_agg(self) -> Tuple[bool, List[str]]:
        """
        Check if there are methods called after .agg().

        Returns:
            (has_methods, list_of_method_names_after_agg)
        """
        try:
            agg_index = self.methods.index('agg')
            methods_after = self.methods[agg_index + 1:]
            return (len(methods_after) > 0, methods_after)
        except ValueError:
            # No .agg() in chain
            return (False, [])

    def __repr__(self):
        chain_str = ' -> '.join(self.methods)
        return f"MethodChain(line={self.line}, methods=[{chain_str}])"


class MethodChainCollector(cst.CSTVisitor):
    """
    Visitor that collects all method call chains in a function.

    Builds a list of MethodChain objects by visiting Call nodes and
    walking up attribute access chains.
    """

    # Declare metadata dependencies
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, target_function_name: str):
        self.target_function_name = target_function_name
        self.chains: List[MethodChain] = []
        self.current_function: Optional[str] = None
        self._processed_nodes: Set[int] = set()  # Track processed nodes to avoid duplicates

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track which function we're currently visiting."""
        self.current_function = node.name.value

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Reset current function when leaving."""
        if node.name.value == self.current_function:
            self.current_function = None

    def visit_Call(self, node: cst.Call) -> None:
        """
        Visit each Call node and extract method chains.

        For each call, if it's part of a method chain (i.e., calling a method
        on an object), we walk up the chain to collect all method names.
        """
        # Only process calls in our target function
        if self.current_function != self.target_function_name:
            return

        # Avoid processing the same node multiple times
        node_id = id(node)
        if node_id in self._processed_nodes:
            return

        # Check if this is a method call (func is an Attribute)
        if not isinstance(node.func, cst.Attribute):
            return

        # Extract the method chain
        methods = self._extract_method_chain(node)

        if methods:
            # Get line number (position) of this call
            line = self._get_line_number(node)

            # Create and store the chain
            chain = MethodChain(methods, line)
            self.chains.append(chain)

            # Mark as processed
            self._processed_nodes.add(node_id)

    def _extract_method_chain(self, node: cst.Call) -> List[str]:
        """
        Extract all method names from a method call chain.

        Walks up the chain by following Attribute nodes and collecting method names.

        Args:
            node: The Call node at the end of the chain

        Returns:
            List of method names in order (first to last)
        """
        methods = []
        current = node

        while isinstance(current, cst.Call):
            if isinstance(current.func, cst.Attribute):
                # Add method name
                method_name = current.func.attr.value
                methods.append(method_name)

                # Move up the chain
                current = current.func.value
            else:
                # Not a method call chain
                break

        # Reverse to get first-to-last order
        methods.reverse()
        return methods

    def _get_line_number(self, node: cst.CSTNode) -> int:
        """
        Get the line number of a node.

        Uses the position metadata if available, otherwise returns 0.
        """
        # Try to get position from metadata
        pos = self.get_metadata(cst.metadata.PositionProvider, node, None)
        if pos and hasattr(pos, 'start'):
            return pos.start.line
        return 0


def validate_measure_method_chain(source_code: str, function_name: str) -> Tuple[bool, str]:
    """
    Validate that a measure function's last polars method chain ends with group_by().agg().

    This function:
    1. Parses the source code with libcst
    2. Collects all method call chains in the function
    3. Filters to chains containing polars methods
    4. Finds the LAST polars chain (by line number)
    5. Validates it ends with group_by_variant().agg() with nothing after

    Args:
        source_code: The source code of the function (dedented)
        function_name: The name of the function to validate

    Returns:
        (is_valid, error_message) - error_message is empty string if valid

    Raises:
        Exception: If source code cannot be parsed
    """
    try:
        # Parse the source code
        module = cst.parse_module(source_code)

        # Wrap with metadata to get position information
        wrapper = cst.metadata.MetadataWrapper(module)

        # Collect all method chains
        collector = MethodChainCollector(function_name)
        wrapper.visit(collector)

        # Filter to polars chains only
        polars_chains = [chain for chain in collector.chains if chain.contains_polars_methods()]

        # Remove subchains - keep only the longest chain on each line
        polars_chains = _remove_subchains(polars_chains)

        if not polars_chains:
            return False, (
                f"Measure '{function_name}' must contain at least one polars method chain "
                f"ending with .group_by().agg() (or .group_by_dynamic().agg() / .rolling().agg())"
            )

        # Sort by line number to find the LAST chain
        polars_chains.sort(key=lambda c: c.line)
        last_chain = polars_chains[-1]

        # First, check if there are methods after .agg() in the last chain
        # This needs to be checked BEFORE checking if it ends with group_by/agg
        has_methods_after, methods_after = last_chain.has_methods_after_agg()
        if has_methods_after:
            methods_str = ', '.join(f'.{m}()' for m in methods_after)
            return False, (
                f"Measure '{function_name}' - the last polars method chain must not have methods after .agg(). "
                f"Found: {methods_str}"
            )

        # Now validate the last chain ends with group_by().agg()
        if not last_chain.ends_with_group_by_agg():
            # Check if it has .agg() at all
            has_agg = 'agg' in last_chain.methods

            if not has_agg:
                return False, (
                    f"Measure '{function_name}' - the last polars method chain must end with .agg(). "
                    f"Last chain: {' -> '.join(last_chain.methods)}"
                )

            # Has .agg() but not preceded by group_by variant
            return False, (
                f"Measure '{function_name}' - the last polars method chain must end with one of: "
                f".group_by().agg(), .group_by_dynamic().agg(), or .rolling().agg(). "
                f"Last chain: {' -> '.join(last_chain.methods)}"
            )

        # All validations passed
        return True, ""

    except Exception as e:
        # If parsing fails, return error
        return False, f"Failed to parse measure '{function_name}': {str(e)}"
