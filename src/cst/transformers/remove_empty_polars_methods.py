"""
Transformer to remove empty polars methods and convert .agg() to .select().

This module provides a libcst transformer that:
1. Removes any polars method with an empty list argument: .method([])
2. Tracks when .group_by([]) is encountered
3. Converts .agg() to .select() when preceded by empty .group_by()
4. Only operates within the target function

Example transformations:
    >>> # Remove empty methods
    >>> .filter([]).group_by(['id'])  →  .group_by(['id'])

    >>> # Convert agg to select after empty group_by
    >>> .group_by([]).agg(pl.col('x').sum())  →  .select(pl.col('x').sum())

    >>> # Multiple empty methods
    >>> .filter([]).group_by([]).agg(...)  →  .select(...)
"""

from typing import Optional, Union
import libcst as cst


class RemoveEmptyPolarsMethods(cst.CSTTransformer):
    """
    Removes polars methods with empty argument lists and converts
    .agg() to .select() when preceded by empty .group_by().

    This transformer ensures that measure code is clean and semantically
    correct before execution. Empty methods serve no purpose and can
    cause confusion or errors.
    """

    def __init__(self, function_name: str):
        """
        Initialize the transformer.

        Args:
            function_name: Name of the function to transform (other functions unchanged)
        """
        super().__init__()
        self.function_name = function_name
        self.current_function: Optional[str] = None
        self.last_method_was_empty_group_by: bool = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track which function we're currently visiting."""
        self.current_function = node.name.value

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        """Reset function tracking when leaving."""
        self.current_function = None
        self.last_method_was_empty_group_by = False
        return updated_node

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> Union[cst.Call, cst.BaseExpression]:
        """
        Main transformation logic.

        Handles three cases:
        1. Empty .group_by() → Remove
        2. .agg() after empty .group_by() → Convert to .select()
        3. Other empty methods → Remove

        Returns:
            - Original call if no transformation needed
            - Modified call if converting agg to select
            - Chain before the call if removing empty method
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Check if this is .agg() preceded by empty .group_by()
        if self._is_agg_method(updated_node):
            # Check the ORIGINAL node's chain (before transformations)
            if (isinstance(original_node.func, cst.Attribute) and
                isinstance(original_node.func.value, cst.Call)):
                prev_call = original_node.func.value
                if self._is_group_by_method(prev_call) and self._is_empty_method_call(prev_call):
                    # Remove empty group_by and convert agg to select
                    # Extract the chain before the .group_by([]) call
                    chain_before_groupby = None

                    # First, try to get chain from the ORIGINAL prev_call (before any transformations)
                    if isinstance(prev_call.func, cst.Attribute):
                        chain_before_groupby = prev_call.func.value

                    # If we couldn't extract the chain, don't transform
                    if chain_before_groupby is None:
                        return updated_node

                    # Return .select() with the cleaned chain
                    # We replace the .group_by([]).agg(...) with chain.select(...)
                    return cst.Call(
                        func=cst.Attribute(
                            value=chain_before_groupby,
                            attr=cst.Name('select')
                        ),
                        args=updated_node.args
                    )
            return updated_node

        # Remove empty group_by methods
        if self._is_group_by_method(updated_node) and self._is_empty_method_call(updated_node):
            # Remove this call by returning the chain before it
            if isinstance(updated_node.func, cst.Attribute):
                return updated_node.func.value
            return updated_node

        # Remove other empty method calls
        if self._is_empty_method_call(updated_node):
            if isinstance(updated_node.func, cst.Attribute):
                return updated_node.func.value

        return updated_node

    def _is_empty_method_call(self, node: cst.Call) -> bool:
        """
        Check if method call has a single empty list argument.

        Args:
            node: Call node to check

        Returns:
            True if this is a method call with empty list argument
        """
        # Must be method call (has Attribute func)
        if not isinstance(node.func, cst.Attribute):
            return False

        # Only check specific polars DataFrame methods
        method_name = node.func.attr.value
        polars_dataframe_methods = {
            'filter', 'select', 'drop', 'with_columns', 'rename',
            'sort', 'group_by', 'dynamic_group_by', 'rolling',
            'drop_nulls', 'unique', 'explode'
        }
        if method_name not in polars_dataframe_methods:
            return False

        # Must have exactly one argument
        if len(node.args) != 1:
            return False

        # Argument must not be keyword arg
        arg = node.args[0]
        if arg.keyword is not None:
            return False

        # Argument must be empty list
        if isinstance(arg.value, cst.List):
            return len(arg.value.elements) == 0

        return False

    def _is_group_by_method(self, node: cst.Call) -> bool:
        """
        Check if this is a group_by/dynamic_group_by/rolling method.

        Args:
            node: Call node to check

        Returns:
            True if this is a group_by variant
        """
        if not isinstance(node.func, cst.Attribute):
            return False

        method_name = node.func.attr.value
        return method_name in ['group_by', 'dynamic_group_by', 'rolling']

    def _is_agg_method(self, node: cst.Call) -> bool:
        """
        Check if this is an agg() method.

        Args:
            node: Call node to check

        Returns:
            True if this is an agg() call
        """
        if not isinstance(node.func, cst.Attribute):
            return False

        return node.func.attr.value == 'agg'


def remove_empty_polars_methods(
    source_code: str,
    function_name: str
) -> str:
    """
    Remove empty polars methods and convert agg() to select() when appropriate.

    This function applies the RemoveEmptyPolarsMethods transformer to clean up
    measure code by removing methods with empty argument lists and ensuring
    semantic correctness when group_by is empty.

    Args:
        source_code: Python source code containing the function
        function_name: Name of function to transform

    Returns:
        Transformed source code with empty methods removed

    Example:
        >>> code = '''
        ... def my_measure():
        ...     return df.filter([]).group_by([]).agg(pl.col('x').sum())
        ... '''
        >>> result = remove_empty_polars_methods(code, 'my_measure')
        >>> # Result: df.select(pl.col('x').sum())
    """
    # Apply transformer multiple times until code stabilizes
    # This handles cases with multiple consecutive empty methods
    current_code = source_code
    max_iterations = 10  # Safety limit to prevent infinite loops

    for _ in range(max_iterations):
        module = cst.parse_module(current_code)
        transformer = RemoveEmptyPolarsMethods(function_name)
        new_module = module.visit(transformer)
        new_code = new_module.code

        # If no changes were made, we're done
        if new_code == current_code:
            break

        current_code = new_code

    return current_code
