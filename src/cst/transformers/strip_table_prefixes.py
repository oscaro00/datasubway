"""
Transformer to strip table prefixes from pl.col() calls.

This transformer:
1. Finds all pl.col('table.column') calls
2. Strips the table prefix to make pl.col('column')
3. This is needed because Polars DataFrames don't have prefixed column names

Example:
    >>> code = '''
    ... def total_revenue(qc):
    ...     return (
    ...         self.tables['sales'].join(self.tables['products'], ...)
    ...         .group_by([pl.col('products.product_name')])
    ...         .agg(pl.col('sales.revenue').sum())
    ...     )
    ... '''
    >>> # After transformation:
    >>> # .group_by([pl.col('product_name')])
    >>> # .agg(pl.col('revenue').sum())
"""

from typing import Optional
import libcst as cst


class StripTablePrefixes(cst.CSTTransformer):
    """
    Strips table prefixes from pl.col() calls.

    This transformer finds all pl.col('table.column') calls and strips
    the table prefix, converting them to pl.col('column').
    """

    def __init__(self, function_name: str):
        """
        Initialize the transformer.

        Args:
            function_name: Name of the function to transform
        """
        super().__init__()
        self.function_name = function_name
        self.current_function: Optional[str] = None

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
        return updated_node

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> cst.Call:
        """
        Strip table prefixes from pl.col() calls.

        Transforms:
            pl.col('table.column') → pl.col('column')
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Check if this is a pl.col() call
        if not self._is_pl_col_call(updated_node):
            return updated_node

        # Extract the column name argument
        if len(updated_node.args) == 0:
            return updated_node

        first_arg = updated_node.args[0]

        # Check if it's a string literal
        if not isinstance(first_arg.value, cst.SimpleString):
            return updated_node

        # Get the column name
        column_name = first_arg.value.value.strip('\'"')

        # Strip table prefix if present
        if '.' in column_name:
            # Split and take only the column part
            stripped_name = column_name.split('.', 1)[1]

            # Create new string literal
            # Preserve the quote style
            quote_char = first_arg.value.value[0]
            new_value = cst.SimpleString(f"{quote_char}{stripped_name}{quote_char}")

            # Create updated argument
            new_arg = first_arg.with_changes(value=new_value)

            # Create updated call with new argument
            new_args = [new_arg] + list(updated_node.args[1:])
            return updated_node.with_changes(args=new_args)

        return updated_node

    def _is_pl_col_call(self, node: cst.Call) -> bool:
        """
        Check if this is a pl.col() call.

        Args:
            node: Call node to check

        Returns:
            True if this is pl.col()
        """
        if not isinstance(node.func, cst.Attribute):
            return False

        # Check if attribute name is 'col'
        if node.func.attr.value != 'col':
            return False

        # Check if it's called on 'pl'
        if isinstance(node.func.value, cst.Name):
            return node.func.value.value == 'pl'

        return False


def strip_table_prefixes(
    source_code: str,
    function_name: str
) -> str:
    """
    Strip table prefixes from pl.col() calls.

    Args:
        source_code: Python source code containing the function
        function_name: Name of function to transform

    Returns:
        Transformed source code with table prefixes stripped

    Example:
        >>> code = '''
        ... def total_revenue(qc):
        ...     return df.group_by([pl.col('products.product_name')])
        ... '''
        >>> result = strip_table_prefixes(code, 'total_revenue')
        >>> # Result: return df.group_by([pl.col('product_name')])
    """
    module = cst.parse_module(source_code)
    transformer = StripTablePrefixes(function_name)
    new_module = module.visit(transformer)
    return new_module.code
