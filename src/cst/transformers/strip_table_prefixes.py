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

from typing import Optional, Set
import libcst as cst
import libcst.matchers as m


class StripTablePrefixes(cst.CSTTransformer):
    """
    Strips table prefixes from pl.col() calls.

    This transformer finds all pl.col('table.column') calls and strips
    the table prefix, converting them to pl.col('column').
    """

    # Method names that need first positional arg stripped
    TIME_GROUPING_METHODS: Set[str] = {'group_by_dynamic', 'rolling'}

    # Keyword args in join() that need stripping
    JOIN_COLUMN_KWARGS: Set[str] = {'on', 'left_on', 'right_on'}

    def __init__(self, function_name: str):
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
        Strip table prefixes from pl.col() calls and string literals in specific contexts.

        Transforms:
            pl.col('table.column') -> pl.col('column')
            .group_by_dynamic('table.column', ...) -> .group_by_dynamic('column', ...)
            .rolling(index_column='table.column', ...) -> .rolling(index_column='column', ...)
            .sort('table.column') -> .sort('column')
            .join(on='table.column') -> .join(on='column')
        """
        if self.current_function != self.function_name:
            return updated_node

        # pl.col() call
        if self._is_pl_col_call(updated_node):
            return self._strip_first_positional_arg(updated_node)

        # group_by_dynamic() or rolling()
        if self._is_method_call(updated_node, self.TIME_GROUPING_METHODS):
            return self._strip_time_grouping_args(updated_node)

        # sort()
        if self._is_method_call(updated_node, {'sort'}):
            return self._strip_first_positional_arg(updated_node)

        # join()
        if self._is_method_call(updated_node, {'join'}):
            return self._strip_join_args(updated_node)

        return updated_node

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    def _strip_prefix_from_string(self, node: cst.SimpleString) -> Optional[cst.SimpleString]:
        """Strip table prefix from a SimpleString node if present."""
        column_name = node.value.strip('\'"')
        if '.' not in column_name:
            return None
        stripped_name = column_name.split('.', 1)[1]
        quote_char = node.value[0]
        return cst.SimpleString(f"{quote_char}{stripped_name}{quote_char}")

    def _strip_prefix_from_arg(self, arg: cst.Arg) -> cst.Arg:
        """Strip table prefixes from an argument (string or list of strings)."""
        if m.matches(arg.value, m.SimpleString()):
            new_value = self._strip_prefix_from_string(arg.value)
            if new_value:
                return arg.with_changes(value=new_value)
        elif m.matches(arg.value, m.List()):
            new_elements = []
            changed = False
            for element in arg.value.elements:
                if m.matches(element.value, m.SimpleString()):
                    new_str = self._strip_prefix_from_string(element.value)
                    if new_str:
                        new_elements.append(element.with_changes(value=new_str))
                        changed = True
                        continue
                new_elements.append(element)
            if changed:
                new_list = arg.value.with_changes(elements=new_elements)
                return arg.with_changes(value=new_list)
        return arg

    # -------------------------------------------------------------------------
    # Type checking helpers
    # -------------------------------------------------------------------------

    def _is_pl_col_call(self, node: cst.Call) -> bool:
        """Check if this is a pl.col() call."""
        return m.matches(
            node,
            m.Call(func=m.Attribute(value=m.Name('pl'), attr=m.Name('col')))
        )

    def _is_method_call(self, node: cst.Call, method_names: Set[str]) -> bool:
        """Check if node is a method call matching any of the given names."""
        if not m.matches(node, m.Call(func=m.Attribute())):
            return False
        return node.func.attr.value in method_names

    # -------------------------------------------------------------------------
    # Handler methods
    # -------------------------------------------------------------------------

    def _strip_first_positional_arg(self, node: cst.Call) -> cst.Call:
        """Strip table prefix from the first positional argument."""
        if not node.args:
            return node

        first_arg = node.args[0]
        if first_arg.keyword is not None:
            return node

        new_arg = self._strip_prefix_from_arg(first_arg)
        if new_arg is first_arg:
            return node

        return node.with_changes(args=[new_arg] + list(node.args[1:]))

    def _strip_time_grouping_args(self, node: cst.Call) -> cst.Call:
        """
        Strip table prefixes from group_by_dynamic() and rolling() calls.

        Handles:
        - group_by_dynamic('table.column', ...) -> first positional arg
        - rolling(index_column='table.column', ...) -> index_column keyword arg
        """
        new_args = []
        changed = False

        for i, arg in enumerate(node.args):
            # First positional arg for group_by_dynamic
            if arg.keyword is None and i == 0:
                new_arg = self._strip_prefix_from_arg(arg)
                if new_arg is not arg:
                    changed = True
                new_args.append(new_arg)
            # index_column keyword for rolling
            elif arg.keyword is not None and arg.keyword.value == 'index_column':
                new_arg = self._strip_prefix_from_arg(arg)
                if new_arg is not arg:
                    changed = True
                new_args.append(new_arg)
            else:
                new_args.append(arg)

        return node.with_changes(args=new_args) if changed else node

    def _strip_join_args(self, node: cst.Call) -> cst.Call:
        """
        Strip table prefixes from join() call arguments.

        Handles:
        - .join(other, on='table.column') -> .join(other, on='column')
        - .join(other, on=['table.col1', 'table.col2']) -> .join(other, on=['col1', 'col2'])
        - .join(other, left_on='table.col1', right_on='table.col2') -> strips both
        """
        new_args = []
        changed = False

        for arg in node.args:
            if arg.keyword is not None and arg.keyword.value in self.JOIN_COLUMN_KWARGS:
                new_arg = self._strip_prefix_from_arg(arg)
                if new_arg is not arg:
                    changed = True
                new_args.append(new_arg)
            else:
                new_args.append(arg)

        return node.with_changes(args=new_args) if changed else node


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
