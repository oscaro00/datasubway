"""
Visitor to extract the Allow() or Exclude() call from the last grouping method.

This module provides a libcst visitor that:
1. Finds all group_by(), group_by_dynamic(), and rolling() calls in a measure function
2. Identifies the LAST one by line number
3. Extracts the Allow() or Exclude() argument as raw source code string
4. For group_by_dynamic/rolling, merges the index_column into the include parameter
"""

from typing import Optional, List
import libcst as cst
import libcst.matchers as m


GROUP_BY_VARIANTS = {'group_by', 'group_by_dynamic', 'rolling'}


class GroupingCallInfo:
    """Represents a grouping method call with its position and extracted info."""

    def __init__(
        self,
        method_name: str,
        line: int,
        allow_exclude_node: Optional[cst.Call],
        index_column_node: Optional[cst.BaseExpression]
    ):
        self.method_name = method_name
        self.line = line
        self.allow_exclude_node = allow_exclude_node
        self.index_column_node = index_column_node


class GetLastGroupingContext(cst.CSTVisitor):
    """
    Visitor that finds the last grouping call and extracts its Allow/Exclude argument.

    The "last" grouping call is determined by line number (highest line number wins).
    """

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, target_function_name: str):
        self.target_function_name = target_function_name
        self.grouping_calls: List[GroupingCallInfo] = []
        self.current_function: Optional[str] = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track which function we're currently visiting."""
        self.current_function = node.name.value

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Reset current function when leaving."""
        if node.name.value == self.current_function:
            self.current_function = None

    def visit_Call(self, node: cst.Call) -> None:
        """Visit Call nodes to find grouping method calls."""
        if self.current_function != self.target_function_name:
            return

        if not isinstance(node.func, cst.Attribute):
            return

        method_name = node.func.attr.value

        if method_name not in GROUP_BY_VARIANTS:
            return

        line = self._get_line_number(node)
        allow_exclude_node = self._extract_allow_exclude(node, method_name)
        index_column_node = self._extract_index_column(node, method_name)

        self.grouping_calls.append(GroupingCallInfo(
            method_name, line, allow_exclude_node, index_column_node
        ))

    def _get_line_number(self, node: cst.CSTNode) -> int:
        """Get line number of node using metadata."""
        try:
            pos = self.get_metadata(cst.metadata.PositionProvider, node)
            if pos and hasattr(pos, 'start'):
                return pos.start.line
        except KeyError:
            pass
        return 0

    def _extract_allow_exclude(
        self, node: cst.Call, method_name: str
    ) -> Optional[cst.Call]:
        """
        Extract Allow() or Exclude() argument from a grouping call.

        For group_by(): First positional argument
        For group_by_dynamic/rolling(): The group_by= keyword argument
        """
        if method_name == 'group_by':
            if node.args and not node.args[0].keyword:
                first_arg = node.args[0].value
                if self._is_allow_or_exclude(first_arg):
                    return first_arg
        else:
            for arg in node.args:
                if arg.keyword and arg.keyword.value == 'group_by':
                    if self._is_allow_or_exclude(arg.value):
                        return arg.value
                    break

        return None

    def _extract_index_column(
        self, node: cst.Call, method_name: str
    ) -> Optional[cst.BaseExpression]:
        """
        Extract index_column from group_by_dynamic or rolling calls.

        For group_by(): Returns None (no index column concept)
        For group_by_dynamic/rolling(): First positional arg OR index_column= keyword
        """
        if method_name == 'group_by':
            return None

        for arg in node.args:
            if arg.keyword and arg.keyword.value == 'index_column':
                return arg.value

        if node.args and not node.args[0].keyword:
            return node.args[0].value

        return None

    def _is_allow_or_exclude(self, node: cst.BaseExpression) -> bool:
        """Check if node is an Allow() or Exclude() call."""
        return (
            m.matches(node, m.Call(func=m.Name('Allow'))) or
            m.matches(node, m.Call(func=m.Name('Exclude')))
        )

    def _node_to_source(self, node: cst.BaseExpression) -> str:
        """Convert a CST node to its source code string."""
        temp_module = cst.Module(body=[cst.Expr(value=node)])
        return temp_module.code.strip()

    def _merge_index_column_into_include(
        self,
        allow_exclude_node: cst.Call,
        index_column_node: cst.BaseExpression
    ) -> cst.Call:
        """
        Merge the index_column into the include parameter of Allow/Exclude.

        If include doesn't exist, creates include=[index_column].
        If include exists as a list, appends index_column.
        If include exists as a single value, converts to [existing, index_column].
        """
        existing_include_arg = None
        include_arg_index = None

        for i, arg in enumerate(allow_exclude_node.args):
            if arg.keyword and arg.keyword.value == 'include':
                existing_include_arg = arg
                include_arg_index = i
                break

        if existing_include_arg is None:
            new_include_arg = cst.Arg(
                keyword=cst.Name('include'),
                value=cst.List(elements=[cst.Element(value=index_column_node)]),
                equal=cst.AssignEqual(
                    whitespace_before=cst.SimpleWhitespace(''),
                    whitespace_after=cst.SimpleWhitespace('')
                )
            )
            new_args = list(allow_exclude_node.args)
            if new_args and new_args[-1].comma == cst.MaybeSentinel.DEFAULT:
                new_args[-1] = new_args[-1].with_changes(
                    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' '))
                )
            new_args.append(new_include_arg)
            return allow_exclude_node.with_changes(args=new_args)

        existing_value = existing_include_arg.value

        if isinstance(existing_value, cst.List):
            new_elements = list(existing_value.elements)
            if new_elements and new_elements[-1].comma == cst.MaybeSentinel.DEFAULT:
                new_elements[-1] = new_elements[-1].with_changes(
                    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' '))
                )
            new_elements.append(cst.Element(value=index_column_node))
            new_list = existing_value.with_changes(elements=new_elements)
            new_include_arg = existing_include_arg.with_changes(value=new_list)
        else:
            new_list = cst.List(elements=[
                cst.Element(
                    value=existing_value,
                    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(' '))
                ),
                cst.Element(value=index_column_node)
            ])
            new_include_arg = existing_include_arg.with_changes(value=new_list)

        new_args = list(allow_exclude_node.args)
        new_args[include_arg_index] = new_include_arg
        return allow_exclude_node.with_changes(args=new_args)

    def get_last_grouping_context(self) -> Optional[str]:
        """
        Get the Allow/Exclude source code from the last grouping call.

        If the last grouping call is group_by_dynamic or rolling with an index_column,
        the index_column is merged into the include parameter of Allow/Exclude.

        Returns:
            Source code string of Allow() or Exclude() call, or None if:
            - No grouping calls found
            - Last grouping call has no Allow/Exclude argument
        """
        if not self.grouping_calls:
            return None

        self.grouping_calls.sort(key=lambda x: x.line)
        last_call = self.grouping_calls[-1]

        if last_call.allow_exclude_node is None:
            return None

        result_node = last_call.allow_exclude_node

        if last_call.index_column_node is not None:
            result_node = self._merge_index_column_into_include(
                result_node, last_call.index_column_node
            )

        return self._node_to_source(result_node)


def get_last_grouping_context(
    source_code: str, function_name: str
) -> Optional[str]:
    """
    Extract the Allow/Exclude source code from the last grouping call.

    For group_by_dynamic() and rolling() calls, the index_column parameter
    is automatically merged into the include parameter of the Allow/Exclude call.

    Args:
        source_code: Python source code containing the function
        function_name: Name of the function to analyze

    Returns:
        Source code string of Allow() or Exclude() call, or None if not found.

    Example:
        >>> code = '''
        ... def my_measure(qc):
        ...     return (
        ...         dm.table('sales')
        ...         .group_by(Allow('*', context=qc.get('group', [])))
        ...         .agg(pl.col('revenue').sum())
        ...     )
        ... '''
        >>> result = get_last_grouping_context(code, 'my_measure')
        >>> print(result)
        "Allow('*', context=qc.get('group', []))"

        >>> code = '''
        ... def rolling_measure(qc):
        ...     return (
        ...         dm.table('sales')
        ...         .group_by_dynamic('sales.date', every='1d', group_by=Allow('*', context=qc.get('group')))
        ...         .agg(pl.col('revenue').mean())
        ...     )
        ... '''
        >>> result = get_last_grouping_context(code, 'rolling_measure')
        >>> print(result)
        "Allow('*', context=qc.get('group'), include=['sales.date'])"
    """
    try:
        module = cst.parse_module(source_code)
        wrapper = cst.metadata.MetadataWrapper(module)

        visitor = GetLastGroupingContext(function_name)
        wrapper.visit(visitor)

        return visitor.get_last_grouping_context()

    except Exception:
        return None
