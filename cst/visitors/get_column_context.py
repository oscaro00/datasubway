from typing import List, Dict, Optional
import libcst as cst
import libcst.matchers as m


class GetColumnContext(m.MatcherDecoratableVisitor):
    """
    Given a function name, extract all column context instances.
    Essentially, pull occurrences of Allow() and Exclude() out of polars methods.
    """

    def __init__(self, function_name: str) -> None:
        """
        Initialize the visitor to extract Allow() and Exclude() calls.

        Args:
            function_name: Single function name to analyze (e.g., 'revenue_by_item')
        """
        super().__init__()
        self.function_name = function_name
        self.allow_calls: List[Dict] = []
        self.exclude_calls: List[Dict] = []

    @m.visit(m.Call(func=m.Name("Allow")))
    def visit_allow_call(self, node: cst.Call) -> None:
        """
        Automatically called only for Call nodes where func is Name("Allow").
        """
        columns = self._extract_call_arguments(node)
        self.allow_calls.append(columns)

    @m.visit(m.Call(func=m.Name("Exclude")))
    def visit_exclude_call(self, node: cst.Call) -> None:
        """
        Automatically called only for Call nodes where func is Name("Exclude").
        """
        columns = self._extract_call_arguments(node)
        self.exclude_calls.append(columns)

    def _extract_call_arguments(self, node: cst.Call) -> Dict:
        """
        Extract both positional and keyword arguments from a Call node.

        For Allow('*', 'table2.*', use=['item_id']), this returns:
        {
            'positional': ['*', 'table2.*'],
            'use': ['item_id']
        }

        Args:
            node: The Call node to extract arguments from

        Returns:
            Dictionary with 'positional' and 'use' keys containing extracted column names
        """
        result = {
            'positional': [],
            'use': []
        }

        for arg in node.args:
            if arg.keyword is None:
                # Positional argument (no keyword=)
                value = self._extract_value(arg.value)
                if value:
                    result['positional'].append(value)

            elif m.matches(arg.keyword, m.Name("use")):
                # Keyword argument: use=[...]
                if m.matches(arg.value, m.List()):
                    list_node = cst.ensure_type(arg.value, cst.List)
                    for element in list_node.elements:
                        value = self._extract_value(element.value)
                        if value:
                            result['use'].append(value)

        return result

    def _extract_value(self, node: cst.BaseExpression) -> Optional[str]:
        """
        Extract string value from an expression using matchers.

        Args:
            node: Expression node to extract value from

        Returns:
            String value if the node is a SimpleString or Name, None otherwise
        """
        # Check if it's a SimpleString (e.g., 'item_id' or "table.*")
        if m.matches(node, m.SimpleString()):
            string_node = cst.ensure_type(node, cst.SimpleString)
            # Remove quotes from the string literal
            return string_node.value.strip('\'"')

        # Check if it's a Name (e.g., a variable reference)
        if m.matches(node, m.Name()):
            name_node = cst.ensure_type(node, cst.Name)
            return name_node.value

        # Could extend with more node types as needed (e.g., FormattedString, ConcatenatedString)
        return None
