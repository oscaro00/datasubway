"""
Transformer to replace dm.table() and self.table() calls with actual LazyFrame code.

This module provides a libcst transformer that:
1. Finds dm.table() or self.table() calls within a target function
2. Extracts the call arguments
3. Calls the DataModel.table() method to get the LazyFrame CST code
4. Replaces the call with the returned CST expression

Example:
    >>> code = '''
    ... def my_measure(qc):
    ...     return (
    ...         dm.table('sales', ['item_id'], {'revenue': 'sum'})
    ...         .group_by(['item_id'])
    ...         .agg(pl.col('revenue').sum())
    ...     )
    ... '''
    >>> # After transformation, dm.table() is replaced with:
    >>> # pl.scan_parquet(self.pre_agg_directory / 'sales_by_item.parquet')
    >>> # or: self.tables['sales']
"""

from typing import Dict, Any, Optional, Union
import libcst as cst
import libcst.matchers as m


class ReplaceTableCalls(cst.CSTTransformer):
    """
    Transforms dm.table() and self.table() calls into LazyFrame source code.

    This transformer:
    1. Finds all dm.table() or self.table() calls within a target function
    2. Evaluates the arguments
    3. Calls DataModel.table() method to get the CST expression
    4. Replaces the Call node with the returned expression
    """

    def __init__(
        self,
        function_name: str,
        runtime_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the transformer.

        Args:
            function_name: Name of the function to transform
            runtime_context: Dict containing 'dm' or 'self' (DataModel instance)
        """
        super().__init__()
        self.function_name = function_name
        self.runtime_context = runtime_context or {}
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
    ) -> Union[cst.Call, cst.BaseExpression]:
        """
        Replace dm.table() and self.table() calls with LazyFrame code.

        Steps:
        1. Check if we're in the target function
        2. Check if call is dm.table() or self.table()
        3. Extract and evaluate arguments
        4. Call DataModel.table() method
        5. Return the CST expression from table() method
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Check if this is a .table() method call
        if not self._is_table_call(updated_node):
            return updated_node

        # Get the DataModel instance from runtime context
        data_model = self.runtime_context.get('dm') or self.runtime_context.get('self')
        if data_model is None:
            # No DataModel instance available - leave unchanged
            return updated_node

        try:
            # Extract arguments
            args = self._extract_table_arguments(updated_node)
            if args is None:
                return updated_node

            original_table, group_by_cols, agg_cols, allow_pre_aggs = args

            # Call the table() method to get the CST expression
            table_expr = data_model.table(
                original_table=original_table,
                group_by_cols=group_by_cols,
                agg_cols=agg_cols,
                allow_pre_aggs=allow_pre_aggs
            )

            # Return the CST expression
            return table_expr

        except Exception as e:
            # If transformation fails, leave unchanged
            # Could add logging here for debugging
            return updated_node

    def _is_table_call(self, node: cst.Call) -> bool:
        """
        Check if this is a dm.table() or self.table() call.

        Args:
            node: Call node to check

        Returns:
            True if this is a table() method call
        """
        if not isinstance(node.func, cst.Attribute):
            return False

        # Check if attribute name is 'table'
        if node.func.attr.value != 'table':
            return False

        # Check if it's called on 'dm' or 'self'
        if isinstance(node.func.value, cst.Name):
            return node.func.value.value in ['dm', 'self']

        return False

    def _extract_table_arguments(
        self,
        node: cst.Call
    ) -> Optional[tuple[str, list[str], dict[str, str], bool]]:
        """
        Extract arguments from dm.table() call.

        Expected signature:
            dm.table(original_table, group_by_cols, agg_cols, allow_pre_aggs=True)

        Args:
            node: The dm.table() Call node

        Returns:
            Tuple of (original_table, group_by_cols, agg_cols, allow_pre_aggs)
            or None if extraction fails
        """
        try:
            # Need at least 3 positional arguments
            if len(node.args) < 3:
                return None

            # Extract positional arguments
            original_table_arg = node.args[0].value
            group_by_cols_arg = node.args[1].value
            agg_cols_arg = node.args[2].value

            # Extract optional allow_pre_aggs argument (default True)
            allow_pre_aggs = True
            if len(node.args) >= 4:
                allow_pre_aggs_arg = node.args[3].value
                allow_pre_aggs = self._evaluate_bool_arg(allow_pre_aggs_arg)
            else:
                # Check keyword arguments
                for arg in node.args:
                    if arg.keyword and arg.keyword.value == 'allow_pre_aggs':
                        allow_pre_aggs = self._evaluate_bool_arg(arg.value)

            # Evaluate arguments to Python values
            original_table = self._evaluate_string_arg(original_table_arg)
            group_by_cols = self._evaluate_list_arg(group_by_cols_arg)
            agg_cols = self._evaluate_dict_arg(agg_cols_arg)

            if original_table is None or group_by_cols is None or agg_cols is None:
                return None

            return original_table, group_by_cols, agg_cols, allow_pre_aggs

        except Exception:
            return None

    def _evaluate_string_arg(self, node: cst.BaseExpression) -> Optional[str]:
        """Evaluate a string argument."""
        if isinstance(node, cst.SimpleString):
            return node.value.strip('\'"')
        return None

    def _evaluate_list_arg(self, node: cst.BaseExpression) -> Optional[list[str]]:
        """Evaluate a list argument."""
        # Handle direct list literal
        if isinstance(node, cst.List):
            result = []
            for element in node.elements:
                if isinstance(element, cst.Element):
                    if isinstance(element.value, cst.SimpleString):
                        result.append(element.value.value.strip('\'"'))
                    else:
                        return None
            return result

        # Handle function call like qc.get('group', [])
        # We need to evaluate it
        if isinstance(node, cst.Call):
            try:
                temp_module = cst.Module(body=[cst.Expr(value=node)])
                call_code = temp_module.code.strip()
                result = eval(call_code, {}, self.runtime_context)
                if isinstance(result, list):
                    return result
            except Exception:
                pass

        return None

    def _evaluate_dict_arg(self, node: cst.BaseExpression) -> Optional[dict[str, str]]:
        """Evaluate a dict argument."""
        if isinstance(node, cst.Dict):
            result = {}
            for element in node.elements:
                if isinstance(element, cst.DictElement):
                    key_node = element.key
                    value_node = element.value

                    # Extract key
                    if isinstance(key_node, cst.SimpleString):
                        key = key_node.value.strip('\'"')
                    else:
                        return None

                    # Extract value
                    if isinstance(value_node, cst.SimpleString):
                        value = value_node.value.strip('\'"')
                    else:
                        return None

                    result[key] = value
            return result

        return None

    def _evaluate_bool_arg(self, node: cst.BaseExpression) -> bool:
        """Evaluate a boolean argument."""
        # Handle direct boolean literal
        if isinstance(node, cst.Name):
            if node.value == 'True':
                return True
            elif node.value == 'False':
                return False

        # Handle function call like qc.get('allow_pre_aggs', True)
        if isinstance(node, cst.Call):
            try:
                temp_module = cst.Module(body=[cst.Expr(value=node)])
                call_code = temp_module.code.strip()
                result = eval(call_code, {}, self.runtime_context)
                return bool(result)
            except Exception:
                pass

        # Default to True
        return True


def replace_table_calls(
    source_code: str,
    function_name: str,
    runtime_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Replace dm.table() and self.table() calls with LazyFrame source code.

    This function applies the ReplaceTableCalls transformer to replace
    table() method calls with the actual LazyFrame code (either pre-agg
    scans or table access/joins).

    Args:
        source_code: Python source code containing the function
        function_name: Name of function to transform
        runtime_context: Dict containing 'dm' or 'self' (DataModel instance)

    Returns:
        Transformed source code with table() calls replaced

    Example:
        >>> code = '''
        ... def my_measure(qc):
        ...     return dm.table('sales', ['item_id'], {'revenue': 'sum'})
        ... '''
        >>> dm = DataModel(...)
        >>> result = replace_table_calls(code, 'my_measure', {'dm': dm})
        >>> # Result: return self.tables['sales']
    """
    module = cst.parse_module(source_code)
    transformer = ReplaceTableCalls(function_name, runtime_context)
    new_module = module.visit(transformer)
    return new_module.code
